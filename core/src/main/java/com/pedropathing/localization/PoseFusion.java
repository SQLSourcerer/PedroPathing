package com.pedropathing.localization;

/**
 * Host-agnostic SE(2) pose estimator that fuses dead-reckoning odometry with absolute (e.g. vision)
 * measurements. This is pure math: it owns no clock and no sensor — callers push timestamped inputs
 * and read the fused pose back. Two symmetric inputs drive it:
 * <ul>
 *     <li>{@link #predict} — an odometry update (process step: grows the covariance, composes the
 *         motion onto the fused pose at the fused heading),</li>
 *     <li>{@link #correct} — an absolute measurement (Kalman update: shrinks the covariance).</li>
 * </ul>
 * Out-of-order measurements are supported: {@code correct} rewinds to the measurement's timestamp
 * within a bounded history and re-propagates the odometry since then, making this a fixed-lag
 * smoother. All inputs and outputs are primitives and the 3×3 linear algebra is a self-contained
 * {@code Mat3}, so the estimator depends on nothing outside the JDK — no pose, matrix, or localizer
 * type. Adapters (e.g. {@link FusionLocalizer} for PedroPathing) wire it to a concrete localizer and clock.
 */
public class PoseFusion {
    /** Floor applied to per-axis measurement variance so a "fully trusted" (variance 0) axis can't
     * freeze that axis or make the innovation covariance S = Pm + R singular. */
    private static final double MEASUREMENT_VARIANCE_FLOOR = 1e-6;
    /** Floor applied to each covariance diagonal entry after a process or measurement update, so a
     * long run of vision corrections can't drive P toward zero (an overconfident filter that then
     * ignores new vision). It only ever raises a diagonal entry — P + a non-negative diagonal stays
     * positive-definite — and, being idempotent, leaves loop-rate invariance and a stationary robot's
     * covariance untouched (the clamp is a no-op until an axis would actually collapse). */
    private static final double MIN_COVARIANCE = 1e-6;
    /** History is kept for this wall-clock window (the acceptable vision-latency budget), making the
     * budget independent of loop rate; {@code bufferSize} additionally caps the entry count. */
    private static final long BUFFER_DURATION_NANOS = 1_000_000_000L;

    // Fused state and the last odometry sample are held as raw components so the per-update hot path
    // allocates nothing; callers read primitives via getX()/getY()/getHeading().
    private double curX, curY, curH;
    private double lastOdomX, lastOdomY, lastOdomH;
    private boolean hasLastOdom;

    private Mat3 P; //State Covariance
    private final Mat3 Q; //Process Noise Covariance
    private final Mat3 R; //Measurement Noise Covariance
    private final History history;
    private final int bufferSize;

    /** Reusable [x, y, heading] output buffer for the SE(2) helpers and history interpolation reads. */
    private final double[] pose = new double[3];

    /**
     * @param initialCovariance   the initial state covariance diagonal {@code [x, y, heading]}
     * @param processVariance     the per-axis process-noise coefficients {@code [x, y, heading]},
     *                            scaled by the distance/rotation actually travelled
     *                            ({@code ΔP = R(θ)·diag(|Δx|·qₓ, |Δy|·q_y, |Δθ|·q_θ)·R(θ)ᵀ}) — units
     *                            are variance per inch / per radian, not per second²
     * @param measurementVariance the default per-axis measurement variance {@code [x, y, heading]};
     *                            each axis is floored to {@value #MEASUREMENT_VARIANCE_FLOOR}
     * @param bufferSize          the maximum number of history entries to retain (a count cap on top
     *                            of the {@code BUFFER_DURATION_NANOS} wall-clock latency window)
     */
    public PoseFusion(double[] initialCovariance, double[] processVariance, double[] measurementVariance, int bufferSize) {
        this.P = Mat3.diag(initialCovariance[0], initialCovariance[1], initialCovariance[2]);
        this.Q = Mat3.diag(processVariance[0], processVariance[1], processVariance[2]);
        this.R = Mat3.diag(measurementVariance[0], measurementVariance[1], measurementVariance[2]);
        this.bufferSize = bufferSize;
        this.history = new History(bufferSize);
    }

    /**
     * Odometry update from an absolute odometry pose reading. The core differences it against the
     * previous reading to recover the body-frame increment (zero on the first call). Symmetric with
     * {@link #correct}: both take an absolute sensor reading at time {@code t}.
     * <p>
     * Feed it whatever your localizer reports as its current pose. Both PedroPathing and Road Runner
     * accumulate odometry into an absolute pose ({@code getPose()}), so this is the entry point for both.
     */
    public void predict(double odomX, double odomY, double odomH, long t) {
        double dx = 0, dy = 0, dh = 0;
        if (hasLastOdom) {
            relativeTransform(lastOdomX, lastOdomY, lastOdomH, odomX, odomY, odomH, pose);
            dx = pose[0]; dy = pose[1]; dh = pose[2];
        }
        applyIncrement(dx, dy, dh, odomX, odomY, odomH, t);
    }

    /** Process step: grow the covariance, compose the increment onto the fused pose, append history. */
    private void applyIncrement(double dx, double dy, double dh, double odomX, double odomY, double odomH, long t) {
        // Grow the covariance, then compose the increment onto the fused pose. Translation follows the
        // *fused* heading, so vision heading corrections are honored and the step is exact in SE(2).
        addProcessNoiseInPlace(P, dx, dy, dh, curH);
        composeOnto(curX, curY, curH, dx, dy, dh, pose);
        curX = pose[0]; curY = pose[1]; curH = pose[2];

        lastOdomX = odomX; lastOdomY = odomY; lastOdomH = odomH;
        hasLastOdom = true;

        history.append(t, curX, curY, curH, odomX, odomY, odomH, P.copy());
        history.trimOlderThan(t - BUFFER_DURATION_NANOS);
        history.enforceCountCap(bufferSize);
    }

    /**
     * Absolute measurement update using the default measurement variance. Any axis passed as
     * {@link Double#NaN} is treated as unobserved and left uncorrected.
     */
    public void correct(double measX, double measY, double measH, long t) {
        applyCorrection(measX, measY, measH, t, R.copy());
    }

    /**
     * Absolute measurement update with a per-measurement variance {@code [varX, varY, varH]} (each
     * floored to {@value #MEASUREMENT_VARIANCE_FLOOR}). Any axis passed as {@link Double#NaN} is
     * treated as unobserved and left uncorrected.
     */
    public void correct(double measX, double measY, double measH, long t, double varX, double varY, double varH) {
        applyCorrection(measX, measY, measH, t, Mat3.diag(varX, varY, varH));
    }

    private void applyCorrection(double measX, double measY, double measH, long t, Mat3 measurementR) {
        // Floor variances so a "fully trusted" axis (variance 0) can't freeze the axis or make S singular.
        for (int i = 0; i < 3; i++)
            measurementR.set(i, i, Math.max(measurementR.get(i, i), MEASUREMENT_VARIANCE_FLOOR));

        // Reject if timestamp is outside our history time window
        if (history.isEmpty() || t < history.firstTime() || t > history.lastTime())
            return;

        // Fused pose at the measurement time
        double pastX, pastY, pastH;
        if (history.interpolateFusedInto(t, pose)) {
            pastX = pose[0]; pastY = pose[1]; pastH = pose[2];
        } else {
            pastX = curX; pastY = curY; pastH = curH;
        }

        // Measurement residual y = z - x (zero on unobserved axes)
        boolean useX = !Double.isNaN(measX);
        boolean useY = !Double.isNaN(measY);
        boolean useH = !Double.isNaN(measH);
        double yx = useX ? measX - pastX : 0;
        double yy = useY ? measY - pastY : 0;
        double yh = useH ? normalizeAngleSigned(measH - pastH) : 0;

        // Covariance at measurement time (floor entry: latest sample at or before the timestamp)
        Mat3 Pm = history.covAt(history.floorIndex(t));

        // Innovation covariance S = P + R
        Mat3 S = Pm.plus(measurementR);

        // Gain K = P * (P + R)^(-1); skip (don't crash) if S is singular / ill-conditioned
        Mat3 K;
        try {
            K = Pm.multiply(S.inverse());
        } catch (IllegalArgumentException | IllegalStateException e) {
            return;
        }

        // Measurement mask: zero the rows of K for unobserved axes so those states are not updated.
        if (!useX) { K.set(0, 0, 0); K.set(0, 1, 0); K.set(0, 2, 0); }
        if (!useY) { K.set(1, 0, 0); K.set(1, 1, 0); K.set(1, 2, 0); }
        if (!useH) { K.set(2, 0, 0); K.set(2, 1, 0); K.set(2, 2, 0); }

        // State update x += K·y
        double updX = pastX + K.get(0, 0) * yx + K.get(0, 1) * yy + K.get(0, 2) * yh;
        double updY = pastY + K.get(1, 0) * yx + K.get(1, 1) * yy + K.get(1, 2) * yh;
        double updH = normalizeAngle(pastH + K.get(2, 0) * yx + K.get(2, 1) * yy + K.get(2, 2) * yh);

        // Joseph-form covariance update: (I-K)·Pm·(I-K)ᵀ + K·R·Kᵀ
        Mat3 IK = Mat3.identity().minus(K);
        Mat3 updatedCovariance =
                IK.multiply(Pm).multiply(IK.transposed())
                        .plus(K.multiply(measurementR).multiply(K.transposed()));
        floorCovariance(updatedCovariance);

        // Insert (or overwrite) the corrected sample at the measurement time. The odometry pose at
        // that time is interpolated so every sample carries a full (fused, odom, covariance) row.
        boolean haveOdom = history.interpolateOdomInto(t, pose);
        double odomX = haveOdom ? pose[0] : 0;
        double odomY = haveOdom ? pose[1] : 0;
        double odomH = haveOdom ? pose[2] : 0;
        history.putCorrection(t, updX, updY, updH, odomX, odomY, odomH, updatedCovariance);

        // Re-propagate every later sample from the correction, replaying the stored odometry
        // increments (same SE(2) composition as a process step, so the correction's heading is honored).
        double prevX = updX, prevY = updY, prevH = updH;
        double prevOX = odomX, prevOY = odomY, prevOH = odomH;
        boolean prevHaveOdom = haveOdom;
        Mat3 prevCov = updatedCovariance;

        for (int i = history.floorIndex(t) + 1; i < history.size(); i++) {
            double currOX = history.odomXAt(i), currOY = history.odomYAt(i), currOH = history.odomHAt(i);

            double incX = 0, incY = 0, incH = 0;
            if (prevHaveOdom) {
                relativeTransform(prevOX, prevOY, prevOH, currOX, currOY, currOH, pose);
                incX = pose[0]; incY = pose[1]; incH = pose[2];
            }

            composeOnto(prevX, prevY, prevH, incX, incY, incH, pose);
            double nextX = pose[0], nextY = pose[1], nextH = pose[2];
            history.setFused(i, nextX, nextY, nextH);

            // Copy once (each stored row needs its own matrix), then accumulate noise in place.
            Mat3 nextCov = prevCov.copy();
            addProcessNoiseInPlace(nextCov, incX, incY, incH, prevH);
            history.setCov(i, nextCov);

            prevX = nextX; prevY = nextY; prevH = nextH;
            prevOX = currOX; prevOY = currOY; prevOH = currOH; prevHaveOdom = true;
            prevCov = nextCov;
        }

        curX = history.lastFusedX();
        curY = history.lastFusedY();
        curH = history.lastFusedH();
        P = history.lastCov().copy();

        // A new sample was inserted at the correction time; re-apply the count cap (drops the oldest)
        // so the buffer is back at <= bufferSize before the next measurement or update.
        history.enforceCountCap(bufferSize);
    }

    /**
     * Re-initializes the fused pose to {@code (x, y, h)} at time {@code t}, clearing the history. Use
     * this to (re)seed the estimator.
     */
    public void reset(double x, double y, double h, long t) {
        curX = x; curY = y; curH = h;
        lastOdomX = x; lastOdomY = y; lastOdomH = h;
        hasLastOdom = true;
        history.clear();
        history.put(t, x, y, h, x, y, h, P.copy());
    }

    /**
     * Overrides the current fused pose in place without clearing history, and re-seeds
     * the odometry baseline so the next {@link #predict} produces no jump. (Maps a localizer setPose.)
     */
    public void setPose(double x, double y, double h) {
        curX = x; curY = y; curH = h;
        lastOdomX = x; lastOdomY = y; lastOdomH = h;
        hasLastOdom = true;
        if (history.isEmpty()) {
            history.put(0L, x, y, h, x, y, h, P.copy());
        } else {
            int last = history.size() - 1;
            history.setFused(last, x, y, h);
            history.setOdom(last, x, y, h);
        }
    }

    public double getX() { return curX; }
    public double getY() { return curY; }
    public double getHeading() { return curH; }

    /** @return a copy of the current 3×3 state covariance as a row-major {@code double[3][3]}. */
    public double[][] getCovariance() {
        return new double[][]{
                {P.get(0, 0), P.get(0, 1), P.get(0, 2)},
                {P.get(1, 0), P.get(1, 1), P.get(1, 2)},
                {P.get(2, 0), P.get(2, 1), P.get(2, 2)},
        };
    }

    public boolean isNaN() {
        return Double.isNaN(curX) || Double.isNaN(curY) || Double.isNaN(curH);
    }

    /**
     * Writes the SE(2) body-frame increment that takes pose {@code from} to pose {@code to}
     * (from⁻¹ ⊕ to) into {@code out} as [Δx, Δy, Δθ].
     */
    private static void relativeTransform(double fromX, double fromY, double fromH,
                                          double toX, double toY, double toH, double[] out) {
        double co = Math.cos(fromH), si = Math.sin(fromH);
        double dx = toX - fromX, dy = toY - fromY;
        out[0] = dx * co + dy * si;
        out[1] = -dx * si + dy * co;
        out[2] = normalizeAngleSigned(toH - fromH);
    }

    /**
     * Writes the SE(2) composition {@code base ⊕ increment} into {@code out} as [x, y, heading]:
     * applies a body-frame increment at the base pose's heading.
     */
    private static void composeOnto(double baseX, double baseY, double baseH,
                                    double incX, double incY, double incH, double[] out) {
        double co = Math.cos(baseH), si = Math.sin(baseH);
        out[0] = baseX + incX * co - incY * si;
        out[1] = baseY + incX * si + incY * co;
        out[2] = normalizeAngle(baseH + incH);
    }

    /**
     * Interpolates between SE(2) poses {@code a} and {@code b} along the constant-twist geodesic at
     * {@code ratio} ∈ [0, 1], writing [x, y, heading] into {@code out}. This is {@code a ⊕ exp(ratio ·
     * log(a⁻¹ ⊕ b))}: the relative motion is taken to the Lie-algebra twist, scaled, and mapped back,
     * so the result follows the arc the robot drove between the two samples rather than the straight
     * chord. Allocation-free; reduces to plain translation lerp as the heading change goes to zero.
     */
    private static void geodesicInterpolate(double ax, double ay, double ah,
                                            double bx, double by, double bh,
                                            double ratio, double[] out) {
        double co = Math.cos(ah), si = Math.sin(ah);

        // rel = a⁻¹ ⊕ b, the body-frame relative pose as an SE(2) group element.
        double dx = bx - ax, dy = by - ay;
        double rx = dx * co + dy * si;
        double ry = -dx * si + dy * co;
        double rth = normalizeAngleSigned(bh - ah);

        // log(rel): recover the body twist (ux, uy, rth) whose flow for unit time yields rel.
        double ux, uy;
        if (Math.abs(rth) < 1e-6) {
            ux = rx; uy = ry;
        } else {
            double v = Math.sin(rth) / rth;
            double w = (1 - Math.cos(rth)) / rth;
            double denom = v * v + w * w;
            ux = (v * rx + w * ry) / denom;
            uy = (v * ry - w * rx) / denom;
        }

        // exp(ratio · twist): scale the twist and map back to a group increment.
        double sth = ratio * rth, sx = ratio * ux, sy = ratio * uy;
        double ix, iy;
        if (Math.abs(sth) < 1e-6) {
            ix = sx; iy = sy;
        } else {
            double v = Math.sin(sth) / sth;
            double w = (1 - Math.cos(sth)) / sth;
            ix = v * sx - w * sy;
            iy = w * sx + v * sy;
        }

        // a ⊕ increment.
        out[0] = ax + ix * co - iy * si;
        out[1] = ay + ix * si + iy * co;
        out[2] = normalizeAngle(ah + sth);
    }

    /**
     * Adds the process-noise contribution G·Q·Gᵀ for one odometry increment directly into
     * {@code target} (a 3×3 covariance), allocating nothing.
     * <p>
     * Instead of the loop-rate-dependent {@code Q·Δt²} of a fixed-rate model, the noise is scaled by
     * the distance/rotation actually travelled:
     * <pre>
     *     ΔP = R(θ) · diag(|Δx|·qₓ, |Δy|·q_y, |Δθ|·q_θ) · R(θ)ᵀ
     * </pre>
     * This matches how odometry drifts, is invariant to loop rate, and keeps a stationary robot's
     * covariance from inflating (|Δ| ≈ 0 ⇒ no growth).
     * <p>
     * Because G = R(θ) is a planar rotation (identity on the heading axis) and Q is diagonal, the
     * product has a closed form, so this writes the five affected entries in place rather than
     * allocating the rotation, the diagonal, and two matrix-multiply temporaries each call. The
     * factor order matches {@code G.multiply(scaledQ.multiply(G.transposed()))} entry-for-entry.
     *
     * @param target  the covariance to accumulate into (mutated)
     * @param ix      body-frame increment Δx
     * @param iy      body-frame increment Δy
     * @param ih      body-frame increment Δθ
     * @param heading the fused heading the increment is applied at, rotating Q into the world frame
     */
    private void addProcessNoiseInPlace(Mat3 target, double ix, double iy, double ih, double heading) {
        double co = Math.cos(heading);
        double si = Math.sin(heading);
        double qx = Math.abs(ix) * Q.get(0, 0);
        double qy = Math.abs(iy) * Q.get(1, 1);
        double qh = Math.abs(ih) * Q.get(2, 2);

        // scaledQ · Gᵀ = [[qx·co, qx·si, 0], [-qy·si, qy·co, 0], [0, 0, qh]]
        double m00 = qx * co, m01 = qx * si;
        double m10 = -(qy * si), m11 = qy * co;

        // G · (scaledQ · Gᵀ), top-left 2×2 (heading axis is just qh)
        double d00 = co * m00 + (-si) * m10;
        double d01 = co * m01 + (-si) * m11;
        double d10 = si * m00 + co * m10;
        double d11 = si * m01 + co * m11;

        target.set(0, 0, target.get(0, 0) + d00);
        target.set(0, 1, target.get(0, 1) + d01);
        target.set(1, 0, target.get(1, 0) + d10);
        target.set(1, 1, target.get(1, 1) + d11);
        target.set(2, 2, target.get(2, 2) + qh);

        floorCovariance(target);
    }

    /**
     * Raises any covariance diagonal entry below {@link #MIN_COVARIANCE} up to that floor, in place.
     * Only ever increases a diagonal, so P stays symmetric positive-definite and the filter can't
     * become so confident on an axis that it stops responding to new measurements.
     */
    private static void floorCovariance(Mat3 m) {
        for (int i = 0; i < 3; i++)
            if (m.get(i, i) < MIN_COVARIANCE) m.set(i, i, MIN_COVARIANCE);
    }

    /** Wraps an angle to {@code [0, 2π)}. */
    private static double normalizeAngle(double angle) {
        angle %= 2 * Math.PI;
        return angle < 0 ? angle + 2 * Math.PI : angle;
    }

    /** Wraps an angle to {@code [-π, π)}. */
    private static double normalizeAngleSigned(double angle) {
        angle = normalizeAngle(angle);
        return angle >= Math.PI ? angle - 2 * Math.PI : angle;
    }

    /**
     * Minimal self-contained 3×3 matrix (row-major, flat backing array) — just the operations the
     * Kalman update needs, so the estimator carries no external linear-algebra dependency. Every op
     * returns a fresh matrix except {@link #set}, which mutates in place.
     */
    private static final class Mat3 {
        private final double[] m; // row-major: m[r*3 + c]

        private Mat3(double[] m) { this.m = m; }

        static Mat3 diag(double a, double b, double c) {
            double[] m = new double[9];
            m[0] = a; m[4] = b; m[8] = c;
            return new Mat3(m);
        }

        static Mat3 identity() { return diag(1, 1, 1); }

        double get(int r, int c) { return m[r * 3 + c]; }
        void set(int r, int c, double v) { m[r * 3 + c] = v; }

        Mat3 copy() { return new Mat3(m.clone()); }

        Mat3 plus(Mat3 o) {
            double[] r = new double[9];
            for (int i = 0; i < 9; i++) r[i] = m[i] + o.m[i];
            return new Mat3(r);
        }

        Mat3 minus(Mat3 o) {
            double[] r = new double[9];
            for (int i = 0; i < 9; i++) r[i] = m[i] - o.m[i];
            return new Mat3(r);
        }

        Mat3 multiply(Mat3 o) {
            double[] r = new double[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    double s = 0;
                    for (int k = 0; k < 3; k++) s += m[i * 3 + k] * o.m[k * 3 + j];
                    r[i * 3 + j] = s;
                }
            return new Mat3(r);
        }

        Mat3 transposed() {
            double[] r = new double[9];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    r[j * 3 + i] = m[i * 3 + j];
            return new Mat3(r);
        }

        /** @throws IllegalArgumentException if singular, so callers' existing catch handles it. */
        Mat3 inverse() {
            double a = m[0], b = m[1], c = m[2];
            double d = m[3], e = m[4], f = m[5];
            double g = m[6], h = m[7], i = m[8];
            double A = e * i - f * h;   // cofactors along the first row
            double B = f * g - d * i;
            double C = d * h - e * g;
            double det = a * A + b * B + c * C;
            if (det == 0.0) throw new IllegalArgumentException("Mat3 not invertible");
            double inv = 1.0 / det;
            double[] r = new double[9];
            r[0] = A * inv;             r[1] = (c * h - b * i) * inv; r[2] = (b * f - c * e) * inv;
            r[3] = B * inv;             r[4] = (a * i - c * g) * inv; r[5] = (c * d - a * f) * inv;
            r[6] = C * inv;             r[7] = (b * g - a * h) * inv; r[8] = (a * e - b * d) * inv;
            return new Mat3(r);
        }
    }

    /**
     * Fixed-capacity, time-sorted ring buffer holding the (timestamp, fused pose, odometry pose,
     * covariance) history. Replaces three {@link java.util.TreeMap}s: the keys are monotonic, so
     * appends are O(1) and lookups are O(log n) binary searches — same asymptotics as the trees —
     * but with no boxed keys, no per-entry node objects, and no garbage on trim, since the backing
     * arrays are allocated once and slots are reused. Pose data is stored as primitive {@code double}
     * columns so neither storing history nor reading it back allocates pose objects.
     */
    private static final class History {
        private final long[] time;
        private final double[] fusedX, fusedY, fusedH;
        private final double[] odomX, odomY, odomH;
        private final Mat3[] cov;
        private final int capacity;
        private int head; // logical index 0 lives at array index `head`
        private int size;

        History(int maxEntries) {
            // +2 headroom absorbs a transient interior insert (a vision sample) before the count cap
            // is re-applied, so the arrays never have to grow.
            capacity = Math.max(maxEntries, 1) + 2;
            time = new long[capacity];
            fusedX = new double[capacity]; fusedY = new double[capacity]; fusedH = new double[capacity];
            odomX = new double[capacity]; odomY = new double[capacity]; odomH = new double[capacity];
            cov = new Mat3[capacity];
        }

        boolean isEmpty() { return size == 0; }
        int size() { return size; }
        long firstTime() { return time[head]; }
        long lastTime() { return time[arr(size - 1)]; }

        double odomXAt(int i) { return odomX[arr(i)]; }
        double odomYAt(int i) { return odomY[arr(i)]; }
        double odomHAt(int i) { return odomH[arr(i)]; }
        Mat3 covAt(int i) { return cov[arr(i)]; }
        double lastFusedX() { return fusedX[arr(size - 1)]; }
        double lastFusedY() { return fusedY[arr(size - 1)]; }
        double lastFusedH() { return fusedH[arr(size - 1)]; }
        Mat3 lastCov() { return cov[arr(size - 1)]; }

        void setFused(int i, double x, double y, double h) { int p = arr(i); fusedX[p] = x; fusedY[p] = y; fusedH[p] = h; }
        void setOdom(int i, double x, double y, double h) { int p = arr(i); odomX[p] = x; odomY[p] = y; odomH[p] = h; }
        void setCov(int i, Mat3 c) { cov[arr(i)] = c; }

        /** Array index backing logical index {@code i} (0 == oldest). */
        private int arr(int i) { return (head + i) % capacity; }

        private long timeAt(int i) { return time[arr(i)]; }

        /** Drops all entries, releasing covariance references. */
        void clear() {
            for (int i = 0; i < size; i++) cov[arr(i)] = null;
            head = 0;
            size = 0;
        }

        /** Appends a strictly-newer sample at the back; evicts the oldest if somehow at capacity. */
        void append(long t, double x, double y, double h, double ox, double oy, double oh, Mat3 c) {
            if (size == capacity) evictOldest();
            store(arr(size), t, x, y, h, ox, oy, oh, c);
            size++;
        }

        /** Inserts or fully overwrites a sample, keeping the buffer time-sorted. */
        void put(long t, double x, double y, double h, double ox, double oy, double oh, Mat3 c) {
            int fi = floorIndex(t);
            if (fi >= 0 && timeAt(fi) == t) {
                store(arr(fi), t, x, y, h, ox, oy, oh, c);
            } else {
                insertAt(fi + 1, t, x, y, h, ox, oy, oh, c);
            }
        }

        /**
         * Applies a vision correction at {@code t}: overwrites fused + covariance if a sample already
         * exists there (leaving its odometry pose), otherwise inserts a new full row.
         */
        void putCorrection(long t, double x, double y, double h, double ox, double oy, double oh, Mat3 c) {
            int fi = floorIndex(t);
            if (fi >= 0 && timeAt(fi) == t) {
                int p = arr(fi);
                fusedX[p] = x; fusedY[p] = y; fusedH[p] = h; cov[p] = c;
            } else {
                insertAt(fi + 1, t, x, y, h, ox, oy, oh, c);
            }
        }

        /** Largest logical index whose time is ≤ {@code ts}, or -1 if none (binary search). */
        int floorIndex(long ts) {
            int lo = 0, hi = size - 1, res = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                if (timeAt(mid) <= ts) { res = mid; lo = mid + 1; }
                else hi = mid - 1;
            }
            return res;
        }

        /** Smallest logical index whose time is ≥ {@code ts}, or {@code size} if none. */
        private int ceilingIndex(long ts) {
            int lo = 0, hi = size - 1, res = size;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                if (timeAt(mid) >= ts) { res = mid; hi = mid - 1; }
                else lo = mid + 1;
            }
            return res;
        }

        boolean interpolateFusedInto(long ts, double[] out) { return interpolate(fusedX, fusedY, fusedH, ts, out); }
        boolean interpolateOdomInto(long ts, double[] out) { return interpolate(odomX, odomY, odomH, ts, out); }

        /**
         * Interpolates a pose column at {@code ts} into {@code out}; false if out of range. Uses the
         * SE(2) exponential map (constant-twist geodesic) rather than straight component lerp, so a
         * sample taken mid-arc lands on the arc the robot actually drove, not on the chord — the
         * error a plain lerp makes grows with the heading change across the bracketing samples.
         */
        private boolean interpolate(double[] colX, double[] colY, double[] colH, long ts, double[] out) {
            int lo = floorIndex(ts);
            int hi = ceilingIndex(ts);
            if (lo < 0 || hi >= size) return false;

            int lower = arr(lo);
            if (lo == hi) {
                out[0] = colX[lower]; out[1] = colY[lower]; out[2] = colH[lower];
                return true;
            }
            int upper = arr(hi);
            double ratio = (double) (ts - time[lower]) / (time[upper] - time[lower]);

            geodesicInterpolate(colX[lower], colY[lower], colH[lower],
                    colX[upper], colY[upper], colH[upper], ratio, out);
            return true;
        }

        /** Drops samples older than the most recent sample at or before {@code cutoff}. */
        void trimOlderThan(long cutoff) {
            int floor = floorIndex(cutoff);
            for (int k = 0; k < floor; k++) evictOldest();
        }

        void enforceCountCap(int max) {
            while (size > max) evictOldest();
        }

        private void evictOldest() {
            cov[head] = null; // release the only reference type for GC
            head = (head + 1) % capacity;
            size--;
        }

        /** Writes a full row's columns at array index {@code p} (does not touch size/head). */
        private void store(int p, long t, double x, double y, double h, double ox, double oy, double oh, Mat3 c) {
            time[p] = t;
            fusedX[p] = x; fusedY[p] = y; fusedH[p] = h;
            odomX[p] = ox; odomY[p] = oy; odomH[p] = oh;
            cov[p] = c;
        }

        /**
         * Inserts a row at logical position {@code pos}, shifting the (small) suffix right by one.
         * Callers keep {@code size <= capacity - 1} before inserting (the +2 headroom), so there is
         * always room for the new row without disturbing the logical indices.
         */
        private void insertAt(int pos, long t, double x, double y, double h,
                              double ox, double oy, double oh, Mat3 c) {
            for (int i = size; i > pos; i--) {
                int dst = arr(i), src = arr(i - 1);
                store(dst, time[src], fusedX[src], fusedY[src], fusedH[src], odomX[src], odomY[src], odomH[src], cov[src]);
            }
            store(arr(pos), t, x, y, h, ox, oy, oh, c);
            size++;
        }
    }
}
