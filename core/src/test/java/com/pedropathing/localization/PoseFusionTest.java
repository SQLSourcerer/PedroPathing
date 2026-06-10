package com.pedropathing.localization;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pedropathing.geometry.Pose;

import org.junit.jupiter.api.Test;

/**
 * Tests for the host-agnostic {@link PoseFusion} estimator, driven directly through its
 * {@code predict}/{@code correct} API with explicit timestamps (no clock, no localizer). Covers the
 * three correctness fixes (fused-frame propagation, distance-scaled process noise, degenerate-variance
 * handling) and geodesic interpolation.
 */
class PoseFusionTest {
    private static final double EPS = 1e-6;

    private static double[] xyh(double x, double y, double h) { return new double[]{x, y, h}; }

    private static PoseFusion fusion(double[] initialCov, double[] processVar, double[] measVar) {
        PoseFusion f = new PoseFusion(initialCov, processVar, measVar, 1000);
        f.reset(0, 0, 0, 0);
        return f;
    }

    private static void assertPose(PoseFusion f, double x, double y, double h, double tol) {
        assertEquals(x, f.getX(), tol, "x");
        assertEquals(y, f.getY(), tol, "y");
        assertEquals(h, f.getHeading(), tol, "heading");
    }

    // ---- #1: frame correctness --------------------------------------------------------------

    /** With no measurement, the fused pose is exactly the odometry pose (SE(2) increments composed onto the same start). */
    @Test
    void noMeasurement_fusedTracksOdometryThroughAnArc() {
        PoseFusion f = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));

        double[][] path = {
                {2, 0, 0},
                {4, 1, Math.PI / 6},
                {5, 3, Math.PI / 3},
                {5, 6, Math.PI / 2},
        };
        long t = 1;
        for (double[] p : path) {
            f.predict(p[0], p[1], p[2], t++);
            assertPose(f, p[0], p[1], p[2], 1e-9);
        }
    }

    /**
     * The key fix: after a vision heading correction, subsequent forward odometry motion must move
     * the fused pose along the <em>corrected</em> heading, not the (uncorrected) odometry heading.
     */
    @Test
    void visionHeadingCorrection_redirectsSubsequentTranslation() {
        // Huge initial covariance + tiny measurement variance => the correction snaps to the measurement.
        PoseFusion f = fusion(xyh(1e6, 1e6, 1e6), xyh(1, 1, 1), xyh(EPS, EPS, EPS));

        // Two stationary updates so timestamp 1 is a valid (interior) measurement time.
        f.predict(0, 0, 0, 1);
        f.predict(0, 0, 0, 2);

        // Correct heading to +90 degrees at the origin.
        f.correct(0, 0, Math.PI / 2, 1, EPS, EPS, EPS);
        assertEquals(Math.PI / 2, f.getHeading(), 1e-3, "heading should snap to measurement");

        // Now drive 10" forward in the odometry frame (heading still 0 there).
        f.predict(10, 0, 0, 3);

        // Forward motion along the corrected 90-degree heading => +y, not +x.
        assertEquals(0, f.getX(), 1e-2, "x should not move along the stale odom heading");
        assertEquals(10, f.getY(), 1e-2, "translation should follow the corrected heading");
    }

    // ---- #2: distance-scaled process noise --------------------------------------------------

    /**
     * Stationary updates must not inflate covariance: a measurement applied after many no-motion
     * loops must produce the same correction as one applied immediately (loop-rate invariance).
     */
    @Test
    void stationaryUpdates_doNotInflateCovariance() {
        PoseFusion a = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        a.predict(0, 0, 0, 1);
        a.predict(0, 0, 0, 2);
        a.correct(1, 0, 0, 1, 1, 1, 1);

        PoseFusion b = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        for (long t = 1; t <= 50; t++) b.predict(0, 0, 0, t);
        b.correct(1, 0, 0, 1, 1, 1, 1);

        assertPose(b, a.getX(), a.getY(), a.getHeading(), 1e-9);
        // With P == R the gain is 0.5, so the estimate moves halfway to the measurement.
        assertEquals(0.5, a.getX(), 1e-9, "gain should reflect un-inflated covariance");
    }

    /**
     * Process noise scales with distance, so covering the same distance in one big step or many
     * small steps yields the same covariance, and thus the same correction from an equal measurement.
     */
    @Test
    void processNoise_isInvariantToStepCount() {
        PoseFusion a = fusion(xyh(0.5, 0.5, 0.5), xyh(1, 1, 1), xyh(2, 2, 2));
        a.predict(10, 0, 0, 1);
        a.predict(10, 0, 0, 2);
        a.correct(0, 5, 0, 1, 2, 2, 2);

        PoseFusion b = fusion(xyh(0.5, 0.5, 0.5), xyh(1, 1, 1), xyh(2, 2, 2));
        long t = 1;
        for (int i = 1; i <= 10; i++) b.predict(i, 0, 0, t++);
        b.predict(10, 0, 0, t);
        b.correct(0, 5, 0, t - 1, 2, 2, 2);

        assertPose(b, a.getX(), a.getY(), a.getHeading(), 1e-9);
    }

    // ---- #3: degenerate measurement variances -----------------------------------------------

    /** A measurement variance of 0 (and a second one) must not throw, and should still fuse. */
    @Test
    void zeroVarianceMeasurement_doesNotThrowAndFuses() {
        PoseFusion f = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        f.predict(0, 0, 0, 1);
        f.predict(0, 0, 0, 2);

        assertDoesNotThrow(() -> f.correct(3, 4, 0.5, 1, 0, 0, 0));
        // Trusted measurement => estimate snaps essentially onto it.
        assertPose(f, 3, 4, 0.5, 1e-2);

        // A second zero-variance measurement on the (now collapsed) axes must also be safe.
        assertDoesNotThrow(() -> f.correct(3, 4, 0.5, 1, 0, 0, 0));
    }

    /** NaN axes in a measurement are ignored; only the supplied axis is corrected. */
    @Test
    void partialMeasurement_onlyUpdatesSuppliedAxes() {
        PoseFusion f = fusion(xyh(1e6, 1e6, 1e6), xyh(1, 1, 1), xyh(EPS, EPS, EPS));
        f.predict(5, 3, 0, 1);
        f.predict(5, 3, 0, 2);

        // Only heading is measured (x and y are NaN).
        f.correct(Double.NaN, Double.NaN, 0.4, 1, EPS, EPS, EPS);

        assertEquals(5, f.getX(), 1e-9, "x must be untouched by a heading-only fix");
        assertEquals(3, f.getY(), 1e-9, "y must be untouched by a heading-only fix");
        assertEquals(0.4, f.getHeading(), 1e-3, "heading should track the measurement");
    }

    /** A measurement whose timestamp predates the buffer window is rejected, leaving the estimate intact. */
    @Test
    void staleMeasurement_outsideWindow_isIgnored() {
        PoseFusion f = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        f.predict(2, 0, 0, 10);

        // firstTime is 0 (reset seed); a negative timestamp is before the window.
        f.correct(100, 100, 1, -5, 1, 1, 1);
        assertPose(f, 2, 0, 0, 1e-9);
    }

    // ---- latency / gain ---------------------------------------------------------------------

    /**
     * Latency compensation: a measurement timestamped in the past corrects the historical pose, and
     * the odometry motion accumulated since then is re-applied so the correction reaches the present.
     */
    @Test
    void delayedMeasurement_correctionPropagatesToCurrentPose() {
        PoseFusion f = fusion(xyh(1e6, 1e6, 1e6), xyh(1, 1, 1), xyh(EPS, EPS, EPS));

        // Drive straight along x; the present odometry pose is (15, 0, 0).
        f.predict(5, 0, 0, 1);
        f.predict(10, 0, 0, 2);
        f.predict(15, 0, 0, 3);

        // A delayed fix says "at t=1 you were really at (5, 2, 0)".
        f.correct(5, 2, 0, 1, EPS, EPS, EPS);

        // The +2 y correction must carry through the 10" of x travel since t=1 => present ≈ (15, 2, 0).
        assertPose(f, 15, 2, 0, 1e-2);
    }

    /** A partially trusted measurement nudges the estimate part-way toward the measurement, not all the way. */
    @Test
    void measurement_gainBlendsEstimateAndMeasurement() {
        // P == 3, R == 1  =>  gain = P/(P+R) = 0.75 on each axis (no motion, so P is unchanged).
        PoseFusion f = fusion(xyh(3, 3, 3), xyh(1, 1, 1), xyh(1, 1, 1));
        f.predict(0, 0, 0, 1);
        f.predict(0, 0, 0, 2);

        f.correct(4, 8, 0, 1, 1, 1, 1);

        assertEquals(3.0, f.getX(), 1e-9, "x = 0 + 0.75*(4-0)");
        assertEquals(6.0, f.getY(), 1e-9, "y = 0 + 0.75*(8-0)");
    }

    /** {@code setPose} overrides the current estimate, and the override survives a subsequent predict. */
    @Test
    void setPose_overridesEstimateAndContinues() {
        PoseFusion f = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        f.predict(5, 0, 0, 1);
        assertPose(f, 5, 0, 0, 1e-9);

        f.setPose(20, 10, 1);
        assertPose(f, 20, 10, 1, 1e-12);

        // No odometry motion since the override => the estimate must stay put (no jump back).
        f.predict(20, 10, 1, 2);
        assertPose(f, 20, 10, 1, 1e-9);
    }

    /** {@code isNaN} reflects the fused position. */
    @Test
    void isNaN_reflectsCurrentPosition() {
        PoseFusion f = fusion(xyh(1, 1, 1), xyh(1, 1, 1), xyh(1, 1, 1));
        f.predict(3, 4, 0, 1);
        assertFalse(f.isNaN());

        f.setPose(Double.NaN, 0, 0);
        assertTrue(f.isNaN());
    }

    // ---- exp-map (geodesic) interpolation ---------------------------------------------------

    /** SE(2) exponential map: the pose reached by flowing the body twist (vx, vy, w) for unit time. */
    private static Pose expSE2(double vx, double vy, double w) {
        if (Math.abs(w) < 1e-9) return new Pose(vx, vy, w);
        double s = Math.sin(w) / w;
        double c = (1 - Math.cos(w)) / w;
        return new Pose(s * vx - c * vy, c * vx + s * vy, w);
    }

    /** SE(2) composition a ⊕ b: applies body increment {@code b} at pose {@code a}. */
    private static Pose composePose(Pose a, Pose b) {
        double cos = Math.cos(a.getHeading()), sin = Math.sin(a.getHeading());
        return new Pose(
                a.getX() + b.getX() * cos - b.getY() * sin,
                a.getY() + b.getX() * sin + b.getY() * cos,
                a.getHeading() + b.getHeading());
    }

    /**
     * A measurement timestamped mid-arc must be compared against the pose on the arc the robot
     * actually drove, not the straight chord between the two bracketing samples. Each odometry step
     * here is one constant body twist (forward while turning), so the exact mid-arc pose is known in
     * closed form. Feeding that pose as a fully-trusted measurement at the midpoint timestamp is a
     * no-op (zero residual) only if the history interpolates geodesically; a plain component lerp
     * would land on the chord, see a nonzero residual, and shove the fused pose off {@code p2}.
     */
    @Test
    void measurementMidArc_interpolatesAlongArcNotChord() {
        // One constant body twist per step: 10" forward while turning +60 degrees.
        Pose fullStep = expSE2(10, 0, Math.PI / 3);
        Pose halfStep = expSE2(5, 0, Math.PI / 6);

        Pose p1 = fullStep;                       // origin (0,0,0) ⊕ fullStep
        Pose p2 = composePose(p1, fullStep);
        Pose arcMid = composePose(p1, halfStep);  // true pose halfway (in twist) from p1 to p2
        Pose chordMid = new Pose(                 // where a straight component lerp would land
                (p1.getX() + p2.getX()) / 2,
                (p1.getY() + p2.getY()) / 2,
                arcMid.getHeading());             // heading is linear either way; only translation differs

        // Sanity: the arc bows far enough off the chord that the two interpolations are distinguishable.
        double arcChordGap = Math.hypot(arcMid.getX() - chordMid.getX(), arcMid.getY() - chordMid.getY());
        assertTrue(arcChordGap > 1.0, "arc and chord must differ enough for the test to bite");

        // Trusted vision (tiny variance) measuring the true mid-arc pose => zero residual => no-op.
        PoseFusion f = fusion(xyh(1e6, 1e6, 1e6), xyh(1, 1, 1), xyh(EPS, EPS, EPS));
        f.predict(p1.getX(), p1.getY(), p1.getHeading(), 10);
        f.predict(p2.getX(), p2.getY(), p2.getHeading(), 20);
        assertPose(f, p2.getX(), p2.getY(), p2.getHeading(), 1e-9); // no vision yet: present is exactly p2

        f.correct(arcMid.getX(), arcMid.getY(), arcMid.getHeading(), 15, EPS, EPS, EPS);
        assertPose(f, p2.getX(), p2.getY(), p2.getHeading(), 1e-4); // geodesic interp => residual 0 => still p2

        // Teeth: feeding the off-arc chord point at the same time is *not* a no-op, confirming the
        // interpolation actually follows the arc (a linear interp would have made the arc case fail).
        PoseFusion g = fusion(xyh(1e6, 1e6, 1e6), xyh(1, 1, 1), xyh(EPS, EPS, EPS));
        g.predict(p1.getX(), p1.getY(), p1.getHeading(), 10);
        g.predict(p2.getX(), p2.getY(), p2.getHeading(), 20);
        g.correct(chordMid.getX(), chordMid.getY(), chordMid.getHeading(), 15, EPS, EPS, EPS);
        double moved = Math.hypot(g.getX() - p2.getX(), g.getY() - p2.getY());
        assertTrue(moved > 0.5, "feeding the off-arc chord point should perturb the fused pose");
    }

}
