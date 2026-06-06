package com.pedropathing.localization;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.pedropathing.geometry.Pose;
import com.pedropathing.math.Vector;

import org.junit.jupiter.api.Test;

/**
 * Correctness / regression tests for {@link FusionLocalizer}'s Kalman fusion, complementing the
 * behavior tests in {@link FusionLocalizerTest}. These pin the covariance-side properties that are
 * easy to regress:
 *
 * <ul>
 *     <li>{@link #processNoise_accumulatesWithTravel}: process noise actually accumulates into P as
 *         the robot travels, so a measurement taken after motion is trusted.</li>
 *     <li>{@link #delayedMeasurement_reducesCurrentCovariance}: a delayed measurement's covariance
 *         reduction is propagated forward to the present (checked via the gain of a follow-up
 *         measurement), not dropped.</li>
 *     <li>{@link #processNoise_growsAlongTravelDirection}: process noise grows along the world axis
 *         the robot actually travelled (body-frame Q rotated into the world), not a transposed one.</li>
 *     <li>{@link #processNoise_addsNoUncertaintyPerpendicularToTravel}: straight-line travel adds no
 *         uncertainty perpendicular to the travel direction (the growth is rank-1 along travel), so a
 *         purely perpendicular vision fix is ignored.</li>
 *     <li>{@link #processNoise_usesStartHeadingForNoiseRotation}: the noise is rotated by the heading
 *         the robot was travelling at (start of the step), so travel along one axis does not leak
 *         uncertainty onto an un-travelled axis.</li>
 *     <li>{@link #degenerateMeasurementVariance_isFlooredNotThrown}: a zero measurement variance is
 *         floored so {@code S = P + R} stays invertible instead of throwing.</li>
 * </ul>
 *
 * <p>Process noise here scales with the distance/rotation actually travelled (not velocity x dt), so
 * the motion tests drive real odometry increments rather than setting a velocity.
 */
class FusionLocalizerBugTest {

    /** A {@link Localizer} whose pose/velocity are set directly by the test. */
    private static class FakeLocalizer implements Localizer {
        private Pose pose = new Pose();
        private Pose velocity = new Pose();

        void set(Pose pose) { this.pose = pose; }
        void setVelocity(Pose velocity) { this.velocity = velocity; }

        @Override public void update() { }
        @Override public Pose getPose() { return pose; }
        @Override public Pose getVelocity() { return velocity; }
        @Override public Vector getVelocityVector() { return velocity.getAsVector(); }
        @Override public void setStartPose(Pose setStart) { pose = setStart; }
        @Override public void setPose(Pose setPose) { pose = setPose; }
        @Override public double getTotalHeading() { return pose.getHeading(); }
        @Override public double getForwardMultiplier() { return 1; }
        @Override public double getLateralMultiplier() { return 1; }
        @Override public double getTurningMultiplier() { return 1; }
        @Override public void resetIMU() { }
        @Override public double getIMUHeading() { return pose.getHeading(); }
        @Override public boolean isNAN() { return false; }
        @Override public double getAngularVelocity() { return velocity.getHeading(); }
    }

    /** A {@link FusionLocalizer} with a deterministic, test-driven clock. */
    private static class TestableFusion extends FusionLocalizer {
        long clock = 0;
        TestableFusion(Localizer dr, Pose initialCov, Pose processVar, Pose measVar, int buffer) {
            super(dr, initialCov, processVar, measVar, buffer);
        }
        @Override protected long currentTimeNanos() { return clock; }
    }

    private static final long SECOND = 1_000_000_000L;
    private static final long MILLI = 1_000_000L;

    /** Advances the odometry to {@code odomPose} and runs one fused update at time {@code t}. */
    private static void step(TestableFusion fused, FakeLocalizer odom, long t, Pose odomPose) {
        odom.set(odomPose);
        fused.clock = t;
        fused.update();
    }

    // ---- process noise accumulates ----------------------------------------------------------

    /**
     * After the robot travels with a non-zero process variance the state covariance must grow, so a
     * vision measurement at the current time is largely trusted (gain near 1) and the fused pose snaps
     * toward the measurement. If process noise never accumulated, the gain would be ~0 and the
     * estimate would stay at the odometry pose (~30) instead of moving to the measurement (50).
     */
    @Test
    void processNoise_accumulatesWithTravel() {
        FakeLocalizer odom = new FakeLocalizer();
        // Tiny initial covariance: any trust in the measurement must come from accumulated process noise.
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1e-9, 1e-9, 1e-9),   // initial covariance ~ 0
                new Pose(10, 10, 10),         // process variance (per inch / per radian travelled)
                new Pose(1, 1, 1),            // measurement variance R = 1
                1000);
        fused.setStartPose(new Pose(0, 0, 0));

        // Drive straight: 0 -> 10 -> 20 -> 30 inches in x.
        step(fused, odom, 1 * SECOND, new Pose(10, 0, 0));
        step(fused, odom, 2 * SECOND, new Pose(20, 0, 0));
        step(fused, odom, 3 * SECOND, new Pose(30, 0, 0));
        assertEquals(30, fused.getPose().getX(), 1e-9, "sanity: no-vision pose tracks odometry");

        // Measure x = 50 at the current time. With P grown well above R, the gain is ~1.
        fused.addMeasurement(new Pose(50, 0, 0), 3 * SECOND, new Pose(1, 1, 1));

        assertEquals(50, fused.getPose().getX(), 5.0,
                "post-travel measurement should be trusted; covariance must grow with travel");
    }

    // ---- delayed measurement reduces the present covariance ----------------------------------

    /**
     * A delayed measurement reduces the covariance at the correction time, and that reduction must
     * propagate to the present (there is no motion here, so nothing re-inflates it). A second
     * measurement at the present time then sees the reduced covariance and applies a smaller gain.
     *
     * <pre>
     *   first  measurement: x 0 -> 6   (gain 0.5 of residual 12), covariance 100 -> 50
     *   second measurement: P = 50 => gain 50/150 = 1/3 => x = 6 + 1/3*(12-6) = 8
     * </pre>
     * If the corrected covariance were not propagated forward (present P left at 100), the second gain
     * would be 0.5 and the estimate would land at 9 instead of 8.
     */
    @Test
    void delayedMeasurement_reducesCurrentCovariance() {
        FakeLocalizer odom = new FakeLocalizer();
        // No motion => process noise is zero, isolating the covariance-propagation behaviour.
        TestableFusion fused = new TestableFusion(odom,
                new Pose(100, 100, 100),   // initial covariance P
                new Pose(1, 1, 1),         // process variance (unused: no motion)
                new Pose(100, 100, 100),   // default measurement variance
                1000);
        fused.setStartPose(new Pose(0, 0, 0));

        // Three stationary samples so t1 is an interior correction time with later samples after it.
        // Timestamps are packed well inside the history window so the delayed sample is not trimmed.
        long t1 = 1 * MILLI, t2 = 2 * MILLI, t3 = 3 * MILLI;
        step(fused, odom, t1, new Pose(0, 0, 0));
        step(fused, odom, t2, new Pose(0, 0, 0));
        step(fused, odom, t3, new Pose(0, 0, 0));

        // Delayed correction at t1: x 0 -> 6, covariance on x 100 -> 50.
        fused.addMeasurement(new Pose(12, 0, 0), t1, new Pose(100, 100, 100));
        assertEquals(6, fused.getPose().getX(), 1e-9, "sanity: first correction moves x halfway to 12");

        // Second measurement at the present time; its gain reveals the present covariance.
        fused.addMeasurement(new Pose(12, 0, 0), t3, new Pose(100, 100, 100));

        assertEquals(8, fused.getPose().getX(), 1e-6,
                "present covariance should have been reduced to 50 by the delayed correction (gain 1/3)");
    }

    // ---- process noise grows along the travel direction -------------------------------------

    /**
     * Process noise must grow along the world axis the robot actually travelled. Driving straight
     * forward at heading +90 degrees moves the robot along world +Y, so the covariance must grow in
     * world Y, leaving world X tiny: a Y measurement is trusted while an X measurement is ignored.
     *
     * <p>Two instances driven identically probe one world axis each so their corrections stay
     * independent.
     */
    @Test
    void processNoise_growsAlongTravelDirection() {
        Pose initialCov = new Pose(1e-9, 1e-9, 1e-9); // trust must come only from accumulated noise
        Pose processVar = new Pose(10, 10, 1);

        // --- Probe world Y: it lies along the travel (body-forward) axis, so it should be trusted. ---
        FakeLocalizer odomY = new FakeLocalizer();
        TestableFusion fusedY = new TestableFusion(odomY, initialCov, processVar, new Pose(1, 1, 1), 1000);
        fusedY.setStartPose(new Pose(0, 0, Math.PI / 2));
        step(fusedY, odomY, 1 * SECOND, new Pose(0, 10, Math.PI / 2));
        step(fusedY, odomY, 2 * SECOND, new Pose(0, 20, Math.PI / 2));
        step(fusedY, odomY, 3 * SECOND, new Pose(0, 30, Math.PI / 2));
        assertEquals(30, fusedY.getPose().getY(), 1e-9, "sanity: no-vision pose tracks odometry in Y");

        fusedY.addMeasurement(new Pose(Double.NaN, 40, Double.NaN), 3 * SECOND, new Pose(1, 1, 1));
        assertEquals(40, fusedY.getPose().getY(), 2.0,
                "covariance must grow along world Y (the travel axis), so a Y measurement is trusted");

        // --- Probe world X: perpendicular to travel, so it should be ignored. ---
        FakeLocalizer odomX = new FakeLocalizer();
        TestableFusion fusedX = new TestableFusion(odomX, initialCov, processVar, new Pose(1, 1, 1), 1000);
        fusedX.setStartPose(new Pose(0, 0, Math.PI / 2));
        step(fusedX, odomX, 1 * SECOND, new Pose(0, 10, Math.PI / 2));
        step(fusedX, odomX, 2 * SECOND, new Pose(0, 20, Math.PI / 2));
        step(fusedX, odomX, 3 * SECOND, new Pose(0, 30, Math.PI / 2));

        fusedX.addMeasurement(new Pose(5, Double.NaN, Double.NaN), 3 * SECOND, new Pose(1, 1, 1));
        assertEquals(0, fusedX.getPose().getX(), 0.5,
                "world X (perpendicular to travel) must stay tiny, so an X measurement is ignored");
    }

    // ---- no uncertainty perpendicular to straight-line travel -------------------------------

    /**
     * Straight-line travel grows the covariance only along the travel direction (a rank-1 ellipse),
     * adding no uncertainty perpendicular to it. Driving forward at +45 degrees and then applying a
     * position measurement offset purely <em>perpendicular</em> to travel must therefore leave the
     * estimate put: the perpendicular residual lies in the covariance null space.
     *
     * <p>(A world-axis-isotropic model would instead grow x and y equally and snap the estimate onto
     * the perpendicular measurement near (15 - s, 15 + s); this asserts the body-frame behaviour.)
     */
    @Test
    void processNoise_addsNoUncertaintyPerpendicularToTravel() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1e-9, 1e-9, 1e-9),  // trust must come only from accumulated noise
                new Pose(10, 10, 1),
                new Pose(1, 1, 1),
                1000);
        fused.setStartPose(new Pose(0, 0, Math.PI / 4));

        // Drive forward at 45deg: world pose advances equally in x and y.
        step(fused, odom, 1 * SECOND, new Pose(5, 5, Math.PI / 4));
        step(fused, odom, 2 * SECOND, new Pose(10, 10, Math.PI / 4));
        step(fused, odom, 3 * SECOND, new Pose(15, 15, Math.PI / 4));
        assertEquals(15, fused.getPose().getX(), 1e-9, "sanity: no-vision pose tracks odometry");
        assertEquals(15, fused.getPose().getY(), 1e-9, "sanity: no-vision pose tracks odometry");

        // Measurement offset by 10" purely perpendicular to the 45deg travel: along (-1, 1)/sqrt(2).
        double s = 10.0 / Math.sqrt(2.0);
        fused.addMeasurement(new Pose(15 - s, 15 + s, Double.NaN), 3 * SECOND, new Pose(1, 1, 1));

        // Rank-1 growth along travel => the perpendicular residual is in the null space => no movement.
        assertEquals(15, fused.getPose().getX(), 0.3,
                "no perpendicular uncertainty was added, so a perpendicular fix is ignored in x");
        assertEquals(15, fused.getPose().getY(), 0.3,
                "no perpendicular uncertainty was added, so a perpendicular fix is ignored in y");
    }

    // ---- noise is rotated by the start (travel) heading -------------------------------------

    /**
     * The process-noise rotation must use the heading the robot was actually travelling at (the start
     * of the step), not the post-step (end) heading. In a single step the robot translates purely
     * along world +x while its heading turns to +45 degrees; the uncertainty must land on world x,
     * leaving world y near zero.
     *
     * <p>Rotating the noise by the end heading instead would smear it across world x and y, inflating
     * the world-y variance and causing a spurious world-y vision correction.
     */
    @Test
    void processNoise_usesStartHeadingForNoiseRotation() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1e-9, 1e-9, 1e-9),
                new Pose(10, 10, 1),
                new Pose(1, 1, 1),
                1000);
        fused.setStartPose(new Pose(0, 0, 0));

        // One step: translate 10" along world +x (the start-heading body-forward axis) while turning
        // to +45deg. The translational noise must be rotated by the start heading (0), i.e. into world x.
        step(fused, odom, 1 * SECOND, new Pose(10, 0, Math.PI / 4));
        assertEquals(10, fused.getPose().getX(), 1e-9, "sanity: pose advanced along world x");

        // Probe world y: no travel occurred there, so its variance must stay tiny and a y fix ignored.
        fused.addMeasurement(new Pose(Double.NaN, 10, Double.NaN), 1 * SECOND, new Pose(1, 1, 1));
        assertEquals(0, fused.getPose().getY(), 0.1,
                "world y carried no travel, so its variance must stay tiny and a y measurement is ignored");
    }

    // ---- degenerate measurement variance is floored, not thrown -----------------------------

    /**
     * A measurement variance of 0 on an axis whose covariance is also 0 would make the innovation
     * covariance {@code S = P + R} singular and crash the inversion. The filter floors the per-axis
     * variance so the call completes safely; this confirms it does not throw.
     *
     * <p>Heading is given initial covariance 0 and measured with variance 0 to force the singular axis.
     */
    @Test
    void degenerateMeasurementVariance_isFlooredNotThrown() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(100, 100, 0),   // heading covariance 0
                new Pose(1, 1, 1),
                new Pose(1, 1, 1),
                1000);
        fused.setStartPose(new Pose(0, 0, 0));
        step(fused, odom, 1 * SECOND, new Pose(0, 0, 0));
        step(fused, odom, 2 * SECOND, new Pose(0, 0, 0));

        // Heading measured with variance 0 while its covariance is 0 => S would be singular on heading.
        assertDoesNotThrow(() ->
                fused.addMeasurement(new Pose(Double.NaN, Double.NaN, 0.5), 1 * SECOND, new Pose(1, 1, 0)),
                "a degenerate (zero) measurement variance must be floored, not crash the filter");
    }
}
