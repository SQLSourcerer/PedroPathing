package com.pedropathing.localization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pedropathing.geometry.Pose;
import com.pedropathing.math.Vector;

import org.junit.jupiter.api.Test;

/**
 * Tests for the PedroPathing adapter {@link FusionLocalizer} — the wiring around {@link PoseFusion}:
 * forwarding odometry/vision into the core, materializing {@link Pose}s on the way out, passing
 * velocity and the delegating getters straight through to the dead-reckoning localizer. The fusion
 * math itself is covered by {@link PoseFusionTest}.
 */
class FusionLocalizerTest {
    private static final double EPS = 1e-6;

    /** A {@link Localizer} whose pose/velocity are set directly by the test. */
    private static class FakeLocalizer implements Localizer {
        private Pose pose = new Pose();
        private Pose velocity = new Pose();
        // Distinct constants so delegation through FusionLocalizer is observable.
        double imuHeading = 0;
        double angularVelocity = 0;

        void set(Pose pose) { this.pose = pose; }
        void setVelocity(Pose velocity) { this.velocity = velocity; }

        @Override public void update() { }
        @Override public Pose getPose() { return pose; }
        @Override public Pose getVelocity() { return velocity; }
        @Override public Vector getVelocityVector() { return velocity.getAsVector(); }
        @Override public void setStartPose(Pose setStart) { pose = setStart; }
        @Override public void setPose(Pose setPose) { pose = setPose; }
        @Override public double getTotalHeading() { return pose.getHeading(); }
        @Override public double getForwardMultiplier() { return 2; }
        @Override public double getLateralMultiplier() { return 3; }
        @Override public double getTurningMultiplier() { return 4; }
        @Override public void resetIMU() { }
        @Override public double getIMUHeading() { return imuHeading; }
        @Override public boolean isNAN() { return false; }
        @Override public double getAngularVelocity() { return angularVelocity; }
    }

    /** A {@link FusionLocalizer} with a deterministic, test-driven clock. */
    private static class TestableFusion extends FusionLocalizer {
        long clock = 0;
        TestableFusion(Localizer dr, Pose initialCov, Pose processVar, Pose measVar, int buffer) {
            super(dr, initialCov, processVar, measVar, buffer);
        }
        @Override protected long currentTimeNanos() { return clock; }
    }

    /** Advances the odometry to {@code odomPose} and runs one fused update at time {@code t}. */
    private static void step(TestableFusion fused, FakeLocalizer odom, long t, Pose odomPose) {
        odom.set(odomPose);
        fused.clock = t;
        fused.update();
    }

    private static void assertPoseEquals(Pose expected, Pose actual, double tol) {
        assertEquals(expected.getX(), actual.getX(), tol, "x");
        assertEquals(expected.getY(), actual.getY(), tol, "y");
        assertEquals(expected.getHeading(), actual.getHeading(), tol, "heading");
    }

    /** {@code update} forwards the odometry pose into the core; {@code getPose} materializes it back out. */
    @Test
    void update_forwardsOdometryIntoCore() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1, 1, 1), new Pose(1, 1, 1), new Pose(1, 1, 1), 1000);
        fused.setStartPose(new Pose(0, 0, 0));

        step(fused, odom, 1, new Pose(2, 0, 0));
        step(fused, odom, 2, new Pose(4, 1, Math.PI / 6));
        assertPoseEquals(new Pose(4, 1, Math.PI / 6), fused.getPose(), 1e-9);
    }

    /**
     * Each {@code addMeasurement} overload forwards vision into the core. With a tiny variance a fix
     * snaps the estimate onto the measurement; a fresh instance per overload keeps the (now collapsed)
     * covariance from one fix out of the next.
     */
    @Test
    void addMeasurement_forwardsVisionIntoCore() {
        Pose fix = new Pose(3, 4, 0.5);
        assertSnapsTo(fix, fused -> fused.addMeasurement(fix, 1));                          // default variance
        assertSnapsTo(fix, fused -> fused.addMeasurement(fix, 1, null));                    // null => default variance
        assertSnapsTo(fix, fused -> fused.addMeasurement(fix, 1, new Pose(EPS, EPS, EPS))); // explicit variance
    }

    /** Builds a fused localizer with a tiny default variance, runs {@code measure}, and asserts it snapped to {@code expected}. */
    private static void assertSnapsTo(Pose expected, java.util.function.Consumer<TestableFusion> measure) {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1e6, 1e6, 1e6), new Pose(1, 1, 1), new Pose(EPS, EPS, EPS), 1000);
        fused.setStartPose(new Pose(0, 0, 0));
        step(fused, odom, 1, new Pose(0, 0, 0));
        step(fused, odom, 2, new Pose(0, 0, 0));

        measure.accept(fused);
        assertPoseEquals(expected, fused.getPose(), 1e-2);
    }

    /** Velocity is passed straight through from the dead-reckoning localizer. */
    @Test
    void getVelocity_reflectsDeadReckoningVelocity() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1, 1, 1), new Pose(1, 1, 1), new Pose(1, 1, 1), 1000);
        fused.setStartPose(new Pose(0, 0, 0));

        odom.setVelocity(new Pose(1, 2, 0.5));
        step(fused, odom, 1, new Pose(0, 0, 0));

        assertPoseEquals(new Pose(1, 2, 0.5), fused.getVelocity(), 1e-9);
        Vector v = fused.getVelocityVector();
        assertEquals(1, v.getXComponent(), 1e-9);
        assertEquals(2, v.getYComponent(), 1e-9);
    }

    /** Pass-through getters delegate to the underlying dead-reckoning localizer. */
    @Test
    void delegatingGetters_passThroughToDeadReckoning() {
        FakeLocalizer odom = new FakeLocalizer();
        odom.imuHeading = 0.7;
        odom.angularVelocity = 1.5;
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1, 1, 1), new Pose(1, 1, 1), new Pose(1, 1, 1), 1000);
        fused.setStartPose(new Pose(0, 0, 0));

        assertEquals(2, fused.getForwardMultiplier(), 0);
        assertEquals(3, fused.getLateralMultiplier(), 0);
        assertEquals(4, fused.getTurningMultiplier(), 0);
        assertEquals(0.7, fused.getIMUHeading(), 0);
        assertEquals(1.5, fused.getAngularVelocity(), 0);
    }

    /** {@code setPose} overrides the estimate, and the override survives a subsequent update. */
    @Test
    void setPose_overridesEstimateAndContinues() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1, 1, 1), new Pose(1, 1, 1), new Pose(1, 1, 1), 1000);
        fused.setStartPose(new Pose(0, 0, 0));
        step(fused, odom, 1, new Pose(5, 0, 0));
        assertPoseEquals(new Pose(5, 0, 0), fused.getPose(), 1e-9);

        fused.setPose(new Pose(20, 10, 1));
        assertPoseEquals(new Pose(20, 10, 1), fused.getPose(), 1e-12);

        // No odometry motion since the reset => the estimate must stay put (no jump back to the old pose).
        step(fused, odom, 2, new Pose(20, 10, 1));
        assertPoseEquals(new Pose(20, 10, 1), fused.getPose(), 1e-9);
    }

    /** {@code isNAN} reflects the fused position. */
    @Test
    void isNAN_reflectsCurrentPosition() {
        FakeLocalizer odom = new FakeLocalizer();
        TestableFusion fused = new TestableFusion(odom,
                new Pose(1, 1, 1), new Pose(1, 1, 1), new Pose(1, 1, 1), 1000);
        fused.setStartPose(new Pose(0, 0, 0));
        step(fused, odom, 1, new Pose(3, 4, 0));
        assertFalse(fused.isNAN());

        fused.setPose(new Pose(Double.NaN, 0, 0));
        assertTrue(fused.isNAN());
    }
}
