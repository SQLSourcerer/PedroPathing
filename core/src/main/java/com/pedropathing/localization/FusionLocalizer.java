package com.pedropathing.localization;

import com.pedropathing.geometry.Pose;
import com.pedropathing.math.Vector;

/**
 * PedroPathing adapter around the host-agnostic {@link PoseFusion} estimator. This class supplies
 * everything {@code PoseFusion} deliberately omits: it implements the {@link Localizer} interface so
 * it drops into a follower, owns the dead-reckoning {@code Localizer} whose increments are fused,
 * owns the clock, and translates between PedroPathing {@link Pose}s and the estimator's primitive
 * API. All of the fusion math lives in {@link PoseFusion}; this is wiring only.
 */
public class FusionLocalizer implements Localizer {
    private final Localizer deadReckoning;
    private final PoseFusion core;

    /**
     * @param deadReckoning      the underlying odometry localizer whose increments are fused
     * @param initialCovariance  the initial state covariance diagonal (x, y, heading variances)
     * @param processVariance    the per-axis process-noise coefficients (x, y, heading); scaled by the
     *                           distance/rotation actually travelled — see {@link PoseFusion}
     * @param measurementVariance the default per-axis vision measurement variance (x, y, heading)
     * @param bufferSize         the maximum number of history entries to retain
     */
    public FusionLocalizer(
            Localizer deadReckoning,
            Pose initialCovariance,
            Pose processVariance,
            Pose measurementVariance,
            int bufferSize
    ) {
        this.deadReckoning = deadReckoning;
        this.core = new PoseFusion(
                new double[]{initialCovariance.getX(), initialCovariance.getY(), initialCovariance.getHeading()},
                new double[]{processVariance.getX(), processVariance.getY(), processVariance.getHeading()},
                new double[]{measurementVariance.getX(), measurementVariance.getY(), measurementVariance.getHeading()},
                bufferSize);
        setGlitchBounds(80.0, 4.0 * Math.PI);
    }

    /**
     * Source of the monotonic clock (nanoseconds) used to timestamp odometry updates. Exposed so
     * tests can supply deterministic timestamps; production uses {@link System#nanoTime()}.
     *
     * @return the current time in nanoseconds
     */
    protected long currentTimeNanos() {
        return System.nanoTime();
    }

    @Override
    public void update() {
        deadReckoning.update();
        Pose odometryPose = deadReckoning.getPose();
        core.predict(odometryPose.getX(), odometryPose.getY(), odometryPose.getHeading(), currentTimeNanos());
    }

    /**
     * Adds a vision measurement using the default measurement variance.
     * @param measuredPose the measured position by the camera, enter NaN to a specific axis if the camera couldn't measure that axis
     * @param timestamp the timestamp of the measurement
     */
    public void addMeasurement(Pose measuredPose, long timestamp) {
        core.correct(measuredPose.getX(), measuredPose.getY(), measuredPose.getHeading(), timestamp);
    }

    /**
     * Adds a vision measurement with a custom variance for this specific measurement.
     * @param measuredPose the measured position by the camera, enter NaN to a specific axis if the camera couldn't measure that axis
     * @param timestamp the timestamp of the measurement
     * @param measurementVariance the variance for this specific measurement (x, y, heading), or null to use the default
     */
    public void addMeasurement(Pose measuredPose, long timestamp, Pose measurementVariance) {
        if (measurementVariance == null) {
            core.correct(measuredPose.getX(), measuredPose.getY(), measuredPose.getHeading(), timestamp);
        } else {
            core.correct(measuredPose.getX(), measuredPose.getY(), measuredPose.getHeading(), timestamp,
                    measurementVariance.getX(), measurementVariance.getY(), measurementVariance.getHeading());
        }
    }

    /**
     * Enables the odometry glitch guard's magnitude clamp (off by default). Bounds are in the
     * localizer's units — typically inches/second and radians/second. Non-finite odometry samples are
     * always rejected regardless of this setting.
     */
    public void setGlitchBounds(double maxLinearVel, double maxAngularVel) {
        core.setGlitchBounds(maxLinearVel, maxAngularVel);
    }

    /** @see PoseFusion#getRejectedSamples() */
    public long getRejectedSamples() { return core.getRejectedSamples(); }

    /** @see PoseFusion#getClampedSamples() */
    public long getClampedSamples() { return core.getClampedSamples(); }

    /** @see PoseFusion#getRejectedMeasurements() */
    public long getRejectedMeasurements() { return core.getRejectedMeasurements(); }

    /** @see PoseFusion#getCovarianceTrace() */
    public double getCovarianceTrace() { return core.getCovarianceTrace(); }

    @Override
    public Pose getPose() { return new Pose(core.getX(), core.getY(), core.getHeading()); }

    @Override
    public Pose getVelocity() { return deadReckoning.getVelocity(); }

    @Override
    public Vector getVelocityVector() { return deadReckoning.getVelocity().getAsVector(); }

    @Override
    public void setStartPose(Pose setStart) {
        deadReckoning.setStartPose(setStart);
        core.reset(setStart.getX(), setStart.getY(), setStart.getHeading(), 0L);
    }

    @Override
    public void setPose(Pose setPose) {
        deadReckoning.setPose(setPose);
        core.setPose(setPose.getX(), setPose.getY(), setPose.getHeading());
    }

    @Override
    public double getTotalHeading() { return core.getHeading(); }

    @Override
    public double getForwardMultiplier() { return deadReckoning.getForwardMultiplier(); }

    @Override
    public double getLateralMultiplier() { return deadReckoning.getLateralMultiplier(); }

    @Override
    public double getTurningMultiplier() { return deadReckoning.getTurningMultiplier(); }

    @Override
    public void resetIMU() throws InterruptedException { deadReckoning.resetIMU(); }

    @Override
    public double getIMUHeading() { return deadReckoning.getIMUHeading(); }

    @Override
    public boolean isNAN() { return core.isNaN(); }

    @Override
    public double getAngularVelocity() { return deadReckoning.getAngularVelocity(); }
}
