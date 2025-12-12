"""
Extended Kalman Filter for fusing dynamics model prediction with NPE visual estimates.

Prediction: Uses drone dynamics model (with optional noise for realism)
Observation: NPE visual pose estimation [x, y, z, yaw]
"""
import numpy as np


class DroneEKF:
    """
    EKF for drone state estimation.
    
    State vector (7D): [px, py, pz, vx, vy, vz, yaw]
    Control input (4D): [ax, ay, az, yaw_rate] (body frame)
    Observation (4D): [x, y, z, yaw] from NPE
    """
    
    def __init__(self, 
                 process_noise_pos=0.01,
                 process_noise_vel=0.05,
                 process_noise_yaw=0.02,
                 obs_noise_pos=0.12,      # ~12cm NPE position noise
                 obs_noise_yaw=0.05,      # ~3° NPE yaw noise
                 dynamics_noise_enabled=False,
                 dynamics_noise_accel=0.5,    # m/s² acceleration noise
                 dynamics_noise_yaw_rate=0.1  # rad/s yaw rate noise
                 ):
        """
        Initialize EKF.
        
        Args:
            process_noise_*: Process noise standard deviations
            obs_noise_*: Observation noise standard deviations (based on NPE error)
            dynamics_noise_enabled: If True, add noise to dynamics model to simulate real world
            dynamics_noise_accel: Acceleration noise std when dynamics_noise_enabled
            dynamics_noise_yaw_rate: Yaw rate noise std when dynamics_noise_enabled
        """
        # State: [px, py, pz, vx, vy, vz, yaw]
        self.x = np.zeros(7)
        self.P = np.eye(7) * 0.5  # Initial covariance
        
        # Process noise covariance Q
        self.Q = np.diag([
            process_noise_pos**2,   # px
            process_noise_pos**2,   # py
            process_noise_pos**2,   # pz
            process_noise_vel**2,   # vx
            process_noise_vel**2,   # vy
            process_noise_vel**2,   # vz
            process_noise_yaw**2    # yaw
        ])
        
        # Observation noise covariance R (based on NPE accuracy)
        self.R = np.diag([
            obs_noise_pos**2,   # x
            obs_noise_pos**2,   # y
            obs_noise_pos**2,   # z
            obs_noise_yaw**2    # yaw
        ])
        
        # Observation matrix H: observe [x, y, z, yaw]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0],  # z
            [0, 0, 0, 0, 0, 0, 1],  # yaw
        ])
        
        # Dynamics noise settings
        self.dynamics_noise_enabled = dynamics_noise_enabled
        self.dynamics_noise_accel = dynamics_noise_accel
        self.dynamics_noise_yaw_rate = dynamics_noise_yaw_rate
        
        self.initialized = False
        
    def initialize(self, x, y, z, yaw, vx=0, vy=0, vz=0):
        """Initialize state with known values."""
        self.x = np.array([x, y, z, vx, vy, vz, yaw])
        self.P = np.eye(7) * 0.1  # Reset covariance
        self.initialized = True
        
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _dynamics(self, x, u, dt):
        """
        Drone dynamics model.
        
        Args:
            x: State [px, py, pz, vx, vy, vz, yaw]
            u: Control [ax, ay, az, yaw_rate] (body frame)
            dt: Time step
            
        Returns:
            New state
        """
        px, py, pz, vx, vy, vz, yaw = x
        ax_body, ay_body, az_body, yaw_rate = u
        
        # Add optional dynamics noise (simulates real-world perturbations)
        if self.dynamics_noise_enabled:
            ax_body += np.random.normal(0, self.dynamics_noise_accel)
            ay_body += np.random.normal(0, self.dynamics_noise_accel)
            az_body += np.random.normal(0, self.dynamics_noise_accel * 0.5)  # Less Z noise
            yaw_rate += np.random.normal(0, self.dynamics_noise_yaw_rate)
        
        # Transform body acceleration to world frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        ax_world = ax_body * cos_yaw - ay_body * sin_yaw
        ay_world = ax_body * sin_yaw + ay_body * cos_yaw
        az_world = az_body
        
        # Update state
        new_px = px + vx * dt + 0.5 * ax_world * dt**2
        new_py = py + vy * dt + 0.5 * ay_world * dt**2
        new_pz = pz + vz * dt + 0.5 * az_world * dt**2
        new_vx = vx + ax_world * dt
        new_vy = vy + ay_world * dt
        new_vz = vz + az_world * dt
        new_yaw = self._wrap_angle(yaw + yaw_rate * dt)
        
        return np.array([new_px, new_py, new_pz, new_vx, new_vy, new_vz, new_yaw])
    
    def _compute_F(self, x, u, dt):
        """
        Compute state transition Jacobian F = ∂f/∂x
        """
        yaw = x[6]
        ax_body, ay_body = u[0], u[1]
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Jacobian of dynamics w.r.t. state
        F = np.eye(7)
        
        # Position derivatives w.r.t. velocity
        F[0, 3] = dt  # ∂px/∂vx
        F[1, 4] = dt  # ∂py/∂vy
        F[2, 5] = dt  # ∂pz/∂vz
        
        # Position derivatives w.r.t. yaw (through acceleration transform)
        dax_dyaw = -ax_body * sin_yaw - ay_body * cos_yaw
        day_dyaw = ax_body * cos_yaw - ay_body * sin_yaw
        F[0, 6] = 0.5 * dax_dyaw * dt**2
        F[1, 6] = 0.5 * day_dyaw * dt**2
        
        # Velocity derivatives w.r.t. yaw
        F[3, 6] = dax_dyaw * dt
        F[4, 6] = day_dyaw * dt
        
        return F
    
    def predict(self, control, dt):
        """
        EKF prediction step using dynamics model.
        
        Args:
            control: [ax, ay, az, yaw_rate] in body frame
            dt: Time step
        """
        if not self.initialized:
            return
            
        # State prediction
        self.x = self._dynamics(self.x, control, dt)
        
        # Covariance prediction
        F = self._compute_F(self.x, control, dt)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, outlier_threshold=3.0):
        """
        EKF update step with NPE observation.
        
        Args:
            z: NPE observation [x, y, z, yaw]
            outlier_threshold: Mahalanobis distance threshold for outlier rejection
            
        Returns:
            Updated state estimate
        """
        if not self.initialized:
            # Initialize with first observation
            self.initialize(z[0], z[1], z[2], z[3])
            return self.x.copy()
        
        # Innovation
        y = z - self.H @ self.x
        
        # Wrap yaw difference to [-pi, pi]
        y[3] = self._wrap_angle(y[3])
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Mahalanobis distance for outlier detection
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = np.sqrt(y.T @ S_inv @ y)
            
            if mahal_dist > outlier_threshold:
                # Outlier detected, skip update
                return self.x.copy()
        except np.linalg.LinAlgError:
            # Singular matrix, skip outlier check
            S_inv = np.linalg.pinv(S)
        
        # Kalman gain
        K = self.P @ self.H.T @ S_inv
        
        # State update
        self.x = self.x + K @ y
        self.x[6] = self._wrap_angle(self.x[6])  # Wrap yaw
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def get_state(self):
        """Get current state estimate."""
        return self.x.copy()
    
    def get_position(self):
        """Get position [x, y, z]."""
        return self.x[:3].copy()
    
    def get_velocity(self):
        """Get velocity [vx, vy, vz]."""
        return self.x[3:6].copy()
    
    def get_yaw(self):
        """Get yaw angle."""
        return self.x[6]
    
    def set_dynamics_noise(self, enabled, accel_noise=0.5, yaw_rate_noise=0.1):
        """
        Enable/disable dynamics noise for real-world simulation.
        
        Args:
            enabled: Whether to add noise to dynamics
            accel_noise: Acceleration noise std (m/s²)
            yaw_rate_noise: Yaw rate noise std (rad/s)
        """
        self.dynamics_noise_enabled = enabled
        self.dynamics_noise_accel = accel_noise
        self.dynamics_noise_yaw_rate = yaw_rate_noise
        
        if enabled:
            print(f"INFO: [EKF] Dynamics noise ENABLED (accel={accel_noise:.2f} m/s², yaw_rate={yaw_rate_noise:.2f} rad/s)")
        else:
            print("INFO: [EKF] Dynamics noise DISABLED (ideal dynamics)")


# Test
if __name__ == "__main__":
    print("Testing DroneEKF...")
    
    ekf = DroneEKF(dynamics_noise_enabled=False)
    ekf.initialize(0, 0, 0, 0)
    
    # Simulate forward motion
    control = [1.0, 0.0, 0.0, 0.1]  # ax=1, yaw_rate=0.1
    dt = 0.05
    
    for i in range(20):
        ekf.predict(control, dt)
        
        # Simulate noisy observation
        true_state = ekf.get_state()
        noisy_obs = true_state[[0,1,2,6]] + np.random.normal(0, 0.1, 4)
        
        filtered = ekf.update(noisy_obs)
        
        if i % 5 == 0:
            print(f"Step {i}: pos=({filtered[0]:.2f}, {filtered[1]:.2f}, {filtered[2]:.2f}), yaw={np.degrees(filtered[6]):.1f}°")
    
    print("\nTesting with dynamics noise...")
    ekf2 = DroneEKF(dynamics_noise_enabled=True, dynamics_noise_accel=0.3)
    ekf2.initialize(0, 0, 0, 0)
    
    for i in range(20):
        ekf2.predict(control, dt)
        noisy_obs = ekf2.get_state()[[0,1,2,6]] + np.random.normal(0, 0.1, 4)
        filtered = ekf2.update(noisy_obs)
        
        if i % 5 == 0:
            print(f"Step {i}: pos=({filtered[0]:.2f}, {filtered[1]:.2f}, {filtered[2]:.2f})")
    
    print("\n✓ EKF test complete")
