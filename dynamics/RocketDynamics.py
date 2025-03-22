import numpy as np
import json


class RocketDynamics:

    def __init__(self, params_file):
        self.params = self.load_params(params_file)

        self.height = self.params.get("height")

        self.radius = self.params.get("radius")

        self.Nx_MPC = self.params.get("Nx_MPC")

        self.Nx_sim = self.params.get("Nx_sim")
        self.Nu = self.params.get("Nu")

        self.thrust_time_constant = self.params.get("thrust_time_constant")

        self.max_thrust = self.params.get("max_thrust")

        self.alpha_thrust = self.params.get("alpha_thrust")

        self.min_thrust = self.params.get("min_thrust")

        self.g_I = self.params.get("g_I")

        self.r_T_B = self.params.get("r_T_B")

        self.J_B = self.params.get("J_B")

        self.m = self.params.get("m")

        self.Input_Uncertainty = self.params.get("Input_Uncertainty")

        self.cos_delta_max = self.params.get("cos_delta_max")

        self.tan_delta_max = self.params.get("tan_delta_max")

        self.max_angular_rates = self.params.get("max_angular_rates")

        self.max_Tdot = self.params.get("max_Tdot")

    @staticmethod
    def load_params(params_file):
        """
        Load parameters from a JSON file.

        Parameters:
        params_file (str): Path to the JSON file.

        Returns:
        params: Parameters loaded from the file.
        """
        with open(params_file, "r") as file:
            params = json.load(file)
        params["x0"] = np.array(params["x0"])
        params["u0"] = np.array(params["u0"])
        params["X_ref"] = np.array(params["X_ref"])
        params["U_ref"] = np.array(params["U_ref"])
        params["t_ref"] = np.array(params["t_ref"])
        params["g"] = np.array(params["g"])
        params["Q"] = np.array(params["Q"])
        params["R"] = np.array(params["R"])
        params["r_T_B"] = np.array(params["r_T_B"])
        params["J_B"] = np.array(params["J_B"])
        return params

    def add_percentage_gaussian_noise(self, actual_value, noise_percentage):
        """
        Adds Gaussian noise as a percentage of the actual value.

        Parameters:
        - rng_key: PRNG key for random number generation (JAX-specific)
        - actual_value: The actual value to which noise will be added (numpy array)
        - noise_percentage: The standard deviation of the noise as a fraction of the actual value (e.g., 0.05 for 5% noise)

        Returns:
        - noisy_value: The actual value with noise added
        """

        noise = np.random.normal(0, noise_percentage, actual_value.shape)
        noisy_value = actual_value * (1 + noise)
        return noisy_value

    def skew(self, v):
        """
        Compute the skew-symmetric matrix for a given vector.

        Parameters:
            v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The skew-symmetric matrix.
        """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def qtoM_WB(self, q):
        """
        Convert a quaternion to a rotation matrix from the body frame to the world frame.

        Parameters:
            q (np.ndarray): The quaternion representing the rotation.

        Returns:
            np.ndarray: The corresponding 3x3 rotation matrix.
        """

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        Q = np.zeros((3, 3))
        Q[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        Q[0, 1] = 2 * (q1 * q2 + q0 * q3)
        Q[0, 2] = 2 * (q1 * q3 - q0 * q2)

        Q[1, 0] = 2 * (q1 * q2 - q0 * q3)
        Q[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        Q[1, 2] = 2 * (q2 * q3 + q0 * q1)

        Q[2, 0] = 2 * (q1 * q3 + q0 * q2)
        Q[2, 1] = 2 * (q2 * q3 - q0 * q1)
        Q[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

        return Q.T

    def Omega(self, q):
        """
        Compute the Omega matrix for quaternion-based angular velocity transformations.

        Parameters:
            q (np.ndarray): The quaternion representing the rotation.

        Returns:
            np.ndarray: The Omega matrix (4x3).
        """
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        Omega_Mat = np.zeros((4, 3))
        Omega_Mat[0, 0] = -q1
        Omega_Mat[0, 1] = -q2
        Omega_Mat[0, 2] = -q3

        Omega_Mat[1, 0] = q0
        Omega_Mat[1, 1] = -q3
        Omega_Mat[1, 2] = q2

        Omega_Mat[2, 0] = q3
        Omega_Mat[2, 1] = q0
        Omega_Mat[2, 2] = -q1

        Omega_Mat[3, 0] = -q2
        Omega_Mat[3, 1] = q1
        Omega_Mat[3, 2] = q0

        return Omega_Mat

    def rocket_dynamics_MPC_Thorpy(self, x, u):
        """
        Rocket dynamics model used as mpc internal model for Tracking thorpy reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs.

        Returns:
            np.ndarray: The time derivative of the state.
        """
        q = x[3:7]
        v = x[7:10]
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        w = x[10:13]
        thrust = x[13:16]

        f = np.zeros(self.Nx_MPC)
        M_WB = self.qtoM_WB(q)
        M_BW = M_WB.T

        f[0:3] = M_WB.dot(v)
        f[3:7] = 0.5 * np.dot(self.Omega(q), w)
        f[7:10] = ((1 / self.m) * thrust) + M_BW.dot(self.g_I) - self.skew(w).dot(v)
        ang_vel_dynamics = np.linalg.inv(self.J_B).dot(
            self.skew(self.r_T_B).dot(thrust) - self.skew(w).dot(self.J_B).dot(w)
        )

        f[10:13] = ang_vel_dynamics
        thrust_dot = u[0:3]
        f[13:16] = thrust_dot

        return f

    def rocket_dynamics_SIM_Thorpy(self, x, u):
        """
        Rocket dynamics model used for closed loop simulation to validate Tracking accuracy of the mpc on thorpy reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs.

        Returns:
            np.ndarray: The time derivative of the state.
        """
        q = x[3:7]
        v = x[7:10]
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        w = x[10:13]

        m = self.m
        thrust = x[13:16]
        Thrust_real = x[16]

        J1 = 1 / 12 * m * (self.height**2 + 3 * self.radius**2)
        J2 = J1
        J3 = 0.5 * m * self.radius**2
        J_B = np.diag(np.array([J1, J2, J3]))

        f = np.zeros(self.Nx_sim)

        M_WB = self.qtoM_WB(q)
        M_BW = M_WB.T

        thrust_command_norm = np.linalg.norm(
            thrust[0:3]
        )  # Thrust magnitude commanded by mpc
        scaling_factor = Thrust_real / thrust_command_norm

        thrust = thrust * scaling_factor

        f[0:3] = M_WB.dot(v)

        f[3:7] = 0.5 * np.dot(self.Omega(q), w)
        f[7:10] = ((1 / m) * thrust[0:3]) + M_BW.dot(self.g_I) - self.skew(w).dot(v)
        ang_vel_dynamics = np.linalg.inv(J_B).dot(
            self.skew(self.r_T_B).dot(thrust[0:3]) - self.skew(w).dot(J_B).dot(w)
        )
        f[10:13] = ang_vel_dynamics

        f[16] = (thrust_command_norm - Thrust_real) / self.thrust_time_constant
        thrust_dot = u[0:3]
        f[13:16] = thrust_dot

        return f

    def rocket_dynamics_SIM_Mass_Thorpy(self, x, u):
        """
        Rocket dynamics model used for closed loop simulation to validate Tracking accuracy of the mpc on thorpy reference trajectories.
        The model takes into account mass variation
        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs.

        Returns:
            np.ndarray: The time derivative of the state.
        """
        q = x[3:7]
        v = x[7:10]
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        w = x[10:13]
        m = x[17]

        thrust = x[13:16]
        Thrust_norm_real = x[16]

        J1 = 1 / 12 * m * (self.height**2 + 3 * self.radius**2)
        J2 = J1
        J3 = 0.5 * m * self.radius**2
        J_B = np.diag(np.array([J1, J2, J3]))

        f = np.zeros(self.Nx_sim)

        M_WB = self.qtoM_WB(q)
        M_BW = M_WB.T
        thrust_command_norm = np.linalg.norm(
            thrust[0:3]
        )  # Thrust magnitude commanded by mpc
        scaling_factor = Thrust_norm_real / thrust_command_norm

        thrust_PT1 = thrust * scaling_factor

        f[0:3] = M_WB.dot(v)
        f[3:7] = 0.5 * np.dot(self.Omega(q), w)
        f[7:10] = ((1 / m) * thrust_PT1[0:3]) + M_BW.dot(self.g_I) - self.skew(w).dot(v)
        ang_vel_dynamics = np.linalg.inv(J_B).dot(
            self.skew(self.r_T_B).dot(thrust_PT1[0:3]) - self.skew(w).dot(J_B).dot(w)
        )
        f[10:13] = ang_vel_dynamics

        thrust_dot = u[0:3]
        f[13:16] = thrust_dot

        f[16] = (thrust_command_norm - Thrust_norm_real) / self.thrust_time_constant

        mass_flow_rate = self.alpha_thrust * (thrust_command_norm / self.max_thrust)
        f[17] = -mass_flow_rate

        return f

    def rocket_dynamics_MPC_SCVx(self, x, u):
        """
        Rocket dynamics model used as mpc internal model for Tracking SCVx reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs.

        Returns:
            np.ndarray: The time derivative of the state.
        """

        r = x[0:3]
        q = x[3:7]
        v = x[7:10]
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        w = x[10:13]
        m = x[13]

        J1 = 1 / 12 * m * (self.height**2 + 3 * self.radius**2)
        J2 = J1
        J3 = 0.5 * m * self.radius**2
        J_B = np.diag(np.array([J1, J2, J3]))

        f = np.zeros(self.Nx_MPC)
        M_WB = self.qtoM_WB(q)
        M_BW = M_WB.T

        f[0:3] = M_WB.dot(v)
        f[3:7] = 0.5 * np.dot(self.Omega(q), w)
        f[7:10] = ((1 / m) * u) + M_BW.dot(self.g_I) - self.skew(w).dot(v)
        ang_vel_dynamics = np.linalg.inv(J_B).dot(
            self.skew(self.r_T_B).dot(u) - self.skew(w).dot(J_B).dot(w)
        )

        f[10:13] = ang_vel_dynamics

        thrust_command_norm = np.linalg.norm(u)
        mass_flow_rate = self.alpha_thrust * (thrust_command_norm / self.max_thrust)
        f[13] = -mass_flow_rate
        return f

    def rocket_dynamics_SIM_SCVx(self, x, u):
        """
        Rocket dynamics model used for closed loop simulation to validate Tracking accuracy of the mpc on SCVx reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs.

        Returns:
            np.ndarray: The time derivative of the state.
        """

        r = x[0:3]
        q = x[3:7]
        v = x[7:10]
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        w = x[10:13]
        m = x[13]

        Thrust_real = x[14]

        J1 = 1 / 12 * m * (self.height**2 + 3 * self.radius**2)
        J2 = J1
        J3 = 0.5 * m * self.radius**2
        J_B = np.diag(np.array([J1, J2, J3]))

        f = np.zeros(self.Nx_sim)
        M_WB = self.qtoM_WB(q)
        M_BW = M_WB.T

        thrust_command_norm = np.linalg.norm(u)  # Thrust magnitude commanded by mpc
        scaling_factor = Thrust_real / thrust_command_norm

        u_real = u * scaling_factor
        u_noisy = u_real
        f[0:3] = M_WB.dot(v)
        f[3:7] = 0.5 * np.dot(self.Omega(q), w)
        f[7:10] = ((1 / m) * u_noisy) + M_BW.dot(self.g_I) - self.skew(w).dot(v)
        ang_vel_dynamics = np.linalg.inv(J_B).dot(
            self.skew(self.r_T_B).dot(u_noisy) - self.skew(w).dot(J_B).dot(w)
        )

        f[10:13] = ang_vel_dynamics
        mass_flow_rate = self.alpha_thrust * (Thrust_real / self.max_thrust)
        f[13] = -mass_flow_rate
        f[14] = (thrust_command_norm - Thrust_real) / self.thrust_time_constant

        return f

    def rocket_dynamics_rk4_MPC_Thorpy(self, x, u, h):
        """
        Apply Runge-Kutta 4th order integration to the rocket dynamics model used as mpc internal model for tracking thorpy reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs for the system.
            h (float): Time step for integration.

        Returns:
            np.ndarray: The updated state after integration.
        """
        f1 = self.rocket_dynamics_MPC_Thorpy(x, u)
        f2 = self.rocket_dynamics_MPC_Thorpy(x + 0.5 * h * f1, u)
        f3 = self.rocket_dynamics_MPC_Thorpy(x + 0.5 * h * f2, u)
        f4 = self.rocket_dynamics_MPC_Thorpy(x + h * f3, u)
        xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[3:7])
        xn[3:7] = xn[3:7] / q_norm
        return xn

    def rocket_dynamics_rk4_SIM_Thorpy(self, x, u, h):
        """
        Apply Runge-Kutta 4th order integration to the rocket dynamics model used for closed loop simulation to validate tracking accuracy on Thorpy reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs for the system.
            h (float): Time step for integration.

        Returns:
            np.ndarray: The updated state after integration.
        """
        f1 = self.rocket_dynamics_SIM_Thorpy(x, u)
        f2 = self.rocket_dynamics_SIM_Thorpy(x + 0.5 * h * f1, u)
        f3 = self.rocket_dynamics_SIM_Thorpy(x + 0.5 * h * f2, u)
        f4 = self.rocket_dynamics_SIM_Thorpy(x + h * f3, u)
        xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[3:7])
        xn[3:7] = xn[3:7] / q_norm
        return xn

    def rocket_dynamics_rk4_SIM_Mass(self, x, u, h):
        """
        Apply Runge-Kutta 4th order integration to the rocket dynamics model used for closed loop simulation to validate tracking accuracy on Thorpy reference trajectories.
        Takes mass variation into account.
        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs for the system.
            h (float): Time step for integration.

        Returns:
            np.ndarray: The updated state after integration.
        """
        f1 = self.rocket_dynamics_SIM_Mass_Thorpy(x, u)
        f2 = self.rocket_dynamics_SIM_Mass_Thorpy(x + 0.5 * h * f1, u)
        f3 = self.rocket_dynamics_SIM_Mass_Thorpy(x + 0.5 * h * f2, u)
        f4 = self.rocket_dynamics_SIM_Mass_Thorpy(x + h * f3, u)
        xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[3:7])
        xn[3:7] = xn[3:7] / q_norm
        return xn

    def rocket_dynamics_rk4_SIM_SCVx(self, x, u, h):
        """
        Apply Runge-Kutta 4th order integration to the rocket dynamics model used for closed loop simulation to validate tracking accuracy on SCVx reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs for the system.
            h (float): Time step for integration.

        Returns:
            np.ndarray: The updated state after integration.
        """
        f1 = self.rocket_dynamics_SIM_SCVx(x, u)
        f2 = self.rocket_dynamics_SIM_SCVx(x + 0.5 * h * f1, u)
        f3 = self.rocket_dynamics_SIM_SCVx(x + 0.5 * h * f2, u)
        f4 = self.rocket_dynamics_SIM_SCVx(x + h * f3, u)
        xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[3:7])
        xn[3:7] = xn[3:7] / q_norm
        return xn

    def rocket_dynamics_rk4_MPC_SCVx(self, x, u, h):
        """
        Apply Runge-Kutta 4th order integration to the rocket dynamics model used as mpc internal model for tracking SCVx reference trajectories.

        Parameters:
            x (np.ndarray): State vector of the rocket.
            u (np.ndarray): Control inputs for the system.
            h (float): Time step for integration.

        Returns:
            np.ndarray: The updated state after integration.
        """
        f1 = self.rocket_dynamics_MPC_SCVx(x, u)
        f2 = self.rocket_dynamics_MPC_SCVx(x + 0.5 * h * f1, u)
        f3 = self.rocket_dynamics_MPC_SCVx(x + 0.5 * h * f2, u)
        f4 = self.rocket_dynamics_MPC_SCVx(x + h * f3, u)
        xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[3:7])
        xn[3:7] = xn[3:7] / q_norm
        return xn
