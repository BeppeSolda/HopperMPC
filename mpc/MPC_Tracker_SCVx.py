import numpy as np
import json
from utils.utility_functions import interpolate_trajectory, extend_reference_trajectory
import copy
import time
import pickle


class MPC_Tracker:
    def __init__(
        self,
        params_file,
        rocket_dynamics,
        A_func,
        B_func,
        rk4,
        Controller_dir,
        cpg_solve,
    ):
        """
        Initializes the MPC_gen class with the given parameters.

        Parameters:
            params_file (str): Path to the parameters JSON file.
            rocket_dynamics (object): An instance of the rocket dynamics model.
            A_func (function): Function to compute the Jacobian matrix A with respect to state.
            B_func (function): Function to compute the Jacobian matrix B with respect to control input.
            rk4 (function): Function for Runge-Kutta 4 integration.
            Controller_dir (str): Directory for storing generated controller code.
            cpg_solve: Solver function, which is part of the generated C code for the MPC controller.
                                  It receives updated parameters and solves the optimization problem to compute
                                  control inputs for the system.
        """
        self.rocket_dynamics = rocket_dynamics
        self.params = self.load_params(params_file)
        self.A_func = A_func
        self.B_func = B_func
        self.rk4 = rk4
        self.Controller_dir = Controller_dir
        self.cpg_solve = cpg_solve

        self.Nu = self.rocket_dynamics.Nu
        self.Nx_MPC = self.rocket_dynamics.Nx_MPC

        self.Input_Uncertainty = self.rocket_dynamics.Input_Uncertainty

        self.x0 = self.params.get("x0")
        self.U_ref = self.params.get("U_ref")
        self.X_ref = self.params.get("X_ref")
        self.t_ref = self.params.get("t_ref")
        self.N_mpc = self.params.get("N_mpc")

        self.T_mpc = self.params.get("T_mpc")
        self.h_mpc = self.params.get("h_mpc")
        self.h_reference = self.params.get("h_reference")
        self.controller_frequency = self.params.get("controller_frequency")

        self.Q = self.params.get("Q")
        self.R = self.params.get("R")
        self.cos_delta_max = self.params.get("cos_delta_max")

        self.tan_delta_max = self.params.get("tan_delta_max")

        self.max_angular_rates = self.params.get("max_angular_rates")

        self.max_Tdot = self.params.get("max_Tdot")

        self.Mass_Flag = self.params.get("Mass_Flag")
        self.SCVX_Flag = self.params.get("SCVX_Flag")

        with open(self.Controller_dir, "rb") as f:
            prob = pickle.load(f)
        self.prob = prob

        self.T_sim = self.t_ref[-1]
        print("Simulation time: ", self.T_sim)

        self.Nt = len(self.t_ref)
        print("Nt", self.Nt)
        print("Reference time step size [seconds]", self.h_reference)
        print("MPC time step size [seconds]", self.h_mpc)

        self.error_vect = np.zeros((self.Nx_MPC + self.Nu, self.Nt))
        self.X_sim = [copy.deepcopy(self.x0) for _ in range(self.Nt)]

        self.solver_times = []  # Store solver times
        self.tracking_errors = []  # Store tracking errors at each timestep

        self.X_ref_ext, self.U_ref_ext, self.t_ext = extend_reference_trajectory(
            self.X_ref,
            self.U_ref,
            int(self.T_mpc / self.h_reference),
            self.T_mpc,
            self.t_ref,
        )

        self.X_bar = self.X_ref_ext
        self.U_bar = self.U_ref_ext

        self.NN = int(self.T_mpc / self.h_reference)
        self.U_Tlb_Par_1 = np.zeros((self.N_mpc - 1, self.Nu))
        self.U_Tlb_Par_2 = np.zeros((self.N_mpc - 1))

        h_controller = 1.0 / self.controller_frequency

        print("h controller", h_controller)

        self.controller_update_frequency = int(h_controller / self.h_reference)
        print("MPC iteration every ", self.controller_update_frequency)
        print("Simulation steps")

        N_ext = int(self.T_mpc / self.h_reference)
        self.A_list, self.B_list = self.get_jacobians(
            self.X_bar, self.U_bar, self.Nt, N_ext
        )

        self.X_nominal = np.zeros_like(self.X_bar)

        for i in range(self.Nt + N_ext - 1):

            self.X_nominal[:, i] = (
                self.rk4(self.X_bar[:, i], self.U_bar[:, i], self.h_mpc)
                .full()
                .flatten()
            )

    @staticmethod
    def load_params(params_file):
        """
        Load parameters from a JSON file.

        Parameters:
        params_file (str): Path to the JSON file.

        Returns:
        dict: Parameters loaded from the file.
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

    def compute_jacobian_x(self, x, u):
        """
        Computes the Jacobian matrix with respect to the state.

        Parameters:
            x (np.array): State vector.
            u (np.array): Control input vector.

        Returns:
            np.array: Jacobian matrix with respect to the state.
        """
        return self.A_func(x, u, self.h_mpc)

    def compute_jacobian_u(self, x, u):
        """
        Computes the Jacobian matrix with respect to the control input.

        Parameters:
            x (np.array): State vector.
            u (np.array): Control input vector.

        Returns:
            np.array: Jacobian matrix with respect to the control input.
        """
        return self.B_func(x, u, self.h_mpc)

    def compute_rk4(self, x, u):
        """
        Computes the next state using the Runge-Kutta 4 integration method.

        Parameters:
            x (np.array): Current state vector.
            u (np.array): Current control input vector.

        Returns:
            np.array: Next state after integration.
        """
        return self.rk4(x, u, self.h_mpc)

    def get_jacobians(self, X_ref, U_ref, N, bho):
        """
        Computes the Jacobian matrices A and B for the given reference trajectories X_ref and U_ref.

        Parameters:
        X_ref: The reference state trajectory (Nx x N+N_ext).
        U_ref: The reference control trajectory (Nu x N+N_ext).

        Returns:
        A_list: List of A matrices (Nx x Nx).
        B_list: List of B matrices (Nx x Nu).
        """

        A_list = []
        B_list = []

        for i in range(N + bho - 1):
            X = X_ref[:, i]
            U = U_ref[:, i]

            A = self.compute_jacobian_x(X, U)

            B = self.compute_jacobian_u(X, U)

            A_list.append(A)
            B_list.append(B)
        return A_list, B_list

    def solve(self, x, u, idx, time_sim):
        """
        Solves the optimal control problem for a given time step and updates the control inputs.

        This method performs the following steps:
        - Selects the correct slices of the reference trajectory for both the state and control inputs.
        - Updates the problem parameters with the reference values, Jacobians, and nominal trajectory.
        - Solves the optimization problem using the compiled controller and measures the time taken for the solution.
        - Adds noise to the control input to simulate uncertainty.
        - Returns the optimized control input for the current time step, which is then used in the closed loop simulation.

        Parameters:
            x (ndarray): Current state of the system.
            u (ndarray): Control inputs of the system.
            idx (int): Current time step index.
            time_sim (float): Current simulation time.

        Returns:
            ndarray: Updated control input after solving the optimal control problem.
        """

        print("Controller run")
        idx_interp = int(time_sim / self.h_reference)
        print("idx interp", idx_interp)
        X_bar_new1 = self.X_bar[:, idx_interp : (idx_interp + self.NN)]
        U_bar_new1 = self.U_bar[:, idx_interp : (idx_interp + self.NN - 1)]
        X_nom = self.X_nominal[:, idx_interp : (idx_interp + self.NN)]

        t_new = self.t_ext[idx_interp : (idx_interp + self.NN)]

        U_bar_new, X_bar_new, t = interpolate_trajectory(
            U_bar_new1, X_bar_new1, t_new, self.N_mpc
        )
        U_bar_new, X_nom_new, t = interpolate_trajectory(
            U_bar_new1, X_nom, t_new, self.N_mpc
        )

        for i in range(self.N_mpc - 1):

            self.U_Tlb_Par_1[i, :] = U_bar_new[:, i] / np.linalg.norm(U_bar_new[:, i])
            self.U_Tlb_Par_2[i] = self.U_Tlb_Par_1[i, :] @ U_bar_new[:, i]
        self.prob.param_dict["X_ref_Par"].value = np.array(X_bar_new)
        self.prob.param_dict["U_ref_Par"].value = np.array(U_bar_new)

        self.prob.param_dict["U_Tlb_Par_1"].value = np.array(self.U_Tlb_Par_1)
        self.prob.param_dict["U_Tlb_Par_2"].value = np.array(self.U_Tlb_Par_2)

        x0mpc = x[0 : self.Nx_MPC, idx]

        self.prob.param_dict["x_0"].value = np.array(x0mpc)

        for i in range(self.N_mpc - 1):

            self.prob.param_dict[f"A_{i}"].value = np.array(self.A_list[idx_interp + i])
            self.prob.param_dict[f"B_{i}"].value = np.array(self.B_list[idx_interp + i])
            self.prob.param_dict[f"X_nominal_{i}"].value = X_nom_new[:, i]
        self.prob.register_solve("CPG", self.cpg_solve)

        t0 = time.time()
        error = self.prob.solve(method="CPG")
        t1 = time.time()
        solve_time = t1 - t0

        self.solver_times.append(solve_time)
        print("\nCVXPYgen\nSolve time: %.3f ms" % (1000 * (solve_time)))
        print("Objective function value: %.6f\n" % error)

        DeltaU = self.prob.var_dict["DeltaU"].value[:, 0]

        DeltaX = self.prob.var_dict["DeltaX"].value[:, 1]
        uu = U_bar_new[:, 0] + DeltaU

        self.X_sim[idx] = self.X_bar[:, 1] + DeltaX

        uu = self.rocket_dynamics.add_percentage_gaussian_noise(
            uu, self.Input_Uncertainty
        )

        u[:, idx] = uu

        return uu
