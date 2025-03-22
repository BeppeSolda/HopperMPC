import cvxpy as cp
import numpy as np
from utils.utility_functions import interpolate_trajectory, extend_reference_trajectory
import time
import sys
import json


class MPC_gen:
    """
    A class for generating c code of a Model Predictive Controller (MPC)  for tracking a Thropy reference trajectory.

    """

    def __init__(
        self,
        params_file,
        rocket_dynamics,
        A_func,
        B_func,
        rk4,
        Controller_dir,
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
        """
        self.rocket_dynamics = rocket_dynamics
        self.params = self.load_params(params_file)
        self.A_func = A_func
        self.B_func = B_func
        self.rk4 = rk4
        self.Controller_dir = Controller_dir

        self.Nu = self.rocket_dynamics.Nu
        self.Nx_MPC = self.rocket_dynamics.Nx_MPC

        self.Input_Uncertainty = self.rocket_dynamics.Input_Uncertainty

        self.x0 = self.params.get("x0")
        self.U_ref = self.params.get("U_ref")
        self.X_ref = self.params.get("X_ref")
        self.t = self.params.get("t_ref")
        self.N_mpc = self.params.get("N_mpc")

        self.T_mpc = self.params.get("T_mpc")
        self.h_mpc = self.params.get("h_mpc")
        self.h_reference = self.params.get("h_reference")

        self.Q = self.params.get("Q")
        self.R = self.params.get("R")
        self.cos_delta_max = self.params.get("cos_delta_max")

        self.max_angular_rates = self.params.get("max_angular_rates")

        self.max_Tdot = self.params.get("max_Tdot")
        self.T_sim = self.t[-1]
        print("Simulation time: ", self.T_sim)

        self.Nt = len(self.t)
        self.N_ext = int(self.T_mpc / self.h_reference)

        self.X_ref_ext, self.U_ref_ext, self.t_ext = extend_reference_trajectory(
            self.X_ref,
            self.U_ref,
            self.N_ext,
            self.T_mpc,
            self.t,
        )

        self.X_bar = self.X_ref_ext
        self.U_bar = self.U_ref_ext
        self.error_vect = np.zeros((self.Nx_MPC + self.Nu, self.Nt))

        self.X_nominal = np.zeros_like(self.X_bar)

        self.x = np.zeros((self.rocket_dynamics.Nx_sim, self.Nt))
        self.u = np.zeros((self.Nu, self.Nt))
        self.x[:, 0] = self.x0

        self.A_list, self.B_list = self.get_jacobians(
            self.X_bar, self.U_bar, self.Nt, self.N_ext
        )
        for i in range(self.Nt + self.N_ext - 1):

            self.X_nominal[:, i] = (
                self.compute_rk4(self.X_bar[:, i], self.U_bar[:, i]).full().flatten()
            )

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

    def get_jacobians(self, X_ref, U_ref, N, N_ext):
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

        for i in range(N + N_ext - 1):
            X = X_ref[:, i]
            U = U_ref[:, i]

            A = self.compute_jacobian_x(X, U)

            B = self.compute_jacobian_u(X, U)

            A_list.append(A)
            B_list.append(B)
        return A_list, B_list

    def convex_mpc_deltas(self, X_bar, U_bar, X_nom, A_list, B_list, x0, idx):
        """
        Purpose:
            This function sets up and solves an optimal control problem for tracking a reference trajectory using convex MPC.
            It solves the first iteration of the problem, and if successful, generates C code for the MPC controller using cvxpygen.

        Workflow:
            1. An instance of the optimal control problem is created to solve for tracking the reference trajectory.
            2. The first iteration of the optimization problem is solved.
            3. If the solution is successful, C code for the MPC controller is generated using cvxpygen.
            4. The user is prompted to decide whether they want to continue the process. It is recommended to run a few iterations
               and then stop, as the Python implementation is significantly slower than the compiled C code version (MPC_Tracker_Scvx).
            5. The solver may encounter errors after a few iterations. However, this is not an issue, as the compiled C code will
               work correctly if it was generated successfully.
        """
        A = [
            cp.Parameter(
                (self.Nx_MPC, self.Nx_MPC),
                name=f"A_{_}",
            )
            for _ in range(self.N_mpc - 1)
        ]
        B = [
            cp.Parameter((self.Nx_MPC, self.Nu), name=f"B_{_}")
            for _ in range(self.N_mpc - 1)
        ]

        X_nominal = [
            cp.Parameter(self.Nx_MPC, name=f"X_nominal_{_}")
            for _ in range(self.N_mpc - 1)
        ]
        X_Tlb_Par_1 = cp.Parameter((self.N_mpc - 1, self.Nu), name="X_Tlb_Par_1")
        X_Tlb_Par_2 = cp.Parameter((self.N_mpc - 1), name="X_Tlb_Par_2")

        U_ref_Par = cp.Parameter((self.Nu, self.N_mpc - 1), name="U_ref_Par")
        X_ref_Par = cp.Parameter((self.Nx_MPC, self.N_mpc), name="X_ref_Par")

        x_0 = cp.Parameter((self.Nx_MPC), name="x_0")

        DeltaX = cp.Variable((self.Nx_MPC, self.N_mpc), name="DeltaX")
        DeltaU = cp.Variable((self.Nu, self.N_mpc - 1), name="DeltaU")

        x_0.value = np.array(x0)

        X_ref_Par.value = np.array(X_bar)
        U_ref_Par.value = np.array(U_bar)

        for i in range(self.N_mpc - 1):

            A[i].value = np.array(A_list[i])
            B[i].value = np.array(B_list[i])
            X_nominal[i].value = X_nom[:, i]
        X_Tlb_Par_1_values = np.zeros((self.N_mpc - 1, self.Nu))
        X_Tlb_Par_2_values = np.zeros(self.N_mpc - 1)
        for i in range(self.N_mpc - 1):

            X_Tlb_Par_1_values[i, :] = X_bar[13:16, i] / np.linalg.norm(X_bar[13:16, i])
            X_Tlb_Par_2_values[i] = X_Tlb_Par_1_values[i, :] @ X_bar[13:16, i]
        X_Tlb_Par_1.value = X_Tlb_Par_1_values
        X_Tlb_Par_2.value = X_Tlb_Par_2_values

        cost = 0.0

        for i in range(self.N_mpc):
            # State deviation cost

            xi = (
                X_ref_Par[:, i] + DeltaX[:, i]
            )  # Predicted state aka nominal state X_bar + delta deviation.
            cost += 0.5 * cp.quad_form(xi[0:13] - X_ref_Par[0:13, i], self.Q)
        for i in range(self.N_mpc - 1):
            cost += 0.5 * cp.quad_form(DeltaU[:, i], 0.000001 * self.R)
        #   ui = U_bar[:,i] + DeltaU[:, i]
        #   cost += 0.5 * cp.quad_form(ui - U_ref[:,i], 0.00001*self.R)

        constraints = []
        constraints += [X_ref_Par[:, 0] + DeltaX[:, 0] == x_0]

        for i in range(self.N_mpc - 1):

            constraints += [
                X_ref_Par[:, i + 1] + DeltaX[:, i + 1]
                == X_nominal[i] + A[i] @ DeltaX[:, i] + B[i] @ DeltaU[:, i],
                self.rocket_dynamics.min_thrust
                - X_Tlb_Par_1[i, :] @ (DeltaX[13:16, i])
                - X_Tlb_Par_2[i]
                <= 0,
                cp.norm(X_ref_Par[13:16, i] + DeltaX[13:16, i])
                <= self.rocket_dynamics.max_thrust,
                cp.norm((X_ref_Par[13:16, i] + DeltaX[13:16, i]), axis=0)
                * self.rocket_dynamics.cos_delta_max
                <= (X_ref_Par[15, i] + DeltaX[15, i]),
                cp.abs((X_ref_Par[10, i] + DeltaX[10, i]))
                <= self.rocket_dynamics.max_angular_rates,
                cp.abs((X_ref_Par[11, i] + DeltaX[11, i]))
                <= self.rocket_dynamics.max_angular_rates,
                cp.norm((U_ref_Par[:, i] + DeltaU[:, i]), axis=0)
                <= self.rocket_dynamics.max_Tdot,
            ]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        t0 = time.time()
        val = prob.solve(solver=cp.ECOS, verbose=False)
        t1 = time.time()
        print("\nCVXPY\nSolve time: %.3f ms" % (1000 * (t1 - t0)))
        print("Objective function value: %.6f\n" % val)
        if idx == 0:
            from cvxpygen import cpg

            cpg.generate_code(prob, code_dir=self.Controller_dir, solver="ECOS")
            print("\nCode generation successful!")
            print(f"Generated code can be found in: {self.Controller_dir}")
            response = input("\nDo you want to continue? (y/n): ").strip().lower()

            if response != "y":
                print("Aborting execution.")
                sys.exit()
            print("Continuing execution...")
        solveTime = t1 - t0

        return (
            U_bar[:, 0] + DeltaU[:, 0].value,
            X_ref_Par[:, 1].value + DeltaX[:, 1].value,
            solveTime,
        )

    def generate_C_code(self):
        """
        Purpose:
            This function generates C code for the convex MPC controller by solving an optimal control problem for trajectory tracking.
            The C code is generated after verifying that a first iteration of the MPC problem is solved successfully.

        Workflow:
            1. The function begins by performing a first iteration of the convex MPC to check if the problem can be solved successfully.
            2. If the first iteration is successful, it proceeds with iterating through the reference trajectory, performing interpolation,
               and solving the convex MPC problem at each step.
            3. The system's state is updated using Runge-Kutta integration (`rocket_dynamics_rk4_SIM_SCVx`), and the control inputs are
               computed and applied at each step.
            5. After completing the first iteration, C code is generated using the `cvxpygen` library to enable faster execution of the MPC
               controller in compiled form.

            Performance Considerations:
                - The function performs an initial check to ensure that the convex MPC solver works correctly before proceeding with
                  code generation.
                - After the first successful iteration, C code for the controller is generated to optimize execution speed, as Python-based
                  computation is slower compared to compiled code.
                - If the solver encounters any issues after the first successful iteration, it is expected that the generated C code will
                  still function correctly.

        Returns:
            - `x`: The system state over time after each iteration.
            - `u`: The control input over time generated by the MPC controller.

        """
        idx = 0
        time_sim = 0
        SolveTime = 0
        A = []
        B = []
        while time_sim < self.T_sim - 2 * self.h_mpc:
            idx_interp = int(time_sim / self.h_reference)
            print("idx interp", idx_interp)

            X_bar_slice = self.X_bar[:, idx_interp : (idx_interp + self.N_ext)]
            U_bar_slice = self.U_bar[:, idx_interp : (idx_interp + self.N_ext - 1)]
            X_nom_slice = self.X_nominal[:, idx_interp : (idx_interp + self.N_ext)]

            t_new = self.t_ext[idx_interp : (idx_interp + self.N_ext)]

            U_bar_slice_interp, X_bar_slice_interp, t = interpolate_trajectory(
                U_bar_slice, X_bar_slice, t_new, self.N_mpc
            )
            U_bar_new2, X_nom_slice_interp, t = interpolate_trajectory(
                U_bar_slice, X_nom_slice, t_new, self.N_mpc
            )
            for i in range(self.N_mpc - 1):

                AA = np.array(self.A_list[idx_interp + i])
                BB = np.array(self.B_list[idx_interp + i])

                A.append(AA)
                B.append(BB)
            x0MPC = self.x[0 : self.Nx_MPC, idx]

            uu, xx, t_solve = self.convex_mpc_deltas(
                X_bar_slice_interp,
                U_bar_slice_interp,
                X_nom_slice_interp,
                A,
                B,
                x0MPC,
                idx,
            )

            SolveTime += t_solve
            self.u[:, idx] = uu

            self.x[:, idx + 1] = self.rocket_dynamics.rocket_dynamics_rk4_SIM_Thorpy(
                self.x[:, idx], uu, self.h_reference
            )
            idx += 1

            time_sim = time_sim + self.h_reference
            print("Iter", idx)
            print("time sim", time_sim)
        return self.x, self.u, self.error_vect
