import numpy as np
import time


class ClosedLoop_Simulation:
    """
    This class performs a closed-loop simulation for the mpc, where a given controller 
    is applied to a rocket dynamics model. The simulation steps through time, updating states and 
    control inputs at each step, and tracks performance metrics like tracking errors.

    Attributes:
    -----------
    controller : object
        The controller instance used for the simulation.
    X_ref : ndarray
        The reference state trajectory from the controller.
    U_ref : ndarray
        The reference control input trajectory from the controller.
    x0 : ndarray
        The initial state of the system as provided by the controller.
    """
    def __init__(self, controller):
        """
        Initializes the closed-loop simulation with a given controller instance.

        Parameters:
        -----------
        controller : object
            An instance of the controller class that provides the necessary references, control inputs, 
            initial state, and dynamics.
        """

        self.controller = controller
        self.X_ref = self.controller.X_ref
        self.U_ref = self.controller.U_ref
        self.x0 = self.controller.x0


    def simulate(self):
        """
        Runs the closed-loop simulation by stepping through time, updating the state and control inputs, 
        and calculating the tracking errors for each step.

        Returns:
        --------
        x : ndarray
            The state trajectory over time.
        u : ndarray
            The control input trajectory over time.
        error_vect : ndarray
            The vector of errors between the reference trajectory and the simulated trajectory.
        solver_times : list
            Solve time for the optimization problem at each time step.
        tracking_errors : list
            The tracking errors for each step of the simulation.
        max_tracking_error : float
            The maximum tracking error encountered during the simulation.
        """

        x = np.zeros((self.controller.rocket_dynamics.Nx_sim, self.controller.Nt))
        u = np.zeros((self.controller.Nu, self.controller.Nt - 1))
        x[:, 0] = self.x0
        time_sim = 0
        idx = 0
        max_tracking_error = 0  # Track maximum tracking error
        solver_times = []
        if self.controller.Mass_Flag and self.controller.SCVX_Flag == 0:
            Dyamics_rk4 = self.controller.rocket_dynamics.rocket_dynamics_rk4_SIM_Mass_Thorpy
        else:
            Dyamics_rk4 = self.controller.rocket_dynamics.rocket_dynamics_rk4_SIM_Thorpy
            if self.controller.SCVX_Flag ==1:
                Dyamics_rk4 = self.controller.rocket_dynamics.rocket_dynamics_rk4_SIM_SCVx
        while time_sim <= self.controller.T_sim - self.controller.h_reference:

            if idx % self.controller.controller_update_frequency == 0 or idx == 0:
                start_time = time.time()
                u[:, idx] = self.controller.solve(x, u, idx, time_sim)
                end_time = time.time()
                solver_times.append(end_time - start_time)
            else:
                # Hold the previous control input if MPC is not updated

                uu = u[:, idx - 1]
                # uu = np.zeros_like(u[:, idx - 1])

                u[:, idx] = uu
            x[:, idx + 1] = Dyamics_rk4(
                x[:, idx], u[:, idx], self.controller.h_reference
            )
            
            tracking_error = np.linalg.norm(
                self.X_ref[:, idx] - x[0 : self.controller.Nx_MPC, idx]
            )
            self.controller.tracking_errors.append(tracking_error)
            max_tracking_error = max(max_tracking_error, tracking_error)

            self.controller.error_vect[
                0 : self.controller.Nx_MPC, idx
            ] = (self.X_ref[:, idx] - x[0 : self.controller.Nx_MPC, idx])
            self.controller.error_vect[
                self.controller.Nx_MPC : self.controller.Nx_MPC
                + self.controller.Nu,
                idx,
            ] = (
                self.U_ref[:, idx] - u[:, idx]
            )
            idx += 1

            time_sim = time_sim + self.controller.h_reference
            print("time sim", time_sim)
        return (
            x,
            u,
            self.controller.error_vect,
            solver_times,
            self.controller.tracking_errors,
            max_tracking_error,
        )
