
from mpc.MPC_Tracker_Thorpy import MPC_Tracker
from dynamics.RocketDynamics import RocketDynamics
from simulation.ClosedLoop_Simulation import ClosedLoop_Simulation
from utils.plot_results import plot_results_tracker, plot_mpc_metrics
import casadi as ca
from Codegen.ECOS.mpc_ECOS_Nmpc30_T3.cpg_solver import cpg_solve

"""
This script runs the closed loop simulation to validate tracking performance of the compiled mpc on Thorpy reference trajectories.  

### Overview:
- It loads the precompiled CasADi Jacobians and Runge-Kutta 4 (RK4) functions, which are required by the controller.
- The MPC parameters **must match** those used during the C code generation to ensure consistency.
- The closed-loop simulation is executed using the precompiled solver.

### Usage:
1. Modify the MPC parameters and select the reference trajectory.
   - The **MPC parameters must be the same** as those used for generating the C code of the controller.
2. Run `params.py` to update the `params.json` file.
3. Execute this script to run the closed-loop simulation.

### Outputs:
- State and control trajectories compared against the Thorpy reference.
- Solver execution times and tracking errors.
- Performance metrics plotted for analysis.
"""
A_funcgen = ca.external("A_func", "Codegen/Casadi/A_func.so")
B_funcgen = ca.external("B_func", "Codegen/Casadi/B_func.so")
rk4_funcgen = ca.external("rk4_func", "Codegen/Casadi/rk4_func.so")


rocket_dyn = RocketDynamics('params.json')
controller_dir = 'Codegen/ECOS/mpc_ECOS_Nmpc30_T3/problem.pickle'


mpc = MPC_Tracker('params.json',rocket_dyn,A_funcgen,B_funcgen,rk4_funcgen,controller_dir,cpg_solve)
cl_sim = ClosedLoop_Simulation(mpc)

x, u,error_vect, solver_times, tracking_errors, max_tracking_error =cl_sim.simulate()



# save_folder = 'C:/Users/Utente/Desktop/Hopper_plots/Quaternions/T3_N30_Mass_Long2'

plot_results_tracker(mpc.t_ref, x, mpc.X_ref, u, mpc.U_ref)

plot_mpc_metrics(mpc.t_ref, solver_times, tracking_errors, max_tracking_error)

