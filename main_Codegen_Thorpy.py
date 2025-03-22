
from mpc.MPC_Codegen_Thorpy import MPC_gen
from dynamics.RocketDynamics import RocketDynamics
import casadi as ca

"""
This script generates C code for the mpc which shall track Thorpy reference trajectories.  

It begins by loading the precompiled CasADi Jacobians and Runge-Kutta 4 (RK4) functions, which are used within the controller.  
The generated C code will be stored in the directory specified by `controller_dir`.  

### Naming Convention:
The suggested naming format for the generated solver is:  
`mpc_Thorpy_Solver_NumberOfTimesteps_PredictionHorizon` 

### Solver Choices:
For real-time applications, **ECOS** is the preferred solver.  
Other options include:
- **CLARABEL**: Performs well for real-time control, but cvxpygen code generation is not supported on Windows.
- **SCS**: Too slow for real-time applications.
- **OSQP**: Can be used if the problem is reformulated as a Quadratic Program (QP). However, this would require removing several "useful" constraints in the hopper MPC case, making it less suitable.


### Usage:
1. Modify the MPC parameters and select the reference trajectory.  
   - The generated solver is **independent** of the reference trajectory, as long as it is described using the same state and control structure as Thorpy.  
   - However, an initial trajectory must be selected before generating the code in `params.py`.  
2. Run `params.py` to update the `params.json` file.  
3. Execute this script to generate the required C code.  

### Important Notes:
- The MPC parameters (such as horizon length, discretization, and weight matrices) are **fixed** in the generated solver and cannot be changed at runtime.  
- The generated solver can be used with any reference trajectory that follows the same state and control formulation used in Thorpy.
"""
#Load the compiled jacobians and rk4 functions to be used in the controller
A_funcgen = ca.external("A_func", "Codegen/Casadi/A_func.so")
B_funcgen = ca.external("B_func", "Codegen/Casadi/B_func.so")
rk4_funcgen = ca.external("rk4_func", "Codegen/Casadi/rk4_func.so")


rocket_dyn = RocketDynamics('params.json')


controller_dir = 'Codegen/ECOS/mpc_ECOS_Nmpc20_T2'

mpc = MPC_gen('params.json',rocket_dyn,A_funcgen,B_funcgen,rk4_funcgen,controller_dir)

x, u,error_vect = mpc.generate_C_code()


