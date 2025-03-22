import numpy as np
from utils.trajectory import Trajectory
from math import pi
import json

"""
This file contains the hopper parameters and all parameters necessary for the mpc to run. 
Mass_Flag controls the dynamics model which is used by the closed loop simulation, if 1 then the controller will be simulated on a system in which mass depletion is taken into account as a function of the thrust magnitude, otherwise the system mass is held constant.

In order to track/generate controllers for SCVX trajectories, SCVX_Flag = 1 and Mass_Flag = 0 are both necessary.
"""
"""
This file defines the hopper parameters and all necessary settings for running the mpcin closed-loop simulations or for the code generation of the mpc.

### Flags: Used to setup the closed loop simulation, codegeneration and to load the correct reference trajectory.
- `Mass_Flag`: Controls the dynamics model used in the simulation.  
  - If set to `1`, the controller is tested on a system where mass depletion is modeled as a function of thrust magnitude.  
  - If set to `0`, the system mass remains constant.  
- `SCVX_Flag`: SCVX_Flag sets up the closed loop simulation/code generation to accept a model which takes into account the mass depletion also in the mpc, 
    in such a way that the SCVX generated trajectories can be tracked by the mpc. Thrust is the control and not a state in this case. 
  - **Note**: To track SCVX-generated trajectories, both `SCVX_Flag = 1` and `Mass_Flag = 0` are required.

### Initial Conditions and Reference Trajectory:
- The initial state (`x0`) and control input (`u0`) are obtained from the loaded reference trajectory.
- Reference trajectory parameters include:
  - `h_reference`: Reference trajectory time step.  
  - `Nt`: Number of reference trajectory points.  
  - `X_ref`, `U_ref`, `t_ref`: Reference state, control, and time vectors. 
  -If SCVX_Flag =1, the trajectory is loaded from the SCVX folder, while a thorpy trajectory is chosen otherwise.
  -Reference_Interpolation_Points: to control the number of points contained in the reference trajectory.

### Physical Parameters:
- Gravity: `g = 9.81 m/s²`, defined as a vector `g_I` in the inertial frame.
- Hopper dimensions:
  - Height: `2.5 m`
  - Diameter: `0.3 m`
  - Radius: `0.15 m`
- Initial mass: `m_init = 232 kg` (150 kg structure + 82 kg propellant).
- Inertia matrix `J_B` is computed based on standard inertia formulas.
- Thrust application point (`r_T_B`): Vector from the center of gravity (CG) to the thrust point.

### Dynamics and Constraints:
- **Mass variation**: If `Mass_Flag == 1`, mass is added as a state.
- **Inertia Tensor (`J_B`)**: Computed based on the hopper's geometry.
- **Gimbal constraints**:
  - Maximum gimbal deflection: `10°`
  - Corresponding cosine limit: `cos_delta_max = cos(10°)`
- **Thrust constraints**:
  - Maximum thrust: `2700 N`
  - Minimum thrust: `1100 N`
  - Mass depletion proportionality constant: `alpha_thrust = 1.33`
  - Thrust dynamics modeled as a first-order system with time constant `0.25 s`.
- **Angular Rate Constraints**: Maximum allowed body angular rate: `60°/s`.
- **Thrust rate constraint**: Maximum thrust variation per second: `3000 N/s`.

### MPC Parameters:
- **State and control dimensions**:
  - `Nx_MPC`: Number of states used in the MPC (varies depending on flags).
  - `Nu`: Number of control inputs (3).
- **Prediction horizon**:
  - `N_mpc = number of discretization points with which the prediction horizon is discretized.
  - `T_mpc =  total prediction time.
  - `h_mpc = time step, dependent on N_mpc and T_mpc.
- **Weight Matrices**:
  - `Q`: State weight matrix (13x13 identity matrix). (In both Thorpy and SCVX trajectories, only position, velocities, quaternions and angular rates are the goal of the tracking)
  - `R`: Control weight matrix (3x3 identity matrix).
- **Controller Frequency**: Frequency with which the controller is called during closed loop simulation
- **Input Uncertainty**:  uncertainty in the input channel.

Finally, all parameters are stored in a JSON file (`params.json`) for easy access and integration into simulations.
"""


Mass_Flag = 0
SCVX_Flag = 1
if SCVX_Flag == 1:
    Trajectory_Flag = 'SCVx'
    filename = 'Reference_Trajectories/SCVx/SCVx_0_20_60_tf60.npz'
    Reference_Interpolation_Points = 2002
else:
    Trajectory_Flag = 'Thorpy'
    filename = 'Reference_Trajectories/Thorpy/Divert_Land.txt'
    Reference_Interpolation_Points = 1002



  

trajectory = Trajectory(filename,Trajectory_Flag,Reference_Interpolation_Points)

x0, u0 = trajectory.get_initial_conditions()

h_reference = trajectory.h
Nt = trajectory.Nt


X_ref = trajectory.X_ref
U_ref = trajectory.U_ref
t_ref = trajectory.t_ref

g = 9.81
g_I = np.array([0, 0, -g])  # Gravity vector in the inertial frame

height = 2.5 #Hopper height
diameter = 0.3 #Hopper Diameter
radius = diameter / 2



m_init = 150 + 82 #Hopper initial mass
m = m_init
r_T_B = np.array([0, 0, -0.5]) #Lever arm vector in the body frame, distance between CG and thrust application point

if Mass_Flag and SCVX_Flag==0:
    x0 = np.hstack((x0, m_init * g, m_init))
    Nx_sim = 18
    Nx_MPC = 16  # Mpc has 2 less states than the plant model used for simulation, as it does not take into account mass variation and thrust magnitude buildup (PT1)
else:
    
    x0 = np.hstack((x0, m_init * g))
    Nx_sim = 17
    Nx_MPC = 16  # Mpc has 2 less states than the plant model used for simulation, as it does not take into account mass variation and thrust magnitude buildup (PT1)
    if SCVX_Flag==1:
        Nx_MPC = 14  # Mpc has 2 less states than the plant model used for simulation, as it does not take into account mass variation and thrust magnitude buildup (PT1)
        Nx_sim = 15 

#Hopper Inertia
J1 = 1 / 12 * m_init * (height**2 + 3 * radius**2)
J2 = J1
J3 = 0.5 * m_init * radius**2
J_B = np.diag(np.array([J1, J2, J3]))


max_gimbal = 10 #Max TVC deflection [Deg]
cos_delta_max = np.cos(np.deg2rad(max_gimbal))

alpha_thrust = 1.33 #Coefficient of proportionallity between thrust magnitude andd mass variation.
max_thrust = 2700  # Maximum thrust value
min_Thrust = 1100 # Minimum thrust value
thrust_time_constant = 0.25 # Time constant of the first-order system used to model thrust dynamics.
max_Angular_Rates = 60 * (pi / 180)
Max_Tdot = 3000  # (Max_T - Min_T)/ 0.5 s


Nu = 3


#MPC Parameters
N_mpc = 20
T_mpc = 2
h_mpc = T_mpc / N_mpc


Q = np.eye(13)
R = np.eye((Nu))

Controller_Frequency = 20 #[Hz]

Input_Uncertainty = 0#Percentage of uncertainty in the input channel.


parameters = {
    "x0": x0.tolist(),
    "u0": u0.tolist(),
    "h_reference": float(h_reference),
    "Nt": int(Nt),
    "X_ref": X_ref.tolist(),
    "U_ref": U_ref.tolist(),
    "t_ref": t_ref.tolist(),
    "g": float(g),
    "J1": float(J1),
    "J2": float(J2),
    "J3": float(J3),
    "g_I": g_I.tolist(),
    "height": float(height),
    "diameter": float(diameter),
    "radius": float(radius),
    "m_init": float(m_init),
    "m": float(m),
    "r_T_B": r_T_B.tolist(),
    "J_B": J_B.tolist(),
    "max_gimbal": float(max_gimbal),
    "cos_delta_max": float(cos_delta_max),
    "alpha_thrust": float(alpha_thrust),
    "max_thrust": float(max_thrust),
    "min_thrust": float(min_Thrust),
    "thrust_time_constant": float(thrust_time_constant),
    "max_angular_rates": float(max_Angular_Rates),
    "max_Tdot": float(Max_Tdot),
    "Nx_sim": int(Nx_sim),
    "Nx_MPC": int(Nx_MPC),
    "Nu": int(Nu),
    "N_mpc": int(N_mpc),
    "T_mpc": float(T_mpc),
    "h_mpc": float(h_mpc),
    "Q": Q.tolist(),
    "R": R.tolist(),
    "controller_frequency": int(Controller_Frequency),
    "Input_Uncertainty": float(Input_Uncertainty),
    "Mass_Flag": int(Mass_Flag),
    "SCVX_Flag": int(SCVX_Flag),
}


with open("params.json", "w") as json_file:
    json.dump(parameters, json_file, indent=4)
