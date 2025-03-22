import casadi as ca
import numpy as np
import json

"""
This script generates Jacobians and an RK4 function using CasADi, with support for code generation for real-time applications.

To generate the C functions for RK4 integration and Jacobian computation, run this script.

There are two modes:

- For SCVx reference trajectory: Set `Flag = 'SCVx'` and call `SCVx_gen()`, then execute the script.
- For Thorpy reference trajectory: Set `Flag` to any value other than 'SCVx' and call `Thorpy_gen()`, then execute the script.

After generating the C files, move them to the `Codegen/Casadi` directory and compile them using:

    gcc -fPIC -shared function.c -o function.so
"""



def Validate_Compiled_Functions(A_func, B_func, rk4_func,Flag):
    """
    This function validates the compiled A, B, and RK4 functions by comparing them to their corresponding
    CasADi-based implementations using a predefined state and control input.

    Args:
    A_func (CasADi function): A matrix function.
    B_func (CasADi function):  B matrix function.
    rk4_func (CasADi function):  RK4 function.

    Returns:
    None: Prints differences and checks if the discrepancies are below a tolerance level.
    """
    if Flag == 'SCVx':
        x0 = ca.DM(
            [
                1.20934614e-27,
                0.00000000e00,
                -1.05291667e00,
                1.00000000e00,
                0.00000000e00,
                -2.87141940e-28,
                0.00000000e00,
                -5.11183762e-27,
                6.87754316e-26,
                -5.87129189e-54,
                6.53189695e-26,
                4.85493087e-27,
                9.27301538e-69,
                232,
            ]
        )
    else:
        x0 = ca.DM(
            [
                1.20934614e-27,
                0.00000000e00,
                -1.05291667e00,
                1.00000000e00,
                0.00000000e00,
                -2.87141940e-28,
                0.00000000e00,
                -5.11183762e-27,
                6.87754316e-26,
                -5.87129189e-54,
                6.53189695e-26,
                4.85493087e-27,
                9.27301538e-69,
                -1.99860852e-25,
                -4.16550073e-23,
                2.27592000e03,
            ]
        )
        
    u0 = ca.DM([3.86998209e-102, -3.24945006e-103, 1.76284460e-023])  # Control input
    # Time step
    if Flag == 'SCVx':
        A_funcgen = ca.external("A_func_SCVx", "Codegen/Casadi/A_func_SCVx.so")
        B_funcgen = ca.external("B_func_SCVx", "Codegen/Casadi/B_func_SCVx.so")
        rk4_funcgen = ca.external("rk4_func_SCVx", "Codegen/Casadi/rk4_func_SCVx.so")
    else:
        A_funcgen = ca.external("A_func", "Codegen/Casadi/A_func.so")
        B_funcgen = ca.external("B_func", "Codegen/Casadi/B_func.so")
        rk4_funcgen = ca.external("rk4_func", "Codegen/Casadi/rk4_func.so")

    xn_val_gen = rk4_funcgen(x0, u0, h_mpc)  # Next state
    A_val_gen = A_funcgen(x0, u0, h_mpc)  # A matrix (Jacobian wrt state)
    B_val_gen = B_funcgen(x0, u0, h_mpc)  # B matrix (Jacobian wrt control)

    xn_val = rk4_func(x0, u0, h_mpc)  # Next state
    A_val = A_func(x0, u0, h_mpc)  # A matrix (Jacobian wrt state)
    B_val = B_func(x0, u0, h_mpc)  # B matrix (Jacobian wrt control)
    diff_xn = ca.norm_fro(xn_val - xn_val_gen)
    diff_A = ca.norm_fro(A_val - A_val_gen)
    diff_B = ca.norm_fro(B_val - B_val_gen)

    print(f"Difference in next state (RK4): {diff_xn}")
    print(f"Difference in A matrix (Jacobian wrt x): {diff_A}")
    print(f"Difference in B matrix (Jacobian wrt u): {diff_B}")

    # Check if the differences are very small (within a tolerance)

    tolerance = 1e-6  # Set a tolerance for small differences
    if diff_xn < tolerance and diff_A < tolerance and diff_B < tolerance:
        print("Code generation successful!")
    else:
        print("There are discrepancies between the compiled and CasADi functions.")
    print("Next state (RK4):", xn_val)
    print("A matrix (Jacobian wrt x):", A_val)
    print("B matrix (Jacobian wrt u):", B_val)


def qtoM_WB(q):
    """
    Converts a quaternion to a rotation matrix from the body frame to the world frame.

    Args:
    q (CasADi SX vector): Quaternion representing the rotation.

    Returns:
    CasADi SX matrix: The corresponding rotation matrix (3x3).
    """
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Q = ca.SX(3, 3)
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


def skew(v):
    """
    Computes the skew-symmetric matrix of a 3D vector.

    Args:
    v (CasADi SX vector): Input vector of length 3.

    Returns:
    CasADi SX matrix: The corresponding 3x3 skew-symmetric matrix.
    """
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def Omega(q):
    """
    Computes the Omega matrix (used in quaternion-based angular velocity transformations).

    Args:
    q (CasADi SX vector): Quaternion representing the rotation.

    Returns:
    CasADi SX matrix: The corresponding Omega matrix (4x3).
    """

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Omega_Mat = ca.SX.zeros(4, 3)
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


def rocket_dynamics_thorpy(x, u):
    """
    Equations of motion of the hopper used to generate the Jacobians for tracking Thorpy trajectories. 
    Mass variation is not taken into account, thrust components are treated as states, and thrust rates are the control inputs.

    Args:
    x (CasADi SX vector): State vector containing position, quaternion, velocity, angular velocity, and thrust components.
    u (CasADi SX vector): Control input vector, which consists of thrust rates.

    Returns:
    CasADi SX vector: The derivative of the state vector (f), representing the rate of change of the states.
"""
    r = x[0:3]  # Position
    q = x[3:7]  # Quaternion
    v = x[7:10]  # Velocity
    w = x[10:13]  # Angular velocity
    thrust = x[13:16]  # Thrust components
    q = q / ca.norm_2(q)

    f = ca.SX.zeros(Nx)

    M_WB = qtoM_WB(q)
    M_BW = M_WB.T


    f[0:3] = M_WB @ v  # Position derivative (r_dot)
    f[3:7] = 0.5 * Omega(q) @ w  # Quaternion derivative (q_dot)
    f[7:10] = (1 / m) * thrust + M_BW @ g_I - skew(w) @ v  # Velocity derivative (v_dot)
    ang_vel_dynamics = ca.mtimes(ca.inv(J_B), skew(r_T_B) @ thrust - skew(w) @ J_B @ w)
    f[10:13] = ang_vel_dynamics  # Angular velocity derivative (w_dot)
    f[13:16] = u  # Thrust derivative (thrust_dot)

    return f
def rocket_dynamics_SCVx(x, u):
    """"
    Equations of motion of the hopper used to generate the Jacobians for tracking SCVX trajectories. 
    In this case, mass variation is taken into account, and thrusts are considered as the control inputs, 
    rather than states.

    Args:
    x (CasADi SX vector): State vector containing position, quaternion, velocity, angular velocity, 
                           and mass of the hopper.
    u (CasADi SX vector): Control input vector, which consists of thrust components.

    Returns:
    CasADi SX vector: The derivative of the state vector (f), representing the rate of change of the states.
    """
    r = x[0:3]   # Position
    q = x[3:7]   # Quaternion
    v = x[7:10]  # Velocity
    w = x[10:13] # Angular velocity
    #thrust = x[13:16]  # Thrust components
    m = x[13]

    # Normalize quaternion
    q = q / ca.norm_2(q)
    J1 = 1 / 12 * m * (height**2 + 3 * radius** 2)
    J2 = J1
    J3 = 0.5 * m * radius** 2
    J_B = ca.diag(ca.vertcat(J1, J2, J3))
    # Initialize state derivatives f
    
    f = ca.SX.zeros(Nx)
    
    # Rotation matrix from quaternion
    M_WB = qtoM_WB(q)
    M_BW = M_WB.T
    
    # Dynamics equations
    f[0:3] = M_WB @ v  # Position derivative (r_dot)
    f[3:7] = 0.5 * Omega(q) @ w  # Quaternion derivative (q_dot)
    f[7:10] = (1 / m) * u + M_BW @ g_I - skew(w) @ v  # Velocity derivative (v_dot)
    ang_vel_dynamics = ca.mtimes(ca.inv(J_B), skew(r_T_B) @ u - skew(w) @ J_B @ w)
    f[10:13] = ang_vel_dynamics  # Angular velocity derivative (w_dot)
    
    thrust_command_norm = ca.norm_2(u)
    
    f[13]= -alpha_thrust*(thrust_command_norm/max_thrust)


    return f
def rocket_dynamics_SCVx_rk4(x, u, h):
    f1 = rocket_dynamics_SCVx(x, u)
    f2 = rocket_dynamics_SCVx(x + 0.5 * h * f1, u)
    f3 = rocket_dynamics_SCVx(x + 0.5 * h * f2, u)
    f4 = rocket_dynamics_SCVx(x + h * f3, u)
    xn = x + (h / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    q_norm = ca.norm_2(xn[3:7])
    xn[3:7] = xn[3:7] / q_norm

    return xn

def rocket_dynamics_rk4_thorpy(x, u, h):
    f1 = rocket_dynamics_thorpy(x, u)
    f2 = rocket_dynamics_thorpy(x + 0.5 * h * f1, u)
    f3 = rocket_dynamics_thorpy(x + 0.5 * h * f2, u)
    f4 = rocket_dynamics_thorpy(x + h * f3, u)
    xn = x + (h / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    q_norm = ca.norm_2(xn[3:7])
    xn[3:7] = xn[3:7] / q_norm

    return xn



def Thorpy_Gen(x,u,h,opts):
    """
    Generate c code for the rk4 function and jacobians to track Thorpy reference trajectories.
    
    """
    xn = rocket_dynamics_rk4_thorpy(x, u, h)
    A = ca.jacobian(xn, x)  
    B = ca.jacobian(xn, u)  

    rk4_func = ca.Function("rk4_func", [x, u, h], [xn])
    A_func = ca.Function("A_func", [x, u, h], [A])
    B_func = ca.Function("B_func", [x, u, h], [B])

    A_func.generate("A_func.c", opts)
    B_func.generate("B_func.c", opts)
    rk4_func.generate("rk4_func.c", opts)
    return A_func,B_func,rk4_func

def SCVx_Gen(x,u,h,opts):
    """
    Generate c code for the rk4 function and jacobians to track SCVx reference trajectories.

    """

    xn = rocket_dynamics_SCVx_rk4(x, u, h)
    A = ca.jacobian(xn, x)  
    B = ca.jacobian(xn, u) 

    rk4_func_SCVx = ca.Function("rk4_func_SCVx", [x, u, h], [xn])
    A_func_SCVx = ca.Function("A_func_SCVx", [x, u, h], [A])
    B_func_SCVx = ca.Function("B_func_SCVx", [x, u, h], [B])

    A_func_SCVx.generate("A_func_SCVx.c", opts)
    B_func_SCVx.generate("B_func_SCVx.c", opts)
    rk4_func_SCVx.generate("rk4_func_SCVx.c", opts)

    return A_func_SCVx,B_func_SCVx,rk4_func_SCVx

with open("params.json", "r") as f:
    params = json.load(f)
height = params["height"]
radius = params["radius"]
alpha_thrust = params["alpha_thrust"]
max_thrust = params["max_thrust"]
N_mpc = params["N_mpc"]
T_mpc = params["T_mpc"]
h_mpc = params["h_mpc"]
m = params["m"]

Nx_MPC = params["Nx_MPC"]
Nu = params["Nu"]
g_I = np.array(params["g_I"])
r_T_B = np.array(params["r_T_B"])

Flag = 'SCVx'

if Flag!='SCVx':
    J1 = params["J1"]
    J2 = params["J2"]
    J3 = params["J3"]
    J_B = ca.diag([J1, J2, J3])

Nx = Nx_MPC  # State dimension

x = ca.SX.sym("x", Nx)  # State vector
u = ca.SX.sym("u", Nu)  # Control vector
h = ca.SX.sym("h")  # Time step
opts = {"casadi_real": "double"}


#A_func,B_func,rk4_func = Thorpy_Gen(x,u,h,opts)
A_func_SCVx,B_func_SCVx,rk4_func_SCVx = SCVx_Gen(x,u,h,opts)


Validate_Compiled_Functions(A_func_SCVx, B_func_SCVx, rk4_func_SCVx,'SCVx')
