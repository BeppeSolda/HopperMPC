
import jax.numpy as jnp
import numpy as np
import scipy as sp

# Constants
g = 9.81
g_I = jnp.array([0, 0, -g])  # Gravity vector in the inertial frame

height = 2.5
diameter = 0.3
radius  = diameter/2

# Inertia matrix for the rocket body
m_init =  150 + 82
m = m_init
r_T_B = jnp.array([0, 0, -0.5])






J1 = 1 / 12 * m_init * (height**2 + 3 * radius** 2)
J2 = J1
J3 = 0.5 * m_init * radius** 2
J_B = jnp.diag(jnp.array([J1, J2, J3]))


# Transform matrices
vector = np.concatenate(([1], -np.ones(3)))

# Create the diagonal matrix
T = np.diag(vector)

zeros_row = np.zeros((1, 3))
I = np.eye(3)

# Vertically stack the zeros_row and the identity matrix
H = np.vstack((zeros_row, I))
cos_delta_max = np.cos(np.deg2rad(15))
max_gimbal = 10
tan_delta_max = np.tan(np.deg2rad(max_gimbal))
alpha_Thrust = 1.33
# MPC parameters
K = 200
Nx = 17

Nu = 3



Max_Thrust = 2700  # Maximum thrust value
Min_Thrust = 1100
Thrust_PT1_TimeConstant = 0.25

Max_iter = 50 
tol = 1e-4
tr_radius =10

rho0 = 0.0
rho1 = 0.25
rho2 = 0.9
alpha = 2.0
beta = 3.2

