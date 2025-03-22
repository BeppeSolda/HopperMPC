import sympy as sp
import numpy as np
import cvxpy as cvx
from utils import euler_to_quat
from params import K
import jax.numpy as jnp
from jax import  jacfwd,jit
from functools import partial




def skew(v):
    return sp.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def dir_cosine(q):
    return sp.Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])


def omega(w):
    return sp.Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])


class Model:
    """
    A 6 degree of freedom rocket landing problem.
    """
    Nx = 14
    Nu = 3

    # Mass
    # m_wet = 30000.  # 30000 kg
    # m_dry = 22000.  # 22000 kg
    m_wet = 150 + 82
    m_dry = 82
    # Flight time guess
    t_f_guess = 15.  # 10 s

    
    vector = np.concatenate(([1], -np.ones(3)))

# Create the diagonal matrix
    T = np.diag(vector)

    zeros_row = np.zeros((1, 3))
    I = np.eye(3)

    # Vertically stack the zeros_row and the identity matrix
    H = np.vstack((zeros_row, I))
    r_I_init = np.array((10., 5., 30.))  # 2000 m, 200 m, 200 m
    v_I_init = np.array((0., 2., -3.))  # -300 m/s, 50 m/s, 50 m/s
    q_B_I_init = np.array((1.,0., 0., 0.))
    w_B_init = np.deg2rad(np.array((0., 0., 0.)))

    r_I_final = np.array((0., 0., 0.))
    v_I_final = np.array((0., 0., 0.))
    q_B_I_final = np.array((1.,0., 0., 0.))
    w_B_final = np.deg2rad(np.array((0., 0., 0.)))

    w_B_max = np.deg2rad(90)
    
    # Angles
    max_gimbal = 10
    max_angle = 70
    glidelslope_angle = 20

    tan_delta_max = np.tan(np.deg2rad(max_gimbal))
    cos_delta_max = np.tan(np.deg2rad(max_gimbal))
    cos_theta_max = np.cos(np.deg2rad(max_angle))
    tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

    # Thrust limits
    # T_max = 800000.  # 800000 [kg*m/s^2]
    # T_min = T_max * 0.4
    T_max = 2700
    T_min = 1100

    # Angular moment of inertia
    #J_B = np.diag([4000000., 4000000., 100000.])  # 100000 [kg*m^2], 4000000 [kg*m^2], 4000000 [kg*m^2]
    height = 2.5
    diameter = 0.3
    radius  = diameter/2
    J1 = 1 / 12 * m_wet * (height**2 + 3 * radius** 2)
    J2 = J1
    J3 = 0.5 * m_wet * radius** 2
    J_B = np.diag(np.array([J1, J2, J3]))

    # Gravity
    g_I = np.array((0., 0., -9.81))  # -9.81 [m/s^2]

    # Fuel consumption
    alpha_m = 1 / (282 * 9.81)  # 1 / (282 * 9.81) [s/m]
    #alpha_m = 1.33
    # Vector from thrust point to CoM
    #r_T_B = np.array([0., 0., -14.])  # -20 m
    r_T_B = np.array([0, 0, -0.5])
    h = 1. / (K - 1) * t_f_guess

    def set_random_initial_state(self):
        self.r_I_init[2] = 50
        self.r_I_init[0:2] = np.random.uniform(-10, 10, size=2)

        self.v_I_init[2] = np.random.uniform(-10, -6)
        self.v_I_init[0:2] = np.random.uniform(-0.5, -0.2, size=2) * self.r_I_init[0:2]

        self.q_B_I_init = euler_to_quat((np.random.uniform(-30, 30),
                                         np.random.uniform(-30, 30),
                                         0))
        self.w_B_init = np.deg2rad((np.random.uniform(-20, 20),
                                    np.random.uniform(-20, 20),
                                    0))

    # ------------------------------------------ Start normalization stuff
    def __init__(self):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        #self.set_random_initial_state()

        self.x_init = np.concatenate(((self.m_wet,), self.r_I_init, self.v_I_init, self.q_B_I_init, self.w_B_init))
        self.x_final = np.concatenate(((self.m_dry,), self.r_I_final, self.v_I_final, self.q_B_I_final, self.w_B_final))

        self.r_scale = np.linalg.norm(self.r_I_init)
        self.m_scale = self.m_wet

        # slack variable for linear constraint relaxation
        self.s_prime = cvx.Variable((K, 1), nonneg=True)

        # slack variable for lossless convexification
        # self.gamma = cvx.Variable(K, nonneg=True)

    def nondimensionalize(self):
        """ nondimensionalize all parameters and boundaries """

        self.alpha_m *= self.r_scale  # s
        self.r_T_B /= self.r_scale  # 1
        self.g_I /= self.r_scale  # 1/s^2
        self.J_B /= (self.m_scale * self.r_scale ** 2)  # 1

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

        self.T_max = self.u_nondim(self.T_max)
        self.T_min = self.u_nondim(self.T_min)

        self.m_wet /= self.m_scale
        self.m_dry /= self.m_scale

    def x_nondim(self, x):
        """ nondimensionalize a single x row """

        x[0] /= self.m_scale
        x[1:4] /= self.r_scale
        x[4:7] /= self.r_scale

        return x

    def u_nondim(self, u):
        """ nondimensionalize u, or in general any force in Newtons"""
        return u / (self.m_scale * self.r_scale)

    def redimensionalize(self):
        """ redimensionalize all parameters """

        self.alpha_m /= self.r_scale  # s
        self.r_T_B *= self.r_scale
        self.g_I *= self.r_scale
        self.J_B *= (self.m_scale * self.r_scale ** 2)

        self.T_max = self.u_redim(self.T_max)
        self.T_min = self.u_redim(self.T_min)

        self.m_wet *= self.m_scale
        self.m_dry *= self.m_scale

    def x_redim(self, x):
        """ redimensionalize x, assumed to have the shape of a solution """

        x[0, :] *= self.m_scale
        x[1:4, :] *= self.r_scale
        x[4:7, :] *= self.r_scale

        return x

    def u_redim(self, u):
        """ redimensionalize u """
        return u * (self.m_scale * self.r_scale)

  
    
    def qtorp_NJ(self,q):
        return q[1:4] / q[0]
    
    @partial(jit, static_argnums=(0,))
    def skew(self,v):
        return jnp.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @partial(jit, static_argnums=(0,))
    def L(self,q):
        s = q[0]
        v = q[1:4]
        skew_v = self.skew(v)
        v = v.reshape(3, 1)
        return jnp.block([[s, -jnp.transpose(v)], [v, s * jnp.eye(3) + skew_v]])

    @partial(jit, static_argnums=(0,))
    def qtoQ(self,q):
        return jnp.transpose(self.H) @ self.T @ self.L(q) @ self.T @ self.L(q) @ self.H

    @partial(jit, static_argnums=(0,))
    def G(self,q):
        return self.L(q) @ self.H

    @partial(jit, static_argnums=(0,))
    def rptoq(self,phi):
        phi_norm_sq = jnp.dot(phi.T, phi)
        scalar_part = 1 / jnp.sqrt(1 + phi_norm_sq)
        vector_part = scalar_part * phi
        return jnp.concatenate(([scalar_part], vector_part))

    @partial(jit, static_argnums=(0,))
    def qtorp(self,q):
        return q[1:4] / q[0]

   





    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
            r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
            v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
            U[:, k] = (self.T_max - self.T_min) / 2 * np.array([0, 0, 1])

        return X, U

   
    def get_objective(self):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        return cvx.Minimize(1e5 * cvx.sum(self.s_prime))

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            X_v[7:11, 0] == self.x_init[7:11],
            X_v[11:14, 0] == self.x_init[11:14],

            # X_[0, -1] == self.x_final[0], # final mass is free
            X_v[1:, -1] == self.x_final[1:],
            # U_v[1:3, -1] == 0,
        ]

        constraints += [
            # State constraints:
            X_v[0, :] >= self.m_dry,  # minimum mass
            cvx.norm(X_v[1: 3, :], axis=0) <= X_v[3, :] / self.tan_gamma_gs,  # glideslope
            cvx.norm(X_v[8:10, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle
            cvx.norm(X_v[11: 14, :], axis=0) <= self.w_B_max,  # maximum angular velocity

            # Control constraints:
            cvx.norm(U_v[0:2, :], axis=0) <= self.tan_delta_max * U_v[2, :],  # gimbal angle constraint
            #self.cos_delta_max * self.gamma <= U_v[2, :],

            cvx.norm(U_v, axis=0) <= self.T_max,  # upper thrust constraint
            # U_v[2, :] >= self.T_min  # simple lower thrust constraint

            # # Lossless convexification:
            # self.gamma <= self.T_max,
            # self.T_min <= self.gamma,
            # cvx.norm(U_v, axis=0) <= self.gamma
        ]

        # linearized lower thrust constraint
        lhs = [U_last_p[:, k] / (cvx.norm(U_last_p[:, k])) * U_v[:, k] for k in range(K)]
        constraints += [
            self.T_min - cvx.vstack(lhs) <= self.s_prime
        ]

        return constraints

    def get_linear_cost(self):
        cost = np.sum(self.s_prime.value)
        return cost

    def get_nonlinear_cost(self, X=None, U=None):
        magnitude = np.linalg.norm(U, 2, axis=0)
        is_violated = magnitude < self.T_min
        violation = self.T_min - magnitude
        cost = np.sum(is_violated * violation)
        return cost
    
    def rocket_dynamics_rk4(self,x, u):
        f1 = self.rocket_dynamics(x, u)
        f2 = self.rocket_dynamics(x + 0.5 * self.h * f1, u)
        f3 = self.rocket_dynamics(x + 0.5 * self.h * f2, u)
        f4 = self.rocket_dynamics(x + self.h * f3, u)
        xn = x + (self.h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        q_norm = np.linalg.norm(xn[7:11])
        xn = xn.at[7:11].set(xn[7:11] / q_norm)
        return xn
    @partial(jit, static_argnums=(0,))
    def rocket_dynamics(self,x, u):
        
        r = x[1:4]
        q = x[7:11]
        v = x[4:7]
        q_norm = jnp.linalg.norm(q)
        q = q / q_norm
        w = x[11:14]
        m = x[0]
    
        f = jnp.zeros(self.Nx)
        Q = self.qtoQ(q)
        Q_t = Q.T

        f = f.at[1:4].set(Q.dot(v))
        f = f.at[7:11].set(0.5 * jnp.dot(self.L(q), jnp.dot(self.H, w)))
       
        f = f.at[4:7].set(((1 / m) * u) + Q_t.dot(self.g_I) - self.skew(w).dot(v))
        ang_vel_dynamics = jnp.linalg.inv(self.J_B).dot(self.skew(self.r_T_B).dot(u) - self.skew(w).dot(self.J_B).dot(w))
        
        f = f.at[11:14].set(ang_vel_dynamics)
        thrust_command_norm = jnp.linalg.norm(u[0:3])
        mass_flow_rate = self.alpha_m*(thrust_command_norm/self.T_max)
        f = f.at[0].set(-mass_flow_rate )
        
        
        return f
    
    
    @partial(jit, static_argnums=(0,))
    def compute_jacobiaNx(self,x, u):
        dynamics_fn = lambda x: self.rocket_dynamics(x, u)
        jacobian_fn = jacfwd(jit(dynamics_fn))
        return jacobian_fn(x)

    @partial(jit, static_argnums=(0,))
    def compute_jacobiaNu(self, x, u):
        dynamics_fn = lambda u: self.rocket_dynamics(x, u)
        jacobian_fn = jacfwd(jit(dynamics_fn))
        return jacobian_fn(u)

    
    
    def get_jacobians_single(self,x, u):
        
        f = self.rocket_dynamics(x,u)
        A = self.compute_jacobiaNx(x, u)
        B= self.compute_jacobiaNu(x, u)
        return f,A,B
        
      

    def integrate_nonlinear_piecewise(self, K,X_l, U):
        """
        Piecewise integration to verfify accuracy of linearization.
        :param X_l: Linear state evolution
        :param U: Linear input evolution
        :return: The piecewise integrated dynamics
        """
       
        X_nl = np.zeros_like(X_l)
        X_nl[:,0] = X_l[:,0]

        for k in range( K-1):
             X_nl[:, k + 1] = self.rocket_dynamics_rk4(X_l[:, k], U[:, k])

        return X_nl

    
    def integrate_nonlinear_full(self, K,x0, U):
        """
        Simulate nonlinear behavior given an initial state and an input over time.
        :param x0: Initial state
        :param U: Linear input evolution
        :return: The full integrated dynamics
        """
        X_nl = np.zeros([x0.size, K])
        X_nl[:, 0] = x0

        for k in range(K - 1):
           X_nl[:, k + 1] = self.rocket_dynamics_rk4(X_nl[:, k], U[:, k])

        return X_nl
