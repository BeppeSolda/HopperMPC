import numpy as np
import cvxpy as cvx
from Rocket_Model import Model
from params import Nx,Nu,K,Max_iter,tol,rho0,rho1,rho2,alpha,beta,tr_radius
from SCP_setup import SCP_problem
from Discretization import ZOH
from utils import format_line
from rocket_landing_3d_plot import plot
from PlotSave_Animation import save_animation
from PlotAll import plotAll

#model = Rocket_Model(K, tf_guess )
model = Model()
model.nondimensionalize()
tf_guess = model.t_f_guess

X = np.zeros([model.Nx,K])
U = np.zeros([model.Nu,K])

X,U = model.initialize_trajectory(X, U)
all_X = [model.x_redim(X.copy())]
all_U = [model.u_redim(U.copy())]

t = np.linspace(0,tf_guess, K)
#plot_init(X,U,t)


integrator  = ZOH(model,K,tf_guess)
problem = SCP_problem(model,K,X,U)

problem.par['tr_radius'].value = tr_radius
problem.par['weight_nu'].value = 1e5

last_nonlinear_cost = None
converged = False

for iter in range(Max_iter):

    
   
    problem.par['X_last'].value = X
    problem.par['U_last'].value = U
    XX = model.x_redim(X.copy())
    UU = model.u_redim(U.copy())
    
   
    A_d1,B_d1,r_d1  = integrator.compute_discrete_time_matrices(X, U)
    
    problem.par['A_d'].value = A_d1
    problem.par['B_d'].value = B_d1
    problem.par['r_d'].value = r_d1
    

    while True:
        error = problem.prob.solve(verbose=False, solver= cvx.ECOS, max_iters=200)
        print(format_line('Solver Error', error))

        # get solution
        X_new = problem.var['X'].value
        U_new = problem.var['U'].value
       

        X_nl = model.integrate_nonlinear_piecewise(K,X_new, U_new)

        linear_cost_dynamics = np.linalg.norm(problem.var['nu'].value, 1)
        nonlinear_cost_dynamics = np.linalg.norm(X_new - X_nl, 1)

        linear_cost_constraints = model.get_linear_cost()
        nonlinear_cost_constraints = model.get_nonlinear_cost(X=X_new, U=U_new)

        linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
        nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

        if last_nonlinear_cost is None:
            last_nonlinear_cost = nonlinear_cost
            X = X_new
            U = U_new
    
            break

        actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
        predicted_change = last_nonlinear_cost - linear_cost  # delta_L

        print('')
        print(format_line('Virtual Control Cost', linear_cost_dynamics))
        print(format_line('Constraint Cost', linear_cost_constraints))
        print('')
        print(format_line('Actual change', actual_change))
        print(format_line('Predicted change', predicted_change))
        print('')
        

        if abs(predicted_change) < 1e-4:
            converged = True
            break
        else:
            rho = actual_change / predicted_change
            if rho < rho0:
                # reject solution
                tr_radius /= alpha
                print(f'Trust region too large. Solving again with radius={tr_radius}')
            else:
                # accept solution
                X = X_new
                U = U_new
               

                print('Solution accepted.')

                if rho < rho1:
                    print('Decreasing radius.')
                    tr_radius /= alpha
                elif rho >= rho2:
                    print('Increasing radius.')
                    tr_radius *= beta

                last_nonlinear_cost = nonlinear_cost
                break
            problem.par['tr_radius'].value = tr_radius
    
    all_X.append(model.x_redim(X.copy()))
    all_U.append(model.u_redim(U.copy()))
    

    if converged:
        print(f'Converged after {iter + 1} iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)
all_sigma = np.ones(K) * tf_guess

if not converged:
    print('Maximum number of iterations reached without convergence.')


save_animation(all_X, all_U, filename="rocket_trajectory.gif", fps=1)

# plot trajectory
plot(all_X, all_U, all_sigma)
file_path = 'Reference_trajectories/'
delta_t = tf_guess / (K - 1)

# Create the time vector from 0 to tf with K points
t = np.linspace(0, tf_guess, K)
iterations_to_converge = all_X.shape[0] 
# Create new variables excluding the first column
all_X_no_first_col = all_X[-1, :, :]  # Exclude the first feature from all_X
all_U_no_first_col = all_U[-1, :, :] # Exclude the first feature from all_U
plotAll(all_X_no_first_col,all_U_no_first_col,t)
#plot_mios(t,all_X_no_first_col,all_U_no_first_col)


# Save the reference trajectories all_X, all_U, and tf_guess as .npz file
np.savez(file_path + 'SCVX_0_20_60_tf60.npz', X=all_X_no_first_col, U=all_U_no_first_col, time_vect = t)
print(f'Reference trajectories saved to {file_path}SCVX_0_20_60_tf60.npz')


