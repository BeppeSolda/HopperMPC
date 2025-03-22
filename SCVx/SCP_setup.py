import cvxpy as cvx
import numpy as np

class SCP_problem:


    def __init__(self,m,K,X,U):

      self.var = dict()
      self.var['X'] = cvx.Variable((m.Nx, K))
      self.var['U'] = cvx.Variable((m.Nu, K))
      self.var['nu'] = cvx.Variable((m.Nx, K - 1))

      # Parameters:
      self.par = dict()
      self.par['A_d'] = cvx.Parameter((m.Nx * m.Nx, K - 1))
      self.par['B_d'] = cvx.Parameter((m.Nx * m.Nu, K - 1))
      
      self.par['r_d'] = cvx.Parameter((m.Nx, K - 1))

      self.par['X_last'] = cvx.Parameter((m.Nx, K))
      self.par['U_last'] = cvx.Parameter((m.Nu, K))

      self.par['weight_nu'] = cvx.Parameter(nonneg=True)
      self.par['tr_radius'] = cvx.Parameter(nonneg=True)

        # Constraints:
      constraints = []
      constraints += m.get_constraints(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])


      
      m_obj = m.get_objective(self.var['X'])
      scp_obj = cvx.Minimize( self.par['weight_nu']  * cvx.norm(self.var['nu'],1))
      obj = m_obj + scp_obj


      constraints = []
      # self.X_last_iter.value = X
      # self.U_last_iter.value = U
      constraints += m.get_constraints(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])

      # A_list,B_list = m.get_jacobians(X, U)
      # self.Assign_Linearized_Matrices(A_list,B_list,K)
      
      # constraints += [self.X_last_iter [:,i+1] + self.DeltaX[:,i+1] == m.rocket_dynamics_rk4(X[:,i],U[:,i]) 
      #                 + self.A[i]@self.DeltaX[:,i] 
      #                 + self.B[i]@self.DeltaU[:,i] + self.nu[:,i]
                      
      #                  for i in range(K - 1)]
      constraints += [
            self.var['X'][:, k + 1] ==
            cvx.reshape(self.par['A_d'][:, k], (m.Nx, m.Nx)) @ self.var['X'][:, k]
            + cvx.reshape(self.par['B_d'][:, k], (m.Nx, m.Nu)) @ self.var['U'][:, k]
           
            + self.par['r_d'][:, k]
            + self.var['nu'][:, k]
            for k in range(K - 1)
        ]
      
      
     
      du = self.var['U'] - self.par['U_last']
      dx = self.var['X'] - self.par['X_last']
      constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) <= self.par['tr_radius']]

      
      self.prob = cvx.Problem(obj,constraints)

    def Assign_Linearized_Matrices(self,A_list,B_list,K):
        # self.TempA = np.array(A_list[3])
        for i in range(K - 1):
            # if i > 1:
            #    self.TempA = self.A[i].value
            self.A[i].value = np.array(A_list[i])
            self.B[i].value = np.array(B_list[i])
            # print('Assigned different matrices')  
            # print(np.linalg.norm(self.TempA - self.A[i].value))
            
       
            


        


  