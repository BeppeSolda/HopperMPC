import numpy as np


class ZOH:
    def __init__(self,m,K,tf):
        self.K = K
        self.m = m
        self.Nx =  m.Nx
        self.Nu = m.Nu
        self.tf = tf
        self.h = 1/(K-1) * tf

        self.A_d = np.zeros([self.Nx  * self.Nx , K - 1])
        self.B_d = np.zeros([self.Nx  * self.Nu, K - 1])
        self.r_d = np.zeros([m.Nx, K - 1])

        self.A_c = np.zeros([self.Nx  * self.Nx , K - 1])
        self.B_c = np.zeros([self.Nx  * self.Nu, K - 1])
        self.r_c = np.zeros([m.Nx, K - 1])
        
        x_end = m.Nx
        A_d_end = m.Nx*(1+m.Nx)
        B_d_end = m.Nx*(1+m.Nx + m.Nu)
        r_d_end = m.Nx*(1+m.Nx + m.Nu + 1)

        self.x_slice = slice(0, x_end)            # Indices [0:x_end]
        self.A_d_slice = slice(x_end, A_d_end)     # Indices [x_end:A_d_end]
        self.B_d_slice = slice(A_d_end, B_d_end)   # Indices [A_d_end:B_d_end]
        self.r_d_slice = slice(B_d_end, r_d_end)   # Indices [B_d_end:r_d_end]

        self.P0 = np.zeros((m.Nx*(1+m.Nx + m.Nu + 1),))

        #NB reshape (-1) reshapes the array into a one dim array (so it flattens it)) 
        # while keeping all original data. (-1) means "figure out the size of this dimension automatically based on the number of elements,"
        self.P0[self.A_d_slice] = np.eye((self.Nx)).reshape(-1)


    
    def compute_discrete_time_matrices(self,X, U):

        for k in range(self.K - 1):
            self.P0[self.x_slice] = X[:,k]
           
            P = self.P_dot_rk4(self.P0,U[:, k])
            
            Phi = P[self.A_d_slice].reshape((self.Nx,self.Nx))
            self.A_d [:,k] = Phi.flatten(order='F')
 

            self.B_d[:, k] = (Phi @ P[self.B_d_slice].reshape((self.Nx,self.Nu))).flatten(order='F')
            self.r_d[:, k] = (Phi @ P[self.r_d_slice])
          

        return self.A_d, self.B_d, self.r_d

    
    def P_dot(self,P,u):
        x = P[self.x_slice]
        self.f_c,self.A_c,self.B_c = self.m.get_jacobians_single(x,u)
        
        P_inv = np.linalg.inv(P[self.A_d_slice].reshape((self.Nx,self.Nx)))
        Pdot = np.zeros((self.m.Nx*(1+self.m.Nx + self.m.Nu + 1),))

        Pdot[self.x_slice] = self.f_c
        Pdot[self.A_d_slice] = (self.A_c @ P[self.A_d_slice].reshape((self.Nx, self.Nx))).flatten()
        Pdot[self.B_d_slice] = (P_inv @ self.B_c).flatten()

       
        r = self.f_c - self.A_c @ x - self.B_c @ u
        Pdot[self.r_d_slice] = (P_inv @ r)
    
        return Pdot
    
    def P_dot_rk4(self,P, u):
        
        f1 = self.P_dot(P, u)
        
        f2 = self.P_dot(P + 0.5 * self.h * f1, u)
        f3 = self.P_dot(P + 0.5 * self.h * f2, u)
        f4 = self.P_dot(P + self.h * f3, u)
        xn = P + (self.h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        
        return xn
    

 