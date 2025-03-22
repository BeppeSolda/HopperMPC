from utils.get_reference_trajectory import get_reference_trajectory

class Trajectory:
    """
    This class manages the reference trajectory for a given simulation or control task.

    Attributes:
    - filename (str): Path to the reference trajectory file.
    - Trajectory_Flag (str): The type of reference trajectory ('SCVx' or 'Thorpy').
    - Interpolation_Points (int): The number of points for interpolating the trajectory.

    Methods:
    - __init__: Initializes the trajectory by loading and interpolating reference data from the specified file.
    - get_initial_conditions: Returns the initial state and control input from the reference trajectory.

    Description:
    1. **Trajectory Initialization**:
        - Loads the reference trajectory data from a file (either SCVx or Thorpy format) based on the specified `Trajectory_Flag`.
        - Interpolates the reference trajectory to the desired number of points using `Interpolation_Points`.
        - Initializes key attributes such as `X_ref`, `U_ref`, time vector `t_ref`, final time `T_final`, time step `h`, and the number of time steps `Nt`.
    """
    
    def __init__(self,filename,Trajectory_Flag,Interpolation_Points):
        self.filename = filename
        self.Trajectory_Flag = Trajectory_Flag
        self.Interpolation_Points = Interpolation_Points

        U_ref, X_ref, t_ref =  get_reference_trajectory(self.filename,self.Trajectory_Flag,self.Interpolation_Points)

        self.X_ref = X_ref.T
        self.U_ref = U_ref.T
       
        self.t_ref = t_ref
        self.T_final = t_ref[-1]
        self.h = t_ref[1] - t_ref[0]
        self.Nt = int(self.T_final / self.h) + 1

    def get_initial_conditions(self):
        return self.X_ref[:, 0], self.U_ref[:, 0]
