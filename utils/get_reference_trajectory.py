import numpy as np
import json
from scipy.interpolate import interp1d

"""
This module provides utility functions for working and preprocessing the reference trajectory to track. The main functionality includes:

1. **get_reference_trajectory_thorpy**:
   - Extract and process data from a file (e.g., `traj_parameters.txt`) to generate reference control inputs (`U_ref`), reference states (`X_ref`), and time vector (`t_ref`).

2. **Rotation Matrix to Quaternion Conversion**:
   - Utility to convert rotation matrices to quaternions, to get the reference quaternions from rotation matrices used to express attitude in Thorpy.

3. **Trajectory Interpolation**:
   - Interpolate trajectory data to reduce or increase the number of points, ensuring compatibility with control algorithms requiring specific discretization.

4. **get_reference_trajectory_SCVX**:
   - Loads and processes reference trajectory data from a `npz` format file (format of SCVX generated trajectories), extracting control inputs (`U_ref`), state references (`X_ref`), and time vector (`t_ref`).

5. **get_reference_trajectory**:
   - A general function that loads and preprocesses a reference trajectory based on the specified type (`SCVX` or `Thorpy`), then interpolates 
     the trajectory to match the desired number of points for control input and state references.

"""

def get_reference_trajectory_thorpy(filename):
    """
    Extracts and processes reference trajectory data from a specified file (`traj_parameters.txt`) for tracking.

    Parameters:
    filename (str): Path to the reference trajectory file.

    Returns:
    U_ref (numpy.ndarray): Reference control inputs.
    X_ref (numpy.ndarray): Reference states (including position, quaternion, velocity, etc.).
    t (numpy.ndarray): Time vector.
"""

    with open(filename, "r") as file:
        data = json.load(file)  
    

    e1bx = data["e1bx"]
    e1by = data["e1by"]
    e1bz = data["e1bz"]
    e2bx = data["e2bx"]
    e2by = data["e2by"]
    e2bz = data["e2bz"]
    e3bx = data["e3bx"]
    e3by = data["e3by"]
    e3bz = data["e3bz"]
    x = data["x"]
    y = data["y"]
    z = data["z"]
    t = data["t"]
    omega = data["omega"]
    p = omega[0]
    q = omega[1]
    r = omega[2]
    vx = data["vx"]
    vy = data["vy"]
    vz = data["vz"]

    Fx = data["f1"]
    Fy = data["f2"]
    Fz = data["f3"]

    Fx_dot = data["f1_dot"]

    Fy_dot = data["f2_dot"]
    Fz_dot = data["f3_dot"]

    num_samples = len(e1bx)

    quaternions = np.zeros((4, num_samples))
    vx = np.reshape(vx, (-1, 1))
    vy = np.reshape(vy, (-1, 1))
    vz = np.reshape(vz, (-1, 1))
    vx_b = np.zeros_like(vx)
    vy_b = np.zeros_like(vy)
    vz_b = np.zeros_like(vz)
    
    v_b = np.zeros([3, 1])
    for i in range(num_samples):
        R = np.array(
            [
                [e1bx[i], e2bx[i], e3bx[i]],
                [e1by[i], e2by[i], e3by[i]],
                [e1bz[i], e2bz[i], e3bz[i]],
            ]
        )

        quaternions[:, i] = rotation_matrix_to_quaternion(R)
        v_I = np.array([vx[i], vy[i], vz[i]]) #Velocity from Thorpy is in the Inertial frame

        v_b = R.T @ v_I #Convert velocity to body frame for compatibility with the mpc formulation.

        vx_b[i] = v_b[0]
        vy_b[i] = v_b[1]
        vz_b[i] = v_b[2]
   
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    z = np.reshape(z, (-1, 1))

    p = np.reshape(p, (-1, 1))
    q = np.reshape(q, (-1, 1))
    r = np.reshape(r, (-1, 1))
    
    Fx = np.reshape(Fx, (-1, 1))
    Fy = np.reshape(Fy, (-1, 1))
    Fz = np.reshape(Fz, (-1, 1))

    
    U_ref = np.vstack((Fx_dot, Fy_dot, Fz_dot)).T

    X_ref = np.hstack((x, y, z, quaternions.T, vx_b, vy_b, vz_b, p, q, r, Fx, Fy, Fz))

    return U_ref, X_ref, t


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    R (numpy.ndarray): 3x3 rotation matrix

    Returns:
    numpy.ndarray: Quaternion [q0, q1, q2, q3]
    """
    q0_mag = np.sqrt(abs((1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4))
    q1_mag = np.sqrt(abs((1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4))
    q2_mag = np.sqrt(abs((1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4))
    q3_mag = np.sqrt(abs((1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4))

    if q0_mag > q1_mag and q0_mag > q2_mag and q0_mag > q3_mag:
        q0 = q0_mag
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    elif q1_mag > q0_mag and q1_mag > q2_mag and q1_mag > q3_mag:
        q1 = q1_mag
        q0 = (R[2, 1] - R[1, 2]) / (4 * q1)
        q2 = (R[0, 1] + R[1, 0]) / (4 * q1)
        q3 = (R[0, 2] + R[2, 0]) / (4 * q1)
    elif q2_mag > q0_mag and q2_mag > q1_mag and q2_mag > q3_mag:
        q2 = q2_mag
        q0 = (R[0, 2] - R[2, 0]) / (4 * q2)
        q1 = (R[0, 1] + R[1, 0]) / (4 * q2)
        q3 = (R[1, 2] + R[2, 1]) / (4 * q2)
    else:
        q3 = q3_mag
        q0 = (R[1, 0] - R[0, 1]) / (4 * q3)
        q1 = (R[0, 2] + R[2, 0]) / (4 * q3)
        q2 = (R[1, 2] + R[2, 1]) / (4 * q3)
    return np.array([q0, q1, q2, q3])


def interpolate_trajectory(U_ref, X_ref, t, num_points):
    """
    Interpolates the trajectory to reduce/increase the number of points.

    Parameters:
    U_ref (numpy.ndarray): Original control inputs.
    X_ref (numpy.ndarray): Original state references.
    t (numpy.ndarray): Original time vector.
    num_points (int): Number of points for the new interpolated trajectory.

    Returns:
    Interpolated U_ref, X_ref, t
    """
    t_new = np.linspace(t[0], t[-1], num_points)

    

    U_interp = interp1d(t, U_ref, axis=0, kind="linear")
    U_ref_new = U_interp(t_new)

   

    X_interp = interp1d(t, X_ref, axis=0, kind="linear")
    X_ref_new = X_interp(t_new)

    return U_ref_new, X_ref_new, t_new


def get_reference_trajectory_SCVX(filename):
    """
    Loads and processes reference trajectory data from a `.npz` file (SCVX format).

    Parameters:
    filename (str): Path to the SCVX format file.

    Returns:
    U_ref (numpy.ndarray): Reference control inputs.
    X_ref (numpy.ndarray): Reference states.
    t (numpy.ndarray): Time vector.
"""
    data = np.load(filename)
    all_X = data['X']
    all_U = data['U']
    t = data['time_vect']
    U_ref = np.zeros_like(all_U) 
    X_ref = np.zeros((14, all_X.shape[1])) 
    
    U_ref = all_U

    # Assign the specified slices from all_X to X_ref
    X_ref[3:7, :] = all_X[7:11, :]
    X_ref[10:13, :] = all_X[11:14, :]
    X_ref[0:3, :] = all_X[1:4, :]
    X_ref[7:10, :] = all_X[4:7, :]
   
    X_ref[13,:] = all_X[0,:]
    
    return U_ref.T, X_ref.T, t




def get_reference_trajectory(filename,flag,Interpolation_Points):
    """
    Loads and preprocesses a reference trajectory based on the specified type (`SCVX` or `Thorpy`), then interpolates 
    the trajectory to match the desired number of points for control input and state references.

    Parameters:
    filename (str): Path to the reference trajectory file.
    flag (str): Type of reference trajectory (`'SCVx'` or `'Thorpy'`).
    Interpolation_Points (int): Desired number of points for interpolation.

    Returns:
    U_ref (numpy.ndarray): Interpolated reference control inputs.
    X_ref (numpy.ndarray): Interpolated reference states.
    t_ref (numpy.ndarray): Interpolated time vector.
"""


    if flag == 'SCVx':
        U_ref, X_ref, t_ref =get_reference_trajectory_SCVX(filename)
        print('Loaded SCVx Trajectory')
    elif flag == 'Thorpy':
        U_ref, X_ref, t_ref =get_reference_trajectory_thorpy(filename)
        print('Loaded Thorpy Trajectory')
    else:
        raise ValueError("Please specify a valid reference trajectory.")
    
    U_ref,X_ref,t_ref = interpolate_trajectory(U_ref, X_ref, t_ref,Interpolation_Points)

    return U_ref,X_ref,t_ref



