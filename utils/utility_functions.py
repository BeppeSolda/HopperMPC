from scipy.interpolate import interp1d
import numpy as np


def interpolate_trajectory(U_ref, X_ref, t, num_points):
    """
    Interpolates the trajectory to change the number of points.

    Parameters:
    U_ref (numpy.ndarray): Original control inputs.
    X_ref (numpy.ndarray): Original state references.
    t (numpy.ndarray): Original time vector.
    num_points (int): Number of points for the new interpolated trajectory.

    Returns:
    tuple: Interpolated U_ref, X_ref, t
    """
    t = np.array(t)

    t_new = np.linspace(t[0], t[-1], num_points)

    U_interp = interp1d(t[0:-1], U_ref.T, axis=0, kind="linear")
    U_ref_new = U_interp(t_new[0:-1])
    U_ref_new = U_ref_new.T
    

    X_interp = interp1d(t, X_ref.T, axis=0, kind="linear")
    X_ref_new = X_interp(t_new)
    X_ref_new = X_ref_new.T

    return U_ref_new, X_ref_new, t_new


def extend_reference_trajectory(X_ref, U_ref, N_mpc, T_mpc, t):
    """
    Extend the reference trajectories X_ref and U_ref by N_mpc columns.
    Linear interpolation between second to last and last columns of state and control
    trajectories, this (contrary to extend_reference_trajectory_const)
    ensures that the extension of the trajectories is smooth rather than abruptly repeating
    the last value.
    Args:
        X_ref (numpy.ndarray): Original reference trajectory for states.
        U_ref (numpy.ndarray): Original reference trajectory for controls.
        N_mpc (int): Number of columns to extend.

    Returns:
        X_ref_extended (numpy.ndarray): Extended reference trajectory for states.
        U_ref_extended (numpy.ndarray): Extended reference trajectory for controls.
    """
    # Convert to arrays if they are lists

    X_ref = np.array(X_ref)
    U_ref = np.array(U_ref)
    t = np.array(t)
    # Number of states and controls

    nx = X_ref.shape[0]
    nu = U_ref.shape[0]

    # Create the extended columns

    second_last_X_ref = X_ref[:, -2:-1]
    last_X_ref = X_ref[:, -1:]  # Get the last column of X_ref
    second_last_U_ref = U_ref[:, -2:-1]
    last_U_ref = U_ref[:, -1:]  # Get the last column of U_ref

    interpolation_factors = np.linspace(0, 1, N_mpc + 1)

    # Repeat the last value to extend

    extended_X_ref = np.hstack(
        [
            X_ref,
            second_last_X_ref
            + (last_X_ref - second_last_X_ref) * interpolation_factors[1:],
        ]
    )
    extended_U_ref = np.hstack(
        [
            U_ref,
            second_last_U_ref
            + (last_U_ref - second_last_U_ref) * interpolation_factors[1:],
        ]
    )
    b = np.array(extended_X_ref)
    dt = T_mpc / N_mpc
    t_ext = t[-1] + np.arange(1, N_mpc + 1) * dt
    extended_t = np.hstack([t, t_ext])
    
    return np.array(extended_X_ref), np.array(extended_U_ref), np.array(extended_t)


def extend_reference_trajectory_const(X_ref, U_ref, N_mpc):
    """
    Extend the reference trajectories X_ref and U_ref by N_mpc columns.
    Each extended column will contain a copy of the final value of the respective vectors.

    Args:
        X_ref (numpy.ndarray): Original reference trajectory for states.
        U_ref (numpy.ndarray): Original reference trajectory for controls.
        N_mpc (int): Number of columns to extend.

    Returns:
        X_ref_extended (numpy.ndarray): Extended reference trajectory for states.
        U_ref_extended (numpy.ndarray): Extended reference trajectory for controls.
    """
    # Convert to arrays if they are lists

    X_ref = np.array(X_ref)
    U_ref = np.array(U_ref)

    # Number of states and controls

    nx = X_ref.shape[0]
    nu = U_ref.shape[0]

    # Create the extended columns

    last_X_ref = X_ref[:, -1:]  # Get the last column of X_ref
    last_U_ref = U_ref[:, -1:]  # Get the last column of U_ref

    # Repeat the last value to extend

    extended_X_ref = np.hstack([X_ref, np.tile(last_X_ref, (1, N_mpc))])
    extended_U_ref = np.hstack([U_ref, np.tile(last_U_ref, (1, N_mpc))])

    return np.array(extended_X_ref), np.array(extended_U_ref)
