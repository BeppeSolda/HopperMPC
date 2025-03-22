import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import art3d
from params import radius, height
import os
import matplotlib.ticker as ticker
import json


with open("params.json", "r") as f:
    params = json.load(f)
height = params["height"]
radius = params["radius"]
Nx_sim = params["Nx_sim"]


plt.style.use("ggplot")  


def threshold_data(data, threshold=1e-7):
    return np.where(np.abs(data) < threshold, 0, data)



def plot_results_tracker(t, x, X_ref_new, u, U_ref_new, save_path=None):
    """
    This function generates and saves a series of plots to visualize the 
    results of the trajectory tracking, including 3D trajectory, position 
    and velocity tracking, angular rates, and quaternion tracking.

    Parameters:
        t (np.ndarray): The new time vector.
        x (np.ndarray): The actual state trajectory.
        X_ref_new (np.ndarray): The reference state trajectory.
        u (np.ndarray): The actual control inputs.
        U_ref_new (np.ndarray): The reference control inputs.
        save_path (str, optional): Path where the plots will be saved. If None, the plots will not be saved.
    """
    print(x.shape)
    print(u.shape)
    print(U_ref_new.shape)
    t_control = t[:-1]
    if Nx_sim ==15:
        
        
        x = x[:,0:-2]
        t = t[0:-2]
        t_control = t_control[0:-1]
        X_ref_new = X_ref_new[:,0:-2]
        U_ref_new = U_ref_new[:,0:-2]
        u = u[:,0:-1]
    else:
        if np.linalg.norm(x[13:16, -1]) == 0:
            x = x[:,0:-2]
            t = t[0:-2]
            X_ref_new = X_ref_new[:,0:-2]
            U_ref_new = U_ref_new[:,0:-2]
            u = u[:,0:-2]
            print(x.shape)
            print(u.shape)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        X_ref_new[0, :],
        X_ref_new[1, :],
        X_ref_new[2, :],
        "g--",
        linewidth=1.5,
        label=r"Reference",
    )
    ax.plot(x[0, :], x[1, :], x[2, :], "k-", linewidth=1.5, label=r"Actual")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.legend(loc="right")
    ax.set_title(r"3D Trajectory")
    ax.grid(True)

    attitude_scale = 3
    thrust_scale = 0.001

    for k in range(
        0, len(t) - 1, 50
    ):  # Adjust step to control frequency of the vectors
        # Position

        rx, ry, rz = x[0, k], x[1, k], x[2, k]

        # Quaternion for orientation

        q0, q1, q2, q3 = x[3, k], x[4, k], x[5, k], x[6, k]

        # Rotation matrix from quaternion

        CBI = np.array(
            [
                [
                    q0**2 + q1**2 - q2**2 - q3**2,
                    2 * (q1 * q2 + q0 * q3),
                    2 * (q1 * q3 - q0 * q2),
                ],
                [
                    2 * (q1 * q2 - q0 * q3),
                    q0**2 - q1**2 + q2**2 - q3**2,
                    2 * (q2 * q3 + q0 * q1),
                ],
                [
                    2 * (q1 * q3 + q0 * q2),
                    2 * (q2 * q3 - q0 * q1),
                    q0**2 - q1**2 - q2**2 + q3**2,
                ],
            ]
        )

        # Attitude vector (rocket body orientation in the z-direction of the body frame)

        dx, dy, dz = np.dot(CBI, np.array([0, 0, 1]))
        ax.quiver(
            rx,
            ry,
            rz,
            dx,
            dy,
            dz,
            length=attitude_scale,
            arrow_length_ratio=0.1,
            color="blue",
            label=r"$z_b$ axis" if k == 0 else "",
        )

        # Thrust vector
        if Nx_sim == 15:
            Fx, Fy, Fz = u[0, k], u[1, k], u[2, k]
        else:
            Fx, Fy, Fz = x[13, k], x[14, k], x[15, k]
        thrust_direction = np.dot(
            CBI, np.array([-Fx, -Fy, -Fz])
        )  # Adjust sign if needed
        ax.quiver(
            rx,
            ry,
            rz,
            thrust_direction[0],
            thrust_direction[1],
            thrust_direction[2],
            length=thrust_scale,
            arrow_length_ratio=0.1,
            color="red",
            label=r"Thrust Vector" if k == 0 else "",
        )
    # Adding a landing pad (optional)

    pad_radius = 1
    pad = patches.Circle((0, 0), pad_radius, color="grey", alpha=0.5)
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad, z=0, zdir="z")
    ax.set_aspect("auto")
    if save_path:
        plt.savefig(os.path.join(save_path, "3d_trajectory_o.png"))
    
    # Position and Velocity Tracking Subplots

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    labels = ["x ", "y ", "z "]
    for i in range(3):
        axs[i, 0].plot(t, x[i, :], "b-", label=r"Actual")
        axs[i, 0].plot(t, X_ref_new[i, :], "r--", label=r"Reference")
        axs[i, 0].set_ylabel(f"${labels[i]}$ [m]")
        axs[i, 0].grid(True)
        axs[i, 1].plot(t, x[7 + i, :], "b-", label=r"Actual")
        axs[i, 1].plot(t, X_ref_new[7 + i, :], "r--", label=r"Reference")
        axs[i, 1].set_ylabel(f"$v_{{{labels[i]}}}$ [m/s]")
        axs[i, 1].grid(True)
    axs[0, 0].legend(loc="upper right")
    fig.suptitle("Position and Velocity Tracking")
    fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "position_velocity_tracking.png"))
    
    # Angular Rates Plot

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    rate_labels = ["p ", "q ", "r "]
    max_angular_rates = params["max_angular_rates"]
    max_angular_rates = np.rad2deg(
        [max_angular_rates] * len(t)
    )  # Convert to degrees per second
    x[12, :] = threshold_data((x[12, :]), threshold=1e-5)
    X_ref_new[12, :] = threshold_data((X_ref_new[12, :]), threshold=1e-5)
    for i in range(3):
        axs[i].plot(
            t, np.rad2deg(x[10 + i, :]), "b-", label=rf"${rate_labels[i]}$ (Actual)"
        )
        axs[i].plot(
            t,
            np.rad2deg(X_ref_new[10 + i, :]),
            "r--",
            label=rf"${rate_labels[i]}$ (Reference)",
        )

        axs[i].set_ylabel(rf"${rate_labels[i]}$ [deg]")

        axs[i].legend(loc="right")
        axs[i].grid(True)
    fig.suptitle("Angular Rates Tracking")
    fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "angular_rates_tracking.png"))
    
    # Quaternion Plots

    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    quaternion_labels = ["q_0", "q_1", "q_2", "q_3"]
    x[6, :] = threshold_data((x[6, :]), threshold=1e-6)
    X_ref_new[6, :] = threshold_data((X_ref_new[6, :]), threshold=1e-6)
    for i in range(4):
        axs[i].plot(
            t, x[3 + i, :], "b-", label=rf"${quaternion_labels[i]}$ (Actual)"
        )
        axs[i].plot(
            t,
            X_ref_new[3 + i, :],
            "r--",
            label=rf"${quaternion_labels[i]}$ (Reference)",
        )
        axs[i].set_ylabel(rf"${quaternion_labels[i]}$")
        axs[i].legend(loc="right")
        axs[i].grid(True)

        axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4f}"))
    quaternion_norm = np.sqrt(
        np.sum(x[3:7, :] ** 2, axis=0)
    )  # Calculate quaternion norm
    quaternion_norm = threshold_data((quaternion_norm), threshold=1e-5)
    axs[4].plot(t, quaternion_norm, "g-", label=r"$\|\mathbf{q}\|$ (Actual)")
    axs[4].axhline(
        1.0, color="r", linestyle="--", label=r"$\|\mathbf{q}\| = 1$ (Ideal)"
    )
    axs[4].set_ylabel(r"$\|\mathbf{q}\|$")
    axs[4].set_ylim(1 - 1e-5, 1 + 1e-5)
    axs[4].set_xlabel(r"Time [s]")
    axs[4].legend(loc="right")
    axs[4].grid(True)
    axs[4].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4f}"))

    fig.suptitle("Quaternion Tracking and Norm")
    fig.tight_layout()
    fig.suptitle("Quaternion Tracking")
    fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "quaternions_tracking.png"))
    
    if Nx_sim == 15:
        print(x.shape)
        print(u.shape)
        scaling_factor_vect = x[14, :] / np.linalg.norm(u, axis=0)
        F_scaled = u[:, :] * scaling_factor_vect
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        thrust_labels = ["F_x", "F_y", "F_z"]
        max_thrust = params["max_thrust"]
        min_thrust = params["min_thrust"]
        cos_delta_max = params["cos_delta_max"]
        limits = [cos_delta_max * u[2, :], max_thrust, min_thrust]
        t_control = t
        for i in range(3):
            axs[i].plot(
                t_control,
                u[ i, :],
                "b-",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Desired)",
            )
            axs[i].plot(
                t_control,
                F_scaled[i, :],
                "g-",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Actual)",
            )
            axs[i].plot(
                t_control,
                U_ref_new[i, :],
                "r--",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Reference)",
            )
            axs[i].set_ylabel(rf"${thrust_labels[i]}$ [N]")

            axs[i].legend(loc="upper right")
            axs[i].grid(True)
        
    else:
        scaling_factor_vect = x[16, :] / np.linalg.norm(x[13:16, :], axis=0)
        F_scaled = x[13:16, :] * scaling_factor_vect
    # Control Inputs with Limits

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        thrust_labels = ["F_x", "F_y", "F_z"]
        max_thrust = params["max_thrust"]
        min_thrust = params["min_thrust"]
        cos_delta_max = params["cos_delta_max"]
        limits = [cos_delta_max * x[15, :], max_thrust, min_thrust]
        t_control = t
        for i in range(3):
            axs[i].plot(
                t_control,
                x[13 + i, :],
                "b-",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Desired)",
            )
            axs[i].plot(
                t_control,
                F_scaled[i, :],
                "g-",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Actual)",
            )
            axs[i].plot(
                t_control,
                X_ref_new[13 + i, :],
                "r--",
                linewidth=1.5,
                label=rf"${thrust_labels[i]}$ (Reference)",
            )
            axs[i].set_ylabel(rf"${thrust_labels[i]}$ [N]")

            axs[i].legend(loc="upper right")
            axs[i].grid(True)
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Thrusts")
    fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "Thrusts.png"))
    
    if Nx_sim==16 or Nx_sim==17:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        control_labels = [r"\dot{F}_x", r"\dot{F}_y", r"\dot{F}_z"]
        limits = [cos_delta_max * u[2, :], max_thrust, min_thrust]
        t_control = t[:-1]
        for i in range(3):
            axs[i].plot(
                t_control,
                u[i, :],
                "b-",
                linewidth=1.5,
                label=rf"${control_labels[i]}$ (Actual)",
            )

            axs[i].plot(
                t_control,
                U_ref_new[i, :-1],
                "r--",
                linewidth=1.5,
                label=rf"${control_labels[i]}$ (Reference)",
            )
            axs[i].set_ylabel(rf"${control_labels[i]}$ [N/s]")
            axs[i].legend(loc="upper right")
            axs[i].grid(True)
            axs[-1].set_xlabel(r"Time [s]")
        fig.suptitle("Control Inputs with Constraints")
        fig.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "Tdot.png"))
    
    # Adjust layout and show plot

    fig.tight_layout()
    # axs[1].plot(t, X_ref_new[16, :], 'b--','Reference Mass')

    if x.shape[0] > 17:
        m = x[17, :]
        J1 = 1 / 12 * m * (height**2 + 3 * radius**2)
        J2 = J1
        J3 = 0.5 * m * radius**2

        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        # Plot mass

        axs[0].plot(t, m, "b-", label=r"Actual Mass")
        axs[0].set_ylabel(r"Mass [kg]")
        axs[0].grid(True)
        axs[0].legend(loc="right")
        axs[0].set_title(r"Mass Variation")

        # Plot inertia J1

        axs[1].plot(t, J1, "b-", label=r"$I_x$")
        axs[1].set_ylabel(r"$I_x$ [kgm^2]")
        axs[1].grid(True)
        axs[1].legend(loc="right")

        # Plot inertia J2

        axs[2].plot(t, J2, "b-", label=r"$I_y$")
        axs[2].set_ylabel(r"$I_y$ [kgm^2]")
        axs[2].grid(True)
        axs[2].legend(loc="right")

        # Plot inertia J3

        axs[3].plot(t, J3, "b-", label=r"$I_z$")
        axs[3].set_ylabel(r"$I_z$ [kgm^2]")
        axs[3].set_xlabel(r"Time [s]")
        axs[3].grid(True)
        axs[3].legend(loc="right")

        # Adjust layout and show plot

        fig.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "Mass_Inertias.png"))
        plt.show()


def plot_solvers_comparison(error_vect, N_mpc_vect):
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(N_mpc_vect, error_vect, "b-", label="mse vs N_mpc")

    # Set axis labels and title

    axs[0, 0].set_ylabel("mse")
    axs[0, 0].set_xlabel("N_mpc")
    axs[0, 0].set_title("MSE vs. N_mpc")
    axs[0, 0].legend()

    # Adjust layout to prevent overlap

    fig.tight_layout()

    # Show the plot

    plt.show(block=False)


def plot_mpc_metrics(
    t, solver_times, tracking_errors, max_tracking_error, save_path=None
):
    """
    Plot solver times, tracking errors, and key performance metrics for the MPC simulation.

    Parameters:
    - t: Time vector for the simulation.
    - solver_times: List of solver times for each MPC iteration (in seconds).
    - tracking_errors: List of tracking errors at each timestep.
    - max_tracking_error: Maximum tracking error during the simulation.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    # Convert solver times to milliseconds

    solver_times_ms = [time * 1000 for time in solver_times]

    # Create a figure for solver times

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(solver_times_ms)), solver_times_ms, "o-", label="Solver Time")
    plt.axhline(
        np.mean(solver_times_ms), color="r", linestyle="--", label="Mean Solver Time"
    )
    plt.axhline(
        np.min(solver_times_ms), color="g", linestyle="--", label="Min Solver Time"
    )
    plt.axhline(
        np.max(solver_times_ms), color="b", linestyle="--", label="Max Solver Time"
    )
    plt.title("Solver Times")
    plt.xlabel("MPC Iteration")
    plt.ylabel("Solver Time [ms]")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    if save_path:
        plt.savefig(os.path.join(save_path, "Solve_times.png"))
    
    plt.show()
    # Create a figure for tracking errors

    plt.figure(figsize=(10, 6))
    plt.plot(
        t[: len(tracking_errors)], tracking_errors, "b-", label="Tracking Error"
    )
    plt.axhline(
        max_tracking_error, color="r", linestyle="--", label="Max Tracking Error"
    )
    plt.title("Tracking Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(os.path.join(save_path, "Tracking_Errors.png"))
    
    plt.show()
    # Summary Metrics

    print(f"Performance Metrics:")
    print(f" - Mean Solver Time: {np.mean(solver_times_ms):.2f} ms")
    print(f" - Min Solver Time: {np.min(solver_times_ms):.2f} ms")
    print(f" - Max Solver Time: {np.max(solver_times_ms):.2f} ms")
    print(f" - Mean Tracking Error: {np.mean(tracking_errors):.2f}")
    print(f" - Min Tracking Error: {np.min(tracking_errors):.2f}")
    print(f" - Max Tracking Error: {max_tracking_error:.2f}")


def plot_angular_rates_and_quaternions(t, x, X_ref_new, save_path=None):
    """
    Plot angular rates and quaternions in a single figure.

    Parameters:
    - t: Time vector for the simulation.
    - x: Actual state vector containing angular rates and quaternions.
    - X_ref_new: Reference state vector.
    - save_path: Optional path to save the plots.
    """
    fig, axs = plt.subplots(8, 1, figsize=(12, 12), sharex=True)

    # Angular Rates

    angular_rate_labels = [r"$p$", r"$q$", r"$r$"]
    for i in range(3):
        axs[i].plot(
            t,
            np.rad2deg(x[10 + i, :]),
            "b-",
            label=f"{angular_rate_labels[i]} (Actual)",
        )
        axs[i].plot(
            t,
            np.rad2deg(X_ref_new[10 + i, :]),
            "r--",
            label=f"{angular_rate_labels[i]} (Reference)",
        )
        axs[i].set_ylabel(f"{angular_rate_labels[i]} [deg/s]")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    # Quaternions

    quaternion_labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
    for i in range(4):
        axs[3 + i].plot(
            t, x[3 + i, :], "b-", label=f"{quaternion_labels[i]} (Actual)"
        )
        axs[3 + i].plot(
            t,
            X_ref_new[3 + i, :],
            "r--",
            label=f"{quaternion_labels[i]} (Reference)",
        )
        axs[3 + i].set_ylabel(quaternion_labels[i])
        axs[3 + i].legend(loc="upper right")
        axs[3 + i].grid(True)
        axs[3 + i].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.4f}")
        )
    # Quaternion Norm

    quaternion_norm = np.sqrt(np.sum(x[3:7, :] ** 2, axis=0))
    quaternion_norm = threshold_data((quaternion_norm), threshold=1e-5)
    axs[7].plot(t, quaternion_norm, "g-", label=r"$\|\mathbf{q}\|$ (Actual)")
    axs[7].axhline(
        1.0, color="r", linestyle="--", label=r"$\|\mathbf{q}\| = 1$ (Ideal)"
    )
    axs[7].set_ylabel(r"$\|\mathbf{q}\|$")
    axs[7].set_ylim(1 - 1e-5, 1 + 1e-5)
    axs[7].legend(loc="upper right")
    axs[7].grid(True)
    axs[7].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4f}"))

    # Common X-axis label

    axs[-1].set_xlabel("Time [s]")

    # Adjust layout

    fig.tight_layout()
    fig.suptitle("Angular Rates and Quaternions Tracking", y=1.02)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "angular_rates_and_quaternions.png"))
    else:
        plt.show()
