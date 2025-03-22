import matplotlib.pyplot as plt
import numpy as np
def plot_solver_metrics(configurations, mse_data, solve_time_data, save_path=None):
    """
    Plot MSE and Solve Time metrics using mean as dots and max/min as capped vertical lines.
    Highlights mean, max, and min values with distinct colors and includes a legend.

    Parameters:
    - configurations: List of configuration labels (e.g., ["N_mpc=10, PH=1", "N_mpc=20, PH=2", ...]).
    - mse_data: List of tuples [(min_mse, mean_mse, max_mse), ...].
    - solve_time_data: List of tuples [(min_solve_time, mean_solve_time, max_solve_time), ...].
    - save_path: Optional path to save the plots. If None, the plots are displayed.
    """
    latex_configurations = [config.replace("N_mpc", r"$N_\mathrm{mpc}$").replace("T_Horizon", r"$T_\mathrm{horizon}$") for config in configurations]
    x = np.arange(len(configurations))  # X positions for the configurations

    # Plot MSE Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    mse_mins, mse_means, mse_maxs = zip(*mse_data)

    ax.errorbar(
        x, mse_means,
        yerr=[np.array(mse_means) - np.array(mse_mins),
              np.array(mse_maxs) - np.array(mse_means)],
        fmt='o', capsize=5, color='black', ecolor='black', elinewidth=1, label='Error Range'
    )
    ax.scatter(x, mse_means, color='orange', label='Mean (MSE)', zorder=5)
    ax.scatter(x, mse_mins, color='blue', label='Min (MSE)', zorder=5)
    ax.scatter(x, mse_maxs, color='red', label='Max (MSE)', zorder=5)

    ax.set_ylabel("Mean Square Error")
    ax.set_title("MSE Metrics for ECOS Solver - Infinity Trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(latex_configurations, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(f"{save_path}/mse_metrics.png", bbox_inches="tight")
    else:
        plt.show()

    # Plot Solve Time Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    solve_mins, solve_means, solve_maxs = zip(*solve_time_data)

    ax.errorbar(
        x, solve_means,
        yerr=[np.array(solve_means) - np.array(solve_mins),
              np.array(solve_maxs) - np.array(solve_means)],
        fmt='o', capsize=5, color='black', ecolor='black', elinewidth=1, label='Solve Time Range'
    )
    ax.scatter(x, solve_means, color='orange', label='Mean (Solve Time)', zorder=5)
    ax.scatter(x, solve_mins, color='blue', label='Min (Solve Time)', zorder=5)
    ax.scatter(x, solve_maxs, color='red', label='Max (Solve Time)', zorder=5)

    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Solve Time Metrics for ECOS Solver - Infinity Trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(latex_configurations, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(f"{save_path}/solve_time_metrics.png", bbox_inches="tight")
    else:
        plt.show()

configurations = ["N_mpc=20, T_Horizon=2", "N_mpc=30, T_Horizon=3","N_mpc=40, T_Horizon=4","N_mpc=50, T_Horizon=5"]
# mse_data = [(0., 6318.24, 21474.00), (0., 6.99, 50.33), (0., 7.41, 55.31), (0., 7.10, 49.81), (0., 6.94, 52.72)]
# solve_time_data = [(3.98, 6.93, 38.00), (8.00, 43.90, 86.72), ( 14.00, 72.47, 139.06), ( 17.50, 97.77, 189.52), ( 20.51, 126.79, 242.75)]
mse_data = [ (0., 6.99, 50.33), (0., 7.41, 55.31), (0., 7.10, 49.81), (0., 6.94, 52.72)]
solve_time_data = [ (8.00, 43.90, 86.72), ( 14.00, 72.47, 139.06), ( 17.50, 97.77, 189.52), ( 20.51, 126.79, 242.75)]

plot_solver_metrics(configurations, mse_data, solve_time_data, save_path='C:/Users/Utente/Desktop/Hopper_plots/Quaternions/Infinity_New/stats')
