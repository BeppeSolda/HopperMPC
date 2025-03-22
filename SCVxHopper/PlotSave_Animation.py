import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import art3d

# # Glideslope half-angle in radians (e.g., 10 degrees)
# gamma_gs_deg = 30  # Glideslope angle in degrees
# gamma_gs_rad = np.radians(gamma_gs_deg)

# # Unit vectors
# e1 = np.array([1, 0, 0])  # x-axis (e1)
# e2 = np.array([0, 1, 0])  # y-axis (e2)
# e3 = np.array([0, 0, 1])  # z-axis (e3)

# # Function to calculate H_gamma matrix
# def H_gamma():
#     return np.vstack([e2, e3])

# # Function to calculate the glideslope cone boundary
# def plot_glideslope_cone(ax, gamma_gs_rad, max_height=10, resolution=50):
#     # # Define a meshgrid for the cone
#     # theta = np.linspace(0, 2 * np.pi, resolution)  # Circular angle around the cone's apex
#     # phi = np.linspace(0, gamma_gs_rad, resolution)  # Angle defining the cone's height
#     # theta, phi = np.meshgrid(theta, phi)

#     # # Parametric equations for the cone surface
#     # radius = max_height * np.tan(phi)  # Radius at each height based on the glideslope angle
#     # X = radius * np.cos(theta)  # x coordinate
#     # Y = radius * np.sin(theta)  # y coordinate
#     # Z = max_height * np.cos(phi)  # z coordinate (height)

#     # # Plot the cone surface
#     # ax.plot_surface(X, Y, Z, color='green', alpha=0.3, rstride=1, cstride=1)

#     # # Plot the vertical walls of the cone (lines from the origin to the top edge)
#     # theta_wall = np.linspace(0, 2 * np.pi, resolution)
#     # wall_radius = max_height * np.tan(gamma_gs_rad)  # Radius at max height
#     # X_wall = wall_radius * np.cos(theta_wall)
#     # Y_wall = wall_radius * np.sin(theta_wall)
#     # Z_wall = np.ones_like(theta_wall) * max_height

#     # # Plot the walls as lines from the origin (0,0,0) to the top edge (X_wall, Y_wall, Z_wall)
#     # for i in range(len(X_wall)):
#     #     ax.plot([0, X_wall[i]], [0, Y_wall[i]], [0, Z_wall[i]], color='green')
#     # Define a meshgrid for the cone
#     theta = np.linspace(0, 2 * np.pi, resolution)  # Circular angle around the cone's apex
#     phi = np.linspace(0, gamma_gs_rad, resolution)  # Angle defining the cone's height
#     theta, phi = np.meshgrid(theta, phi)

#     # Parametric equations for the cone surface
#     radius = max_height * np.tan(phi)  # Radius at each height based on the glideslope angle
#     X = radius * np.cos(theta)  # x coordinate
#     Y = radius * np.sin(theta)  # y coordinate
#     Z = max_height * np.cos(phi)  # z coordinate (height)

#     # Plot the cone surface
#     ax.plot_surface(X, Y, Z, color='green', alpha=0.3, rstride=1, cstride=1)

#     # Plot the bottom circle (the base of the cone)
#     circle_radius = max_height * np.tan(gamma_gs_rad)
#     X_base = circle_radius * np.cos(theta)
#     Y_base = circle_radius * np.sin(theta)
#     Z_base = np.zeros_like(X_base)

#     # Plot the base of the cone
#     ax.plot_surface(X_base, Y_base, Z_base, color='green', alpha=0.3, rstride=1, cstride=1)

# Set scales for visualization
thrust_scale = 0.0008
attitude_scale = 2.5
# This is the function to plot for each iteration

def my_plot(ax, figures_i, X, U):
    ax.clear()  # Clear the previous plot to avoid overlaying
    # Plot the glideslope cone, always originating from (0, 0, 0)
    # plot_glideslope_cone(ax, gamma_gs_rad)
    gamma_gs_deg = 20 # degrees (you can change this value)
    gamma_gs = np.radians(gamma_gs_deg)  # convert to radians

    # Define the range of Z (height of the cone)
    Z2 = np.linspace(0, 15, 100)  # height from 0 to 5

    # Adjust the radius r2 based on the glide slope angle
    r2 = Z2 * np.tan(gamma_gs)  # radius at each height Z

    # Create a meshgrid for the circular cross-sections of the cone at different heights
    theta2 = np.linspace(0, 2 * np.pi, 100)  # angle around the cone
    t2, R2 = np.meshgrid(theta2, r2)

    # Parametrize the surface of the cone
    X2 = R2 * np.cos(t2)  # X coordinates of the surface
    Y2 = R2 * np.sin(t2)  # Y coordinates of the surface

    # Reshape Z2 to be 2D for plotting
    Z2_surface = np.tile(Z2, (t2.shape[0], 1)).T  # Repeat Z values across all circular sections

    # Plot the surface of the cone with some transparency
    ax.plot_surface(X2, Y2, Z2_surface, alpha=0.4, color="green")

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    ax.set_xlabel('X, east')
    ax.set_ylabel('Y, north')
    ax.set_zlabel('Z, up')
    ax.plot(X_i[1, :], X_i[2, :], X_i[3, :], color='black',alpha = 1,label="Trajectory")
    for k in range(0, K, 10):
        rx, ry, rz = X_i[1:4, k]
        qw, qx, qy, qz = X_i[7:11, k]

        # Define rotation matrix
        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        # Thrust and attitude vectors
        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        Fx, Fy, Fz = np.dot(np.transpose(CBI), U_i[:, k])

        # Plot attitude and thrust vectors
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')
     # Ensure the vectors at the last timestep are plotted
    rx, ry, rz = X_i[1:4, K-2]
    qw, qx, qy, qz = X_i[7:11, K-2]
    
    CBI_last = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
        [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
        [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    # Last timestep vectors (attitude and thrust)
    dx_last, dy_last, dz_last = np.dot(np.transpose(CBI_last), np.array([0., 0., 1.]))
    Fx_last, Fy_last, Fz_last = np.dot(np.transpose(CBI_last), U_i[:, K-2])

    # Plot attitude and thrust vectors at the last time step
    ax.quiver(rx, ry, rz, dx_last, dy_last, dz_last, length=attitude_scale, arrow_length_ratio=0.0, color='blue')
    ax.quiver(rx, ry, rz, -Fx_last, -Fy_last, -Fz_last, length=thrust_scale, arrow_length_ratio=0.0, color='red')
 
     # Ensure the vectors at the last timestep are plotted
    rx, ry, rz = X_i[1:4, K-1]
    qw, qx, qy, qz = X_i[7:11, K-1]
    
    CBI_last = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
        [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
        [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    # Last timestep vectors (attitude and thrust)
    dx_last, dy_last, dz_last = np.dot(np.transpose(CBI_last), np.array([0., 0., 1.]))
    Fx_last, Fy_last, Fz_last = np.dot(np.transpose(CBI_last), U_i[:, K-1])

    # Plot attitude and thrust vectors at the last time step
    ax.quiver(rx, ry, rz, dx_last, dy_last, dz_last, length=attitude_scale, arrow_length_ratio=0.0, color='blue')
    ax.quiver(rx, ry, rz, -Fx_last, -Fy_last, -Fz_last, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    # Scaling and layout
    scale = X_i[3, 0]
    ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

    ax.set_title("Iteration " + str(figures_i))
    ax.plot(X_i[1, :], X_i[2, :], X_i[3, :], color='black')
    ax.set_aspect('equal')
    pad_radius = 4
    pad = patches.Circle((0, 0), pad_radius, color='grey', alpha=0.5)
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad, z=0, zdir="z")
     # Zoom in by adjusting the axis limits
    ax.set_xlim([-30, 30])  # Set the x-axis range to [-5, 5]
    ax.set_ylim([-30, 30])  # Set the y-axis range to [-5, 5]
    ax.set_zlim([0, 60])  # Set the z-axis range to [0, max_height]
    plt.tight_layout()
# Function to update plot for each iteration (called by FuncAnimation)
def update_plot(frame, ax, X, U):
    my_plot(ax, frame, X, U)


# Function to save animation
def save_animation(X_in, U_in, filename="rocket_trajectory.gif", fps=1):
    global X, U
    X = X_in
    U = U_in
    interval=2000
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create an animation
    ani = FuncAnimation(fig, update_plot, frames=np.arange(X.shape[0]),
                        fargs=(ax, X, U), repeat=False, interval=interval)

    # Save the animation
    ani.save(filename, writer='imagemagick', fps=fps)
