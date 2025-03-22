import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.animation import FuncAnimation
import time

figures_i = 0

# vector scaling
thrust_scale = 0.0002
attitude_scale = 2

delay = 0.5            # Adjust delay between frames (seconds)
def save_animation(X, U, iterations, filename="trajectory_animation.gif", fps=10):
    """
    Create and save an animation of the trajectory with thrust vectors.

    Args:
        X (numpy.ndarray): Array of shape (iterations, Nx, K), trajectory data.
        U (numpy.ndarray): Array of shape (iterations, Nu, K), control data.
        iterations (int): Number of iterations to animate.
        filename (str): Output file name (e.g., 'animation.mp4' or 'animation.gif').
        fps (int): Frames per second for the animation.
    """
    # Setup the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        
        # Call `my_plot` for this specific frame
        my_plot( X[frame, :, :], U[frame, :, :])


    # Create the animation
    ani = FuncAnimation(fig, update, frames=iterations, interval=1000 // fps, repeat=False)

    
    ani.save(filename, writer="pillow", fps=fps)

    print(f"Animation saved to {filename}")
def dynamic_plot(X, U, figures_N):
    global figures_i
    figures_i = 0

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_press_event)

    def my_plot2(fig, figures_i):
        ax = fig.add_subplot(111, projection='3d')

        X_i = X[figures_i, :, :]
        U_i = U[figures_i, :, :]
        K = X_i.shape[1]

        ax.set_xlabel('X, east')
        ax.set_ylabel('Y, north')
        ax.set_zlabel('Z, up')

        for k in range(K):
            rx, ry, rz = X_i[1:4, k]
            qw, qx, qy, qz = X_i[7:11, k]

            # Transformation matrix from quaternions
            CBI = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
                [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
                [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
            ])

            # Plot the trajectory point
            ax.scatter(rx, ry, rz, color='blue', s=10)

            # Plot thrust vectors
            thrust = U_i[:3, k] * thrust_scale  # Scale thrust for visualization
            ax.quiver(rx, ry, rz, thrust[0], thrust[1], thrust[2], color='red', label='Thrust')

        ax.set_title(f'Iteration {figures_i + 1}')

    for figures_i in range(figures_N):
        fig.clear()
        my_plot(fig, figures_i)
        plt.draw()
        plt.pause(0.001)  # Brief pause to allow plot rendering
        time.sleep(delay)  # Add delay for slower plotting

    plt.show()
def key_press_event(event):
    global figures_i
    fig = event.canvas.figure

    if event.key == 'q' or event.key == 'escape':
        plt.close(event.canvas.figure)
        return

    if event.key == 'right':
        figures_i = (figures_i + 1) % figures_N
    elif event.key == 'left':
        figures_i = (figures_i - 1) % figures_N

    fig.clear()
    my_plot(fig, figures_i)
    plt.draw()


def my_plot(fig, figures_i):
    ax = fig.add_subplot(111, projection='3d')

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    ax.set_xlabel('X, east')
    ax.set_ylabel('Y, north')
    ax.set_zlabel('Z, up')
    ax.plot(X_i[1, :], X_i[2, :], X_i[3, :], color='lightgrey', alpha=0.6, label="Trajectory")

    for k in range(0, K, 10):
        rx, ry, rz = X_i[1:4, k]
        qw, qx, qy, qz = X_i[7:11, k]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        Fx, Fy, Fz = np.dot(np.transpose(CBI), U_i[:, k])

        # attitude vector
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    scale = X_i[3, 0]
    ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

    pad = plt.Circle((0, 0), 20, color='lightgray')
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad)

    ax.set_title("Iteration " + str(figures_i))
    ax.plot(X_i[1, :], X_i[2, :], X_i[3, :], color='lightgrey')
    ax.set_aspect('equal')

def plot(X_in, U_in, sigma_in):
    global figures_N
    figures_N = X_in.shape[0]
    figures_i = figures_N - 1

    global X, U
    X = X_in
    U = U_in

    fig = plt.figure(figsize=(10, 12))
    my_plot(fig, figures_i)
    cid = fig.canvas.mpl_connect('key_press_event', key_press_event)
    plt.show(block = False)


if __name__ == "__main__":
    import os

    folder_number = str(int(max(os.listdir('output/trajectory/')))).zfill(3)

    X_in = np.load(f"output/trajectory/{folder_number}/X.npy")
    U_in = np.load(f"output/trajectory/{folder_number}/U.npy")
    sigma_in = np.load(f"output/trajectory/{folder_number}/sigma.npy")

    plot(X_in, U_in, sigma_in)
