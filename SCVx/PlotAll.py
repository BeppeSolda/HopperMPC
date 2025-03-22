import numpy as np
import matplotlib.pyplot as plt

def plotAll(X_ref,U_ref,t):
    # X_ref = X_ref.T
    # U_ref = U_ref.T
    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    vector = np.concatenate(([1], -np.ones(3)))

    # Create the diagonal matrix
    T = np.diag(vector)

    zeros_row = np.zeros((1, 3))
    I = np.eye(3)

    # Vertically stack the zeros_row and the identity matrix
    H = np.vstack((zeros_row, I))
    def L(q):
        s = q[0]
        v = q[1:4]
        skew_v = skew(v)
        v = v.reshape(3, 1)
        L=np.block([[s, -np.transpose(v)],
                    [v, s* np.eye(3) + skew_v]])
        return L
    # Function to convert quaternion to rotation matrix
    def qtoQ(q):
        return np.transpose(H) @ T @ L(q) @ T @ L(q) @ H



    # Clear variables, close figures, and clear command window
    plt.close('all')

    # Parse trajectory parameters

    # Set up matrices
    zeros_row = np.zeros((1, 3))
    I = np.eye(3)
    H = np.vstack([zeros_row, I])
    vector = [1, -1, -1, -1]
    T = np.diag(vector)

    # Number of time steps
    Nt = len(t)

    # Plot 3D position trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_ref[1, :], X_ref[2, :], X_ref[3, :], 'r--', label='Reference')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('3D Position Trajectory')
    ax.grid(True)

    # # Plot frames at intervals
    # for k in range(0, Nt-1, 100):
    #     Q = qtoQ(X_ref[3:7, k], H, T)
    #     plotframe(Q, X_ref[0:3, k], 7)

    # Plot position and velocities
    fig, axs = plt.subplots(4, 2)
    axs[0, 0].plot(t, X_ref[1, :], 'r--')
    axs[0, 0].set_ylabel('x')
    axs[1, 0].plot(t, X_ref[2, :], 'g--')
    axs[1, 0].set_ylabel('y')
    axs[2, 0].plot(t, X_ref[3, :], 'b--')
    axs[2, 0].set_ylabel('z')

    axs[0, 1].plot(t, X_ref[4, :], 'r--')
    axs[0, 1].set_ylabel('v_x')
    axs[1, 1].plot(t, X_ref[5, :], 'g--')
    axs[1, 1].set_ylabel('v_y')
    axs[2, 1].plot(t, X_ref[6, :], 'b--')
    axs[2, 1].set_ylabel('v_z')

    axs[3,1].plot(t,X_ref[0,:])
    axs[3, 1].set_ylabel('mass')

    for ax in axs.flat:
        ax.grid(True)

    # Plot control inputs
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, U_ref[0, :], 'r--')
    axs[0].set_ylabel('u_1')
    axs[1].plot(t, U_ref[1, :], 'g--')
    axs[1].set_ylabel('u_2')
    axs[2].plot(t, U_ref[2, :], 'b--')
    axs[2].set_ylabel('u_3')
    # axs[3].plot(t, X_ref[13,:], 'b--')
    # axs[3].set_ylabel('Mass')


    for ax in axs:
        ax.grid(True)

    # Plot quaternion components
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, X_ref[7, :], 'r--')
    axs[0].ticklabel_format(style='plain', axis='y')
    #axs[0].set_ylim([0.8, 1.2])  # Adjust y-axis limits for better visualization

    axs[0].set_ylabel('q0')
    axs[1].plot(t, X_ref[8, :], 'g--')
    axs[1].set_ylabel('q1')
    axs[2].plot(t, X_ref[9, :], 'b--')
    axs[2].set_ylabel('q2')
    axs[3].plot(t, X_ref[10, :], 'k--')
    axs[3].set_ylabel('q3')

    for ax in axs:
        ax.grid(True)

    # Plot additional variables
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, X_ref[11, :])
    axs[0].set_ylabel('p')
    axs[1].plot(t, X_ref[12, :])
    axs[1].set_ylabel('q')
    axs[2].plot(t, X_ref[13, :])
    axs[2].set_ylabel('r')

    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(t, X_ref[13, :])
    # axs[0].set_ylabel('Fx')
    # axs[1].plot(t, X_ref[14, :])
    # axs[1].set_ylabel('Fy')
    # axs[2].plot(t, X_ref[15, :])
    # axs[2].set_ylabel('Fz')

    for ax in axs:
        ax.grid(True)

    plt.show()
