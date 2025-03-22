import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_init(X,U,t):
   


    # Plot 3D position trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0, :], X[1, :], X[2, :], 'r--', label='Reference')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('3D Position Trajectory')
    ax.grid(True)

    # # Plot frames at intervals
    # for k in range(0, Nt-1, 100):
    #     Q = qtoQ(X[3:7, k], H, T)
    #     plotframe(Q, X[0:3, k], 7)

    # Plot position and velocities
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(t, X[0, :], 'r--')
    axs[0, 0].set_ylabel('x')
    axs[1, 0].plot(t, X[1, :], 'g--')
    axs[1, 0].set_ylabel('y')
    axs[2, 0].plot(t, X[2, :], 'b--')
    axs[2, 0].set_ylabel('z')

    axs[0, 2].plot(t, X[7, :], 'r--')
    axs[0, 2].set_ylabel('v_x')
    axs[1, 2].plot(t, X[8, :], 'g--')
    axs[1, 2].set_ylabel('v_y')
    axs[2, 2].plot(t, X[9, :], 'b--')
    axs[2, 2].set_ylabel('v_z')

    for ax in axs.flat:
        ax.grid(True)

    # Plot control inputs
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, U[0, :], 'r--')
    axs[0].set_ylabel('u_1')
    axs[1].plot(t, U[1, :], 'g--')
    axs[1].set_ylabel('u_2')
    axs[2].plot(t, U[2, :], 'b--')
    axs[2].set_ylabel('u_3')
    # axs[3].plot(t, X[13,:], 'b--')
    # axs[3].set_ylabel('Mass')


    for ax in axs:
        ax.grid(True)

    # Plot quaternion components
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, X[3, :], 'r--')
    axs[0].set_ylabel('q0')
    axs[1].plot(t, X[4, :], 'g--')
    axs[1].set_ylabel('q1')
    axs[2].plot(t, X[5, :], 'b--')
    axs[2].set_ylabel('q2')
    axs[3].plot(t, X[6, :], 'k--')
    axs[3].set_ylabel('q3')

    for ax in axs:
        ax.grid(True)

    # Plot additional variables
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, X[10, :])
    axs[0].set_ylabel('p')
    axs[1].plot(t, X[11, :])
    axs[1].set_ylabel('q')
    axs[2].plot(t, X[12, :])
    axs[2].set_ylabel('r')
    axs[3].plot(t, X[13, :])
    axs[3].set_ylabel('mass')

   

    for ax in axs:
        ax.grid(True)

    plt.show()
