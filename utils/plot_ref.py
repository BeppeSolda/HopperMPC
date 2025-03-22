import numpy as np
import matplotlib.pyplot as plt
import json

"""
This file is used to visualize the reference trajectory and control inputs for the system.

It loads trajectory and control data from a JSON file, then generates various plots:

- A 3D plot of the reference position trajectory (x, y, z)
- Subplots of position, velocity, and quaternion components over time
- Subplots of control inputs over time

The visualizations help in analyzing the reference trajectory before tracking it with the controller.
"""


flag = 'SCVx'

with open("params.json", "r") as f:
    params = json.load(f)

X_ref = np.array(params["X_ref"])
U_ref = np.array(params["U_ref"])
t_ref = np.array(params["t_ref"])
t = t_ref



fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(X_ref[0, :], X_ref[1, :], X_ref[2, :], "r--", label="Reference")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
ax.set_title("3D Position Trajectory")
ax.grid(True)


fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t, X_ref[0, :], "r--")
axs[0, 0].set_ylabel("x")
axs[1, 0].plot(t, X_ref[1, :], "g--")
axs[1, 0].set_ylabel("y")
axs[2, 0].plot(t, X_ref[2, :], "b--")
axs[2, 0].set_ylabel("z")

axs[0, 1].plot(t, X_ref[7, :], "r--")
axs[0, 1].set_ylabel("v_x")
axs[1, 1].plot(t, X_ref[8, :], "g--")
axs[1, 1].set_ylabel("v_y")
axs[2, 1].plot(t, X_ref[9, :], "b--")
axs[2, 1].set_ylabel("v_z")

for ax in axs.flat:
    ax.grid(True)
# Plot control inputs


fig, axs = plt.subplots(3, 1)
axs[0].plot(t, U_ref[0, :], "r--")
axs[0].set_ylabel("u_1")
axs[1].plot(t, U_ref[1, :], "g--")
axs[1].set_ylabel("u_2")
axs[2].plot(t, U_ref[2, :], "b--")
axs[2].set_ylabel("u_3")


for ax in axs:
    ax.grid(True)
# Plot quaternion components


fig, axs = plt.subplots(4, 1)
axs[0].plot(t, X_ref[3, :], "r--")
axs[0].ticklabel_format(style="plain", axis="y")
# axs[0].set_ylim([0.8, 1.2])  # Adjust y-axis limits for better visualization


axs[0].set_ylabel("q0")
axs[1].plot(t, X_ref[4, :], "g--")
axs[1].set_ylabel("q1")
axs[2].plot(t, X_ref[5, :], "b--")
axs[2].set_ylabel("q2")
axs[3].plot(t, X_ref[6, :], "k--")
axs[3].set_ylabel("q3")

for ax in axs:
    ax.grid(True)
# Plot additional variables


fig, axs = plt.subplots(3, 1)
axs[0].plot(t, X_ref[10, :])
axs[0].set_ylabel("p")
axs[1].plot(t, X_ref[11, :])
axs[1].set_ylabel("q")
axs[2].plot(t, X_ref[12, :])
axs[2].set_ylabel("r")
for ax in axs:
    ax.grid(True)

if flag == 'SCVx':
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, U_ref[0, :])
    axs[0].set_ylabel("Fx")
    axs[1].plot(t, U_ref[1, :])
    axs[1].set_ylabel("Fy")
    axs[2].plot(t, U_ref[2, :])
    axs[2].set_ylabel("Fz")
else:
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, X_ref[13, :])
    axs[0].set_ylabel("Fx")
    axs[1].plot(t, X_ref[14, :])
    axs[1].set_ylabel("Fy")
    axs[2].plot(t, X_ref[15, :])
    axs[2].set_ylabel("Fz")

for ax in axs:
    ax.grid(True)
plt.show()
