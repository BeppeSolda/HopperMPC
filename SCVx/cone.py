import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the glide slope angle in degrees
gamma_gs_deg = 5 
gamma_gs = np.radians(gamma_gs_deg) 

# Define the range of Z (height of the cone)
Z2 = np.linspace(0, 5, 100)  # height from 0 to 5

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

# Set labels for the axes
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.set_xlim([-5, 5])  # Set the x-axis range to [-5, 5]
ax.set_ylim([-5, 5])  # Set the y-axis range to [-5, 5]
ax.set_zlim([0, 5])  # Set the z-axis range to [0, max_height]
plt.tight_layout()
# Set aspect to 'auto' for proper scaling
ax.set_aspect('auto')

# Plot the surface of the cone with some transparency
ax.plot_surface(X2, Y2, Z2_surface, alpha=0.4, color="green")

# Show the plot
plt.show()
