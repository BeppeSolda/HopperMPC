# Hopper MPC

## Overview

This project is focused on the design of model predictive control algorithms to track given reference trajectories. The controllers can be tested on a variety of conditions in terms of rocket dynamics modeling and disturbances.
It is possible to test the algorithms in closed loop simulation and to retreive the generated code for real time use.

### Current Status

- **Overall Project Stage**: Complete
- **Modules/Folders**:
- **mpc**: Contains MPC implementations for different rocket dynamics models, including simulation and code generation versions.
  - **dynamics**: Defines various models for the rocket dynamics, providing flexibility for different types of simulations.
  - **utils**: A collection of utility functions used across different parts of the project (e.g., data processing, trajectory handling).
  - **simulation**: Contains functions necessary for setting up and running the closed loop simulation.
  - **Reference_Trajectories**: This folder contains text or `.npz` files with reference trajectory data used for testing and simulation.


- **Next Steps**: 
- **Real-Time Implementation**: Currently, the interpolation for retrieving Jacobian slices and reference trajectories during each MPC iteration is implemented in Python. For real-time use, a C implementation of this interpolation is required to meet real-time performance demands.

---

## Dependencies
-numpy
-scipy
-cvxpy
-cvxpygen
-casadi

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt