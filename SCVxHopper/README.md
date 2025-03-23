# SCVxHopper


This folder contains an implementation of Successive convexification for 6-dof rocket powered landing with fixed final time based on: Successive convexification for 6-DoF Mars rocket powered landing with free-final-time. arXiv. https://doi.org/10.48550/arXiv.1802.03827.
 It is part of the larger project for autonomous hopper landing, and is intended to perform online generation of feasible trajectories to be tracked by the mpc tracker.

### Current Status

- **Development Stage**: In Progress
- **Known Issues**: Code is still very messy but functional, documentation not present.
- **Next Steps**: Make it real time operational by generating code with cvxpygen, furthermore, an implementation which uses thrust derivatives as inputs and thrusts as states is necessary.

## How to Run 


Run the main.py, trajectory parameters can be changed in the constructor of Rocket_Model.py

### Dependencies

This folder depends on the following packages/modules:
- numpy
-scipy
-cvxpy

---

## Acknowledgement

The work in this folder is based on a public repository that has been extremely helpful for its development. The public repository is:

EmbersArc (2018). SCVx. Available at: https://github.com/EmbersArc/SCvx

This repository provided foundational code or insights which were modified or extended for the current use case. Special thanks to the original authors for their work.

---