---
Comparison of MLX and py-metal-compute to solve a PDE with a FDTD method
---

## Description
The code in **BHTE - MLX vs Metal Compute - Simple.ipynb** implements a Finite-Difference Time-Difference (FDTD) of the partial differential equation (PDE) of the BioHeat Thermal Equation (BHTE) using the [MLX](https://github.com/ml-explore/mlx) and a [fork of py-metal-compute](https://github.com/ProteusMRIgHIFU/py-metal-compute). Original project is at (py-metal-compute)[https://github.com/baldand/py-metal-compute]. MLX is a high-level library to run functions aimed for ML/AI using Apple Sillicon GPUs. py-metal-compute is aimed mainly for a close-to-hardware interface to run user-defined GPU kernels.

The BHTE is given by

$$
\rho_t c_t \frac{\partial \theta_m}{\partial t} = \nabla \cdot \left(  k_t \nabla \theta_m \right ) - \omega_b c_b \left(  \theta_m - \theta_a \right ) + Q_m
$$

 where $\theta_m$ is the instantaneous temperature at point $m$. $\rho_t$, $c_t$ and $k_t$ are, respectively, the density, specific heat capacity and thermal conductivity of the tissue. $\omega_b$ and $c_b$ are, respectively, the perfusion rate and heat capacity of blood. $\theta_a$ is the body temperature. The term $Q_m$ is the absorbed energy rate due to the absorption of ultrasound by the tissue.

The equation is solved with forward-step solution  $\theta_m(T+t_0)$. This means the GPU kernel is called thousands of times as the thermal map $\theta_m$ is updated in small temporal $\delta t$ steps.

In this example, the MLX solution is 2.5x slower than the pymetal-compute solution. py-metal-compute library uses a closer-to-the hardware implementation (using Swift functions) where we have a bit more control of command buffer creation. MLX uses the new `mx.fast.metal_kernel` function to use custom kernels, and it uses a "lazy computation" approach. 

The difference in the execution model shows when with MLX with an apparent faster iterations through 12000 calls to the kernel, but then performance takes a hit when results are recovered to Numpy arrays.  We need to interface with Numpy arrays as this code is part of much more complex software for the treatment planning of transcranial ultrasound stimulation ([BabelBrain](https://github.com/ProteusMRIgHIFU/BabelBrain)). py-metal-compute takes more time to complete the 12000 calls to the kernel, but the overall wall-time is less than 1/2 that of Metal.

The test below takes 11.7s in MLX and 4.7s with py-metal-compute, which makes MLX 2.5 slower than py-metal-compute.
