# Flow Matching based generative modeling

## 1. what is generative modeling?

Transform a sample from a simple (noise) distribution (e.g. Gaussian) into a sample from a complex (data) distribution (e.g. CIFAR-10).

## 2. how flow matching works?

Flow Matching trains a neural network to **predict how each sample should move at every time step** to follow the path from noise to data, by simulating ODEs.

### 2.1. ODE, flow, and vector field

For a given trajectory $X_t \in \mathbb{R}^d$ with a fixed starting point $x_0$, and its velocity $u_t(X_t)$ at time $t$, we may formalize the trajectory **as a solution** to the following **ordinary differential equation (ODE)**:

$$
\frac{d X_t}{dt} = u_t(X_t), \quad X_0=x_0.
$$

For different $x_0$, the trajectory $X_t$ will be different.

So, $X_t$ is a function of time $t$ and the initial point $x_0$ (note that, $u_t(\cdot)$ is function of time $t$ and $X_t$), we can rewrite it as $\psi_t(x_0)$, which defines the point in the trajectory at any time $t$ starting from any $x_0$. $\psi_t(x_0)$ defines the transformation **flow** of the initial point $x_0$ at time $t\in[0,1]$, which is the solution of the following **flow ODE**:

$$
\frac{d \psi_t(x_0)}{dt} = u_t(\psi_t(x_0)),\quad \psi_0(x_0) = x_0.
$$

$u_t(\cdot)$ defines a **vector feild** in $\mathbb{R}^d$ space at time $t$ (for any point $x\in \mathbb{R}^d$, $u_t(x)$ gives a velocity vector at time $t$ to tell the move direction). **Vector fields defines ODEs whose solutions are flows!**

### 2.2. image generation process is a flow

A noise data $x_0 \in R^d$, and an image data $x_1 \in R^d$ are two points in the same space. **The transformation of a noise sample $x_0$ to a data sample $x_1$ can be seen as a flow!**

If we **use a neural network to approximate the vector field $u_t(\cdot)ut(⋅)$**, for any strating point $x_0$, we can use numerical method such as **Euler Method** to simulate an ODE:

$$
\psi_{t+h}(x_0) = \psi_{t}(x_0)  +h u_t(\psi_t(x_0)), \quad (t=0,h,2h,\dots,1-h)
$$

and $\psi_{0}(x_0)$ we already know is a noise, thus, we can use the Euler Method to get $\psi_{1}(x_0)$ iteritively, that is the transformation of nosie.

### 2.3 distribution density flow

Don't forget our goal is transform a sample from a simple (noise) distribution (e.g. Gaussian) into a sample from a complex (data) distribution (e.g. CIFAR-10).

Now, let us consider a more complex scenario: the starting points are sampled from a fixed distribution (e.g., $X_0 \sim \mathcal{N}(0,1)$), we konw every points will move (under the guidance of vetor field $u_t(\cdot)$) in the space, so the probability density of each point in the space will also change, thus, the data point distribution at each time $t$ is $p_t$ (i.e., $X_t \sim p_t$), we call it **probability density flow**.

The probability density flow satisfies the following key property (**Continuity Equation**):

$$
\frac{\partial p_t(x)}{\partial t} =- \nabla \cdot (p_t(x)\cdot u_t(x))
$$

To provide an intuitive understanding of this equation, note that the left-hand side represents the instantaneous rate of change of the probability density at point $x$ over time. On the right-hand side, $p_t(x)$ denotes the probability density at position $x$, while $u_t(x)$ represents the velocity field--i.e., the speed and direction at which mass (or probability) moves at $x$. The product $p_t(x)\cdot u_t(x)$ describes the probability flux: it tells how much probability mass is flowing through space per unit time. Taking the divergence $\nabla \cdot (p_t(x) \cdot u_t(x))$ measures the net outflow of this flux from point $x$, and the negative sign indicates that when more mass flows out of $x$ than into it, the local density decreases accordingly.