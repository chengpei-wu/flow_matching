# Flow Matching based generative modeling

## 1. what is generative modeling?

Transform a sample from a simple (noise) distribution (e.g. Gaussian) into a sample from a complex (data) distribution (e.g. CIFAR-10).

## 2. how flow matching works?

Flow Matching trains a neural network to **predict how each sample should move at every time step** to follow the path from noise to data, effectively simulating ODEs.

### 2.1. ODE and flow

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

**The transformation of a noise sample $x_0$ to a data sample $x_1$ can be seen as a flow!** A noise data $x_0 \in R^d$, and an image data $x_1 \in R^d$ are two points in the same space.

If we can use a neural network to estimate $\psi_t(x_0)$, we can get $x_1$ directly ($x_0$ is the starting noise point, $x_1$ is the end image point).
