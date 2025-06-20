# Flow Matching

## 1. what is generative modeling?

Transform a sample from a simple (noise) known distribution (e.g. Gaussian) into a sample from a complex (data) unknow distribution (e.g. CIFAR-10).

## 2 what we have?

- An image dataset (empirical distribution of data).
- A neural network (can be seen as a function approximator).

## 3. how flow matching works?

Flow Matching learns a neural network that tells how a sample should move over time from noise to data, by simulating ODEs.

### 3.1. ODE, flow, and vector field

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

### 3.2. image generation process is a flow

A noise data $x_0 \in R^d$, and an image data $x_1 \in R^d$ are two points in the same space. **The transformation of a noise sample $x_0$ to a data sample $x_1$ can be seen as a flow!**

If we **use a neural network to approximate the vector field $u_t(\cdot)$**, for any strating point $x_0$, we can use numerical method such as **Euler Method** to simulate an ODE:

$$
\psi_{t+h}(x_0) = \psi_{t}(x_0)  +h u_t(\psi_t(x_0)), \quad (t=0,h,2h,\dots,1-h)
$$

and $\psi_{0}(x_0)$ we already know is a noise, thus, we can use the Euler Method to get $\psi_{1}(x_0)$ iteritively, that is the transformation of nosie into data.

### 3.3. probability path

Don't forget our goal is transform a sample from a simple (noise) distribution (e.g. Gaussian) into a sample from a complex (data) distribution (e.g. CIFAR-10).

Now, let us consider a more complex scenario of flow: the starting points are sampled from a fixed distribution (e.g., $X_0 \sim \mathcal{N}(0,1)$), we konw every points will move (under the guidance of vetor field $u_t(\cdot)$) in the space, so the probability density of each point in the space will also change, thus, the data point distribution at each time $t$ is $p_t$ (i.e., $X_t \sim p_t$), we call it **probability path**.

The probability density flow satisfies the following key property (**Continuity Equation**):

$$
\frac{\partial p_t(x)}{\partial t} =- \nabla \cdot (p_t(x)\cdot u_t(x))
$$

To provide an intuitive understanding of this equation, note that the left-hand side represents the instantaneous rate of change of the probability density at point $x$ over time. On the right-hand side, $p_t(x)$ denotes the probability density at position $x$, while $u_t(x)$ represents the velocity field--i.e., the speed and direction at which mass (or probability) moves at $x$. The product $p_t(x)\cdot u_t(x)$ describes the probability flux: it tells how much probability mass is flowing through space per unit time. Taking the divergence $\nabla \cdot (p_t(x) \cdot u_t(x))$ measures the net outflow of this flux from point $x$, and the negative sign indicates that when more mass flows out of $x$ than into it, the local density decreases accordingly. This equation is very important for the following introductions.

### 3.4. approximation of vector field

Note again, our goal is transform a sample from Gaussian distribution ($p_0$) into a sample from data distribution ($p_1$), so we need ensure that the probability path $p_t$ satisfing: 1) $p_0=p_{Gaussian}$, and 2) $p_1=p_{data}$.

If we can design a probability path $p_t$ that satisfy the above two constraints (step 1), then we use a neural network to approximate the vector field $u_t(\cdot)$ (step 2), we can generate real data point from noise!

For step 1, unfortunately, we don't kown the analytic form of $p_{data}$, therefore we can't create a parobability path $p_t$ directly (if we know, we can sample real-data directly).

For step 2, unfortunately, there is no analytic ground truth $u^{target}_t(\cdot)$ as supervision signal to train the neural network (if we know, why train a neural network to approximate it? just use it to generate data).

Fortunately, we have an image dataset, we can use it to learn $u^{target}_t(\cdot)$ implicitly.

#### 3.4.1 conditional probability path

Although we can't design a probability path $p_t$, but we have an image dataset:

$$
D = \{z_1, z_2, \dots,z_n\}, \quad z_i \sim p_{data}
$$

for a give data $z$, we can define the conditional probability $p(\cdot|z)$, if we 

#### 3.4.2 conditional vector field
