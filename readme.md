# Flow Matching

## 1. what is generative modeling?

Generative modeling is the process of learning how to sample from the data distribution. Generally, it aims to transform samples from a simple, known distribution (e.g., standard Gaussian) into samples from a complex, unknown data distribution (e.g., CIFAR-10).

## 2. what we have in generative modeling?

- An image dataset (empirical distribution of data).
- A neural network (can be seen as a function approximator).

## 3. how flow matching algorithm works?

Flow Matching learns a neural network to predict how a sample should move from noise to data, step by step, by simulating an ordinary differential equation (ODE).

### 3.1. ODE, flow, and vector field
In order to thoroughly understand flow matching, let us start by understanding ordinary differential equations (ODEs).
We can define a **trajectory** by a function $X: [0,1] \to \mathbb{R}^d (t \to X_t)$, which maps from time $t \in [0,1]$ to some location in $\mathbb{R}^d$. The trajectory is also a solution of the following **ODE**:

```math
\frac{d X_t}{dt} = u_t(X_t), \quad s.t. \quad X_0 =x_0,
```

where $X_0 =x_0$ means the starting point is $x_0$, $X_t$ tells us the position of a point at time $t$, given that it started at $x_0$. $u_t(X_t)$ is the velocity of the trajectory $X$ at time $t$. 

For different starting point $x_0$, the trajectory $X_t$ will be different. We may ask that, for any starting point $x_0$, where is the position at time $t$?
This requirs that $X_t$ is a function of time $t$ and the initial point $x_0$, we can rewrite it as $\psi_t(x_0)$, $\psi: [0,1]\times \mathbb{R}^d \to \mathbb{R}^d$, this function is also called **flow**, which is the solution of the following **flow ODE**:

```math
\frac{d \psi_t(x_0)}{dt} = u_t(\psi_t(x_0)),\quad \psi_0(x_0) = x_0.
```

$u_t(\cdot)$ defines a **vector feild** in $\mathbb{R}^d$ at time $t$ (at time $t$, for any point $\psi_t(x_0)\in \mathbb{R}^d$, $u_t(\psi_t(x_0))$ gives a velocity vector to tell the move direction).  
In other word, **vector fields defines ODEs whose solutions are flows!**

### 3.2. image generation process is a flow

**The process of transforming a noise sample $x_0$ into a data sample $x_1$ can be naturally interpreted as a flow -- a continuous path governed by a vector field.**

If we **approximate the vector field $u_t(\cdot)$ with a neural network**, for any strating point $x_0$, as $u_t(\cdot)$ tells us which direction we should move in, we can use numerical method such as **Euler Method** to simulate the ODE:

```math
\psi_{t+h}(x_0) = \psi_{t}(x_0)  +h \cdot u_t(\psi_t(x_0)), \quad (t=0,h,2h,\dots,1-h)
```

$\psi_{0}(x_0)$ we already know is a noise, we can use the Euler Method to get $\psi_{1}(x_0)$ iteritively, that is the transformation of nosie into data.

In other words, if we know the direction in which each pixel should move at every moment during the continuous transformation from noise to data, we can transform any new noise sample into data. This is because we have learned the underlying 'rule' of the transformation — the velocity vector field.

### 3.3. probability path and continuity

Don't forget our goal is transform samples from a known distribution (e.g., standard Gaussian) into samples from an unknown data distribution (e.g., CIFAR-10).

Now, let us consider a more complex scenario of flow: the starting points are sampled from a fixed distribution, e.g., $x_0 \sim \mathcal{N}(0,1)$, we konw every points will move in the space under the guidance of vetor field $u_t(\cdot)$, so the probability density of each point in the space will also change, thus, the data point distribution at each time $t$ is $p_t$ (i.e., $\psi_t(x_0) \sim p_t$), we call it **probability path**.

The probability density and the vector field satisfie the following key property (**Continuity Equation**):

```math
\frac{\partial p_t(x)}{\partial t} =- \nabla \cdot (p_t(x)\cdot u_t(x))
```

To provide an intuitive understanding of this equation, note that the left-hand side represents the instantaneous rate of change of the probability density at point $x$ over time. On the right-hand side, $p_t(x)$ denotes the probability density at position $x$, while $u_t(x)$ represents the velocity field--i.e., the speed and direction at which mass (or probability) moves at $x$. The product $p_t(x)\cdot u_t(x)$ describes the probability flux: it tells how much probability mass is flowing through space per unit time. Taking the divergence $\nabla \cdot (p_t(x) \cdot u_t(x))$ measures the net outflow of this flux from point $x$, and the negative sign indicates that when more mass flows out of $x$ than into it, the local density decreases accordingly.

**That is to say, if we know $u_t$ and $p_0$, we can directly derive $p_t$; and if we just know $p_t$, we can solve the continuity equation to obtain a valid $u_t$ (the solution is not unique).**

### 3.4. approximation of vector field

Now let us consider to transform a sample from standard Gaussian distribution $p_0 = \mathcal{N}(0,1)$ into a sample from data distribution $p_1=p_{data}$, i.e., the probability path $p_t$ should satisfy: $p_0=p_{Gaussian}$, and $p_1=p_{data}$.
If we can design a probability path $p_t$ with in a flow that satisfy the above two constraints (step 1), then we use a neural network to approximate the vector field $u_t(\cdot)$ (step 2), we will know how to generate data from noise!

- For step 1, unfortunately, we don't know the analytic form of $p_{data}$, therefore we can't create a parobability path $p_t$ directly (if we know, we can sample data directly).
- For step 2, unfortunately, there is no analytic ground truth $u^{target}_t(\cdot)$ as supervision signal to train the neural network (if we know, why train a neural network to approximate it? just use it to generate data form noise using Euler Method mentioned in Section 3.2).

Fortunately, we have an image dataset, we can use it to learn $u^{target}_t(\cdot)$ implicitly.

#### 3.4.1 conditional probability path

Although we can't design a probability path $p_t$, but we have an image dataset:

```math
\mathcal{D} = \{z_1, z_2, \dots,z_n\}, \quad z_i \sim p_{data}
```

for a give data $z$, we can define the conditional probability path $p_t(\cdot|z)$, and we know:

```math
p_t(\cdot) = \mathcal{E}_{z\sim p_{data}}[p_t(\cdot|z)] = \int p_t(x|z) \cdot p_{data}(z) dz
```

If we set $p_0(\cdot|z) = \mathcal{N}(0, 1)$, then

```math
p_0(\cdot) = \mathcal{E}_{z\sim p_{data}}[p_0(\cdot|z)] = p_0(\cdot|z)=\mathcal{N}(0, 1)
```

is also a standard Gaussian distribution.

Similarly, we can set $p_1(\cdot|z) = \delta_z$ (Dirac delta distribution, sampling from $\delta_z$ always returns $z$), then

```math
p_1(\cdot) = \mathcal{E}_{z\sim p_{data}}[p_1(\cdot|z)] = p_{data}(z)=p_{data}
```

Such a conditional probability path $p_t(\cdot|z)$ is easy to design, for example, we can use a linear interpolation between $\mathcal{N}(0,1)$ and $\delta(z)$:

```math
p_t(x|z) = (1-t) \cdot \mathcal{N}(0, 1) + t \cdot \delta_z(x) = \mathcal{N}(t\cdot z, (1-t)^2 I_d), \quad t\in[0,1].
```

As you can see, if we design such a conditional probability path $p_t(\cdot|z)$, the marginal probability path $p_t(\cdot)$ will satisfy the constraints we need.
And we can sample from $p_t(\cdot)$ by first sampling a data point $z$ from $p_{data}$, and then sample a point $x$ from $p_t(\cdot|z)$.

#### 3.4.2 conditional vector field
