---
title: RLcourse note - Lecture 6 Value Function Approximation
author: DS Jung
date: 2021-12-20 17:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec6_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/UoPei5o4fps)
{: .text-center}

## Introduction of Value Function Approximation

Reinforcement learning can be used to solve *large* problems, e.g.

### 6강 소감

### Value Function Approximation

- So far we gave represented value function by a lookup table
  - Every state s has an entry $\mathsf{V(s)}$
  - Or every state-action pair s, a has an entry $\mathsf{Q(s,a)}$
- Probelm with large MDPs:
  - There are too many states and/or actions to store in memory
  - It is too slow to learn the value of each state individually
- Solution for large MDPs:
  - Estimate value function with function approximation
<br><center>$$ \begin{aligned}
\mathsf{\hat{v}(s,w)} &\approx \mathsf{v_\pi(s)} \\
\mathsf{or \;\hat{q}(s,a,w)} &\approx \mathsf{q_\pi(s,a)}
\end{aligned}$$</center>
  - Generalise from seen states to unseen states
  - Update parameter w using MC or TD learning

Function Approximators

- Linear combinations of features
- Neural network
- Decision tree
- Nearest neighbour
- Fourier / wavelet bases
- ...

We require a training method that is suitable for non-stationary, non-iid data

## Incremental Methods

### Gradient Descent

- Let $\mathsf{J(w)}$ be a differentiable function of parameter vector w
- Define the gradient of $\mathsf{J(w)}$ to be
<br><center>$$ \mathsf{\nabla _w J(w)} = \left( \begin{array} \mathsf{ \frac{\partial J(w)}{\partial w_1} } \\ \vdots \\ \mathsf{ \frac{\partial J(w)}{\partial w_n} } \end{array} \right) $$ </center>
- To find a local minimum of $\mathsf{J(w)}$
- Adjust w in direction of -ve gradient
<br><center> $$ \mathsf{ \Delta w = -\frac{1}{2} \alpha \nabla _w J(w) } $$</center>
where $\alpha$ is a step-size parameter

Gradient의 minus값으로 향하는 vector이므로 local minimum을 찾게 됨

#### Value Funcion Approx. By Stochastic Gradient Descent

- Goal: find parameter vector w minimising mean-squared error between approximate value fn $\mathsf{\hat{v}(s,w)}$ and true value fn $\mathsf{v_\pi(s)}$

$$ \mathsf{ J(w) = \mathbb{E}_\pi [(v_\pi(S) - \hat{v} (S,w))^2] } $$

- Gradient descent finds a local minimum

$$ \begin{aligned}
\Delta \mathsf{w} &= \mathsf{-\frac{1}{2}\alpha \nabla _w J(w) } \\
&= \mathsf{ \alpha \mathbb{E}_\pi [(v_\pi(S) - \hat{v}(S,w))\nabla _w \hat{v} (S,w)] }
\end{aligned} $$

- Stochastic gradient descent samples the gradient

$$ \mathsf{ \Delta w = \alpha (v_\pi(S) - \hat{v}(S,w))\nabla _w \hat{v} (S,w) } $$

- Expected update is equal to full gradient update

### Linear Function Approximation

#### Feature Vectors

- Represent state by a feature vector

$$ \mathsf{ x(S)} = \left( \begin{array}\, \mathsf{x_1(S)} \\ \vdots \\ \mathsf{x_n(S)} \end{array} \right) $$

- For example:
  - Distance of robot from landmarks
  - Trends in the stock market
  - Piece and pawn configurations in chess

#### Linear value Function Approximation

- Represent value function by a linear combination of feature

$$ \mathsf{ \hat{v}(S,w) = x(S)^T w=\displaystyle\sum^n_{j=1}x_j(S)w_j } $$

- Objective function is quadratic in parameters $\mathsf{w}$

$$ \mathsf{ J(w) = \mathbb{E}_\pi \left[ (v_\pi(S)-x(S)^T w)^2\right] } $$

- Stochastic gradient descent converges on global optimum
- Update rule is particularly simple

$$ \begin{aligned}
\mathsf{\nabla _w \hat{v}(S,w)} &= \mathsf{x(S)} \\
\mathsf{\Delta w} &= \mathsf{ \alpha (v_\pi(S) - \hat{v}(S,w))x(S) }
\end{aligned} $$

Update - step-size $\times$ prediction error $\times$ feature value

#### Table Lookup Features

- Table lookup is a special case of linear value function approximation
- Using table lookup features

$$ \mathsf{ x^{table}(S)} = \left( \begin{array}\, \mathsf{1(S=s_1)} \\ \vdots \\ \mathsf{1(S=s_n)} \end{array} \right) $$

- Parameter vector $\mathsf{w}$ gives value of each individual state

$$ \mathsf{\hat{v}(S,w)} = \left( \begin{array}\, \mathsf{1(S=s_1)} \\ \vdots \\ \mathsf{1(S=s_n)} \end{array} \right) \cdot \left( \begin{array}\, \mathsf{w_1} \\ \vdots \\ \mathsf{w_n} \end{array} \right) $$

### Incremental Prediction Algorithm

- Have assumed true value function $\mathsf{v_\pi(s)}$ given by supervisor
- But in RL there is no supervisor, only rewards
- In practice, we substitute a target for $\mathsf{v_\pi(s)}$
  - For MC, the target is the return $\mathsf{G_t}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}G_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>
  - For TD(0), the target is the TD target $\mathsf{R_{t+1} + \gamma\hat{v}(S_{t+1},w)}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}R_{t+1} + \gamma\hat{v}(S_{t+1},w)} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>
  - For TD($\lambda$), the target is the $\lambda$-return $\mathsf{G^\lambda_t}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}G^\lambda_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>

#### Monte-Carlo with Value Function Approximation

- Return $\mathsf{G_t}$ is an unbiased, noisy sample of true value $\mathsf{v_\pi(S_t)}$

## Batch Methods
