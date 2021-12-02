---
title: RLcourse note2 - Lecture 2 Markov Decision Processes
author: DS Jung
date: 2021-11-27 10:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec2_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/lfHX2hHRMVQ)
{: .text-center}

## Introduction ot MDPs

- *Markove decision processes* formally describe and envrionment for reinforcement learning
- Where the envrionment is *fully observable*
- Almost all RL problems can be formalised as MDPs 뭐 잘 설계해서 MDP 맞추란 소린데 누가 이런말 못함

## Markov Property

"The future is independent of the past given the present" 문제를 자꾸 단정하고 싶어한다. 그게 속도나 효율면에서 맞긴 하지만 이게 다른 문제를 만들진 않을까? 복잡하다고 불가능하진 않을것 같은데

|Definition||
|:---|---:|
|A state $$S_t$$ is Markov if and only if | $$ \mathbb{P}[S_{t+1} \| S_t] = \mathbb{P}[S_{t+1} \| S_1, ...,S_t] $$|

- The state captures all relevant information from the history
- Once the state is known, the history may be thrown away
- 흐음... 그보다는 environment의 information에 적절히 history data를 축적한 변수를 만들어야 하지 않을까. 그 변수를 전부 현재의 변수라고 설정하는 것 뿐 현재의 envrionment에서 관찰할 수 있는 것은 아니지 않으려나.

### State Transition Matrix

For a Markov state *s* and successor state *s'*, the *state transition probability* is defined by,

$$ \mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'|S_t =s] $$

State transition matrix $\mathcal{P}$ defines transition probabilities from all states $s$ to all successor states $s'$

$$ \mathcal{P} = from \begin{bmatrix} \mathcal{P}_{11}&\dots&\mathcal{P}_{1n}\\ \vdots & & \\ \mathcal{P}_{n1}&\dots&\mathcal{P}_{nn} \end{bmatrix}$$

- state 의 갯수가 n개로 제한된 경우에만 사용하는 건가?

### Markov Process

A Markov process is a memoryless random process

|Definition|
|---|
|a *Markov* Process (or *Markov Chain*) is a tuple $\langle\mathcal{S,P}\rangle$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a (finite) set of states<br>$\tiny{\blacksquare}\quad\normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s]$$|

### Markov Reward Process (MRP)

A Markov reward process is a Markov chain with values

|Definition|
|---|
|a *Markov Reward* Process is a tuple $\langle\mathcal{S,P,}$<span style="color:red">$\mathcal{R,\gamma}$</span>$\rangle$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a (finite) set of states<br>$\tiny{\blacksquare}\quad\normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s]$$ <br> $\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{R}}\;$ is a reward function, $$\mathcal{R}_s=\mathbb{E}[R_{t+1} \| S_t=s] $$</span><br>$\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{\gamma}}\;$ is a discount factor, $$\gamma\in[0,1]$$</span>|

## Return

|Definition|
|---|
|The return $G_t$ is the total discounted reward from tiem-step $t$.<br><center>$$G_t = R_{t+1}+\gamma R_{t+2}+... = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$</center>|

- The discount $\gamma \in [0,1]$ is the present value of future rewards
- The value of receiving reward $R$ after $k+1$ time-steps is $\gamma^k R$
- This values immediate reward above delayed reward
  - $\gamma$ close to 0 leads to "myopic" evaluation
  - $\gamma$ close to 1 leads to "far-sighted" evaluation
- 단순 급수적인 형태로 감가를 시행하고 있지만 경우에 따라서 감가 함수를 별도로 함수로 만들어 사용할 수 있을 듯. 크게 필요한 경우가 있을 진 모르겠지만..

## Value Function

The value function $v(s)$ gives the long-term value of state $s$

|Definition|
|---|
|The *state* vlaue function $v(s)$ of an MRP is the expected return starting from state $s$<br> <center>$$\mathsf{v}(s)=\mathbb{E}[G_t\;\|\;S_t=s] $$</center>|

## Bellman Equation for MRPs

The value function can be decomposed into two parts:

- immediate reward $R_{t+1}$
- discounted value of successor state $\gamma \mathsf{v}(S_{t+1})$

$$\quad\quad\quad\quad\mathsf{v}(s) =\mathbb{E}\,[\,G_t\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma R_{t+2} + \gamma ^2 R_{t+3} + \dots\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma (R_{t+2} + \gamma R_{t+3} + \dots)\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma G_{t+1}\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma \mathsf{v} (S_{t+1})\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\mathsf{v}(s) =\mathcal{R}_s + \gamma \displaystyle\sum_{s' \in S} ^{}\mathcal{P}_{ss'} \mathsf{v} (s')$$

### Bellman Equation in Matrix Form

The Bellman equation can be expressed concisely using matrices.

$$ \mathsf{v} = \mathcal{R} + \gamma \mathcal{P} \mathsf{v} $$

where $\mathsf{v}$ is a column vector with one entry per state

$$  \begin{bmatrix} \mathsf{v} (1) \\ \vdots \\ \mathsf{v} (n) \end{bmatrix} = \begin{bmatrix} \mathcal{R}_1 \\ \vdots \\ \mathcal{R}_n \end{bmatrix} + \gamma \begin{bmatrix} \mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\ \vdots & & \\ \mathcal{P}_{n1} & \dots & \mathcal{P}_{nn} \end{bmatrix} \begin{bmatrix} \mathsf{v} (1) \\ \vdots \\ \mathsf{v} (n) \end{bmatrix} $$

각 state 1~n에서의 value function은 각 state의 reward와 dot (각 state로의 확률), (각 state의 value) * discount factor 의 합이다.

### Solving the Bellman Equation

- Linear equation

$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \mathsf{v} = \mathcal{R+ \gamma P}\mathsf{v}$$<br>
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\; (1 - \gamma \mathcal{P}) \mathsf{v} = \mathcal{R}$$<br>
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \mathsf{v} = (1 - \gamma \mathcal{P})^{-1} \mathcal{R}$$

- Computational complexiy is $O(n^3)$ for n state
- Direct solution only possible for small MRPs
- There are many iterative methods for large MRPs, e.g.
  - Dynamic programming
  - Monte-Carlo evaluation
  - Temporal-Difference learning

## Markov Decision Process

|Definition|
|---|
|a *Markov Reward* Process is a tuple $\langle\mathcal{S, }$<span style="color:red">$\mathcal{A}$</span>$\mathcal{, P, R, \gamma \rangle}$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a finite set of states<br>$\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{A}}\;$ is a finite set of actions</span><br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}^a_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s,$$ <span style="color:red">$$A_t =a$$</span>$$]$$ <br> $\tiny{\blacksquare}\quad \normalsize{\mathcal{R}}\;$ is a reward function, $$\mathcal{R}^a_s=\mathbb{E}[R_{t+1} \| S_t=s,$$ <span style="color:red">$$A_t =a$$</span>$$] $$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{\gamma}}\;$ is a discount factor, $$\gamma\in[0,1]$$|

## Policies

|Definition|
|---|
|A policy $\pi$ is a distribution over actions given states.<br><centor>$$\pi (a\| s) = \mathbb{P} [A_t = a \| S_t = s]$$|

- A policy fully defines the behaviour of an agent
- MDP policies depend on the current state (not the history)
- Policies are stationary (time-independent)

$\quad A_t ~ \pi(. \| S_t), \forall t > 0$

- Given an MDP $\mathcal{M = \langle S, A, P ,R ,\gamma \rangle}$ and a policy $ \pi$
- The state sequence $\mathcal{S_1, S_2, \dots }$ is a Markov process $\mathcal{\langle S, P^{\pi} \rangle}$
- The state and reward sewuence $\mathcal{S_1, R_2, S_2, \dots}$ is a Markov reward process $\mathcal{\langle S, P^{\pi}, R^{\pi}, \gamma \rangle }$
- where

<center>
$$ \mathcal{P^{\pi}_{s, s'}= \displaystyle\sum_{a \in A} \pi (a \| s) P^a_{ss'}}$$
$$ \mathcal{R^{\pi}_{s}= \displaystyle\sum_{a \in A} \pi (a \| s) R^a_{s}}$$</center>

## Value Function

The state-value function $\mathsf{v_{\pi} (s)} $ of an MDP is the expected return starting from state s, and then following policy $\pi$

$$\mathsf{v_{\pi} (s) = \mathbb{E}_{\pi} [G_t | S_t = s]} $$

The action-value function $\mathsf{q_{\pi} (s,a)}$ is the expected return starting from state s, taking action a, and then following policy $\pi$

$$\mathsf{q_{\pi} (s,a) = \mathbb{e}_{\pi} [G_t | S_t = s, A_t = a] }$$

### Bellman Expectation Equation

The state-value function can again be decomposed into immediate reward plus discounted value of successor state

$$ \mathsf{v_{\pi} (s) = \mathbb{E}_{\pi} [R_{t+1} + \gamma v_{\pi} (S_{t+1}) | S_t = s]}$$

The action-value function can similarly be decomposed

$$ \mathsf{q_{\pi} (s,a) = \mathbb{E}_{\pi}[R_{y+1} + \gamma q_{\pi} (S_{t+1}, A_{t+1}) | S_t = s, A_t=a]} $$

### Bellman Expectation Equation Matrix Form

The Bellman expectation equation can be expressed concisely using the induced MRP

$$ \mathsf{v_{\pi} = \mathcal{R}^{\pi} + \gamma \mathcal{P}^{\pi} v_{\pi}}$$

with direct solution

$$ \mathsf{v_{\pi} = (1- \gamma \mathcal{P}^{\pi})^{-1} \mathcal{R}^{\pi}} $$

### Optimal Value Function

The optimal state-value function $\mathsf{v_* (s)}$ is the maximum value function over all policies

$$ \mathsf{v_* (s) = \displaystyle\max_{\pi} v_{\pi} (s)} $$

The optimal action-value function $\mathsf{a_* (s,a)}$ is the maximum action-value function over all policies

$$ \mathsf{q_* (s,a) = \displaystyle\max_{\pi} q_{\pi} (s,a)} $$

- The optimal value function specifies the best possible performance in the MDP
- An MDP is "solved" when we know the optimal value fn

### Optimal Policy

Define a partial ordering over policies

$$ \mathsf{\pi \ge \pi ' \quad if \quad v_{\pi} (s) \ge v_{\pi '} (s), \forall s} $$

|Theorem|
|---|
|For any Markov Decision Process<br>$\bullet\;$ There exists an optimal policy $\pi$, that is better than or equal to all other policies, $\pi _* \geq  \pi, \forall \pi $<br>$\bullet\;$  All optimal policies achieve the optimal value function, $$\mathsf{v_{\pi_*} (s) = v_* (s)}$$<br>$\bullet\;$  All optimal policies achieve the optimal action-value function, $$\mathsf{q_{\pi_*} (s,a) = q_* (s,a)}$$|

#### Finding an Optimal Policy

An optimal policy can be found by maximising over $$\mathsf{q_* (s,a)}$$

$$ \mathsf{\pi_*(a\vert s)} = \begin{cases} \mathsf{1\quad if \; a = \underset{a\in\mathcal{A}}{argmax} \; q_* (s,a)} \\ \mathsf{0 \quad otherwise} \end{cases}$$

- There is always a deterministic optimal policy for any MDP
- If we know $\mathsf{q_* (s,a)}$, we immediately have the optimal policy

#### Bellman Optimality Equation

The optimal value functions are recursively related by the Bellman optimality equations:

$$ \mathsf{v_* (s) = \underset{a}{max} \;q_* (s,a)}$$

$$ \mathsf{q_* (s,a) = \mathcal{R}^a_s + \gamma \displaystyle\sum_{s'\in S} \mathcal{P}^a_{ss'} v_* (s')}$$

$$ \mathsf{v_* (s) = \underset{a}{max} \; \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in S} \mathcal{P}^a_{ss'}v_* (s')}$$

$$ \mathsf{q_* (s,a) = \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in S} \mathcal{P}^a_{ss'} \underset{a'}{max} \; q_* (s', a')} $$

#### Solving the Bellman Optimality Equation

- Bellman Optimality Equation is non-linear
- No closed form solution (in general)
- Many iterative solution methods
  - Value Iteration
  - Policy Iteration
  - Q-learning
  - Sarsa

#### Extensions to MDPs

- Infinite and continuous MDPs
- Partially observable MDPs
- Undiscounted, average reward MDPs

##### Infinite MDPs

The following extensions are all possible:

- Countably infinite state and/or action spaces
  - Straightforward
- Continuous state and/or action spaces
  - Closed form for linear quadratic model(LQR)
- Continuous Time
  - Requires partial differential equations
  - Hamilton-Jaccobian-Bellman (HJB) equation
  - Limiting case of Bellman equation as time-step $\rightarrow 0$

##### POMDPs

A Partially Observab;e Markov Decision Process is an MDP with hidden states. It is a hidden Markov model with actions.

| Definition|
|---|
|A POMDP is a tuple $\mathcal{\langle S, A, \textcolor{red}{O}, P, R,\textcolor{red}{Z}, \gamma \rangle}$<br>$\bullet\;\; \mathcal{S}\;$ is a finite set of states<br>$\bullet\;\; \mathcal{A}\;$ is a finite set of actions<br>$\bullet\;\; \textcolor{red}{\mathcal{A}}\;$ is a finite set of observation<br>$\bullet\;\; \mathcal{P}\;$ is a state transition probability matrix<br>$$\quad\,\mathsf{\mathcal{P}^a_{ss'} = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] }$$<br>$\bullet\;\; \mathcal{R}\;$ is a reward function<br>$$\quad\,\mathsf{\mathcal{R}^a_s = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] }$$<br>$\bullet\;\; \mathcal{\textcolor{red}{Z}}\;$ is an observation function<br>$$\quad\,\mathsf{\mathcal{Z}^a_{s'o} = \mathbb{P} [O_{t+1} = o \vert S_{t+1} = s', A_t = a] }$$<br>$\bullet\;\; \mathcal{\gamma}\;$ is a discount factor $\gamma \in [0, 1]$|

##### Belief States

A history $H_t$ is a sequence of actions, observations and rewards

$$\mathsf{H_t = A_0, O_1, R_1, \dots , A_{t-1}, O_t, R_t}$$

A belief state $b(h) is a probablity distribution over states, conditioned on the history h

$$\mathsf{b(h) = (\mathbb{P} [ S_t = s^1 \vert H_t = h ], \dots , \mathbb{P}[S_t = s^n \vert H_t = h])}$$

##### Reductinos of POMDPs

- The history $\mathsf{H_t}$ satisfies the Markov property
- The belief state $\mathsf{b(H_t)}$ satisfies the Markov property
- A POMDP can be reduced to an (infinite) history tree
- A POMDP can be reduced to an (infinite) belief state tree

#### Ergodic Markov Process

An ergodic Markov process is

- Recurrent: each state is visted an infinite number of times
- Aperiodic: each state is visited without any systematic period

|Theorem|
|---|
|An ergodic Markov process has a limiting stationary distribution $\mathsf{d^{\pi}(s)}$ with the property<br><center>$$ \mathsf{d^{\pi} (s) = \displaystyle\sum_{s' \in S} d^{\pi} (s') \mathcal{P}_{s's}}$$</center>|

#### Ergidic MDP

|Definition|
|---|
| An MDP is ergodic if the Markov chain induced by any policy is ergodic|

For any policy $\pi$, and ergodic MDP has an average reward per time-step $\rho ^{\pi}$ that is independent of start state.

$$ \mathsf{\rho^{\pi} = \underset{T\rightarrow \infty}{lim} \; {1\over T}\mathbb{E} \left[\displaystyle\sum^T_{t=1} R_t\right]} $$

#### Average Reward Value Function

- The value function of an undiscounted, ergodic MDP can be expressed in terms of average reward.
- $\mathsf{\tilde{v}_{\pi} (s)$ is the extra reward due to starting from state s,

$$\mathsf{\tilde{v}_{\pi} (s) = \mathbb{E}_{\pi} \left[ \displaystyle\sum^{\infty}_{k=1} (R_{t+k} - \rho^{\pi})\;\vert\;S_t =s \right ]}$$

There is a corresponding average reward Bellman equation,

$$\mathsf{\tilde{v}_{\pi} (s) = \mathbb{E}_{\pi} \left[(R_{t+1} - \rho^{\pi}) + \displaystyle\sum^{\infty}_{k=1} (R_{t+k} - \rho^{\pi})\;\vert\;S_t =s \right ]}$$

$$\quad\quad = \mathsf{\mathbb{E}_{\pi} \left[(R_{t+1} - \rho^{\pi}) + \tilde{v}_{\pi} (s+1)\;\vert\;S_t =s \right ]}$$
