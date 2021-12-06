---
title: RLcourse note3 - Lecture 3 Planning by Dynamic Programming
author: DS Jung
date: 2021-12-02 10:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec3_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/Nd1-UUMVfz4)
{: .text-center}

## Introduction of Dynamic Programming

**Dynamic** sequential or temporal component to the problem

**Programming** optimising a "program", i.e. a policy

- A method for solving complex problems
- By breaking them down into subproblems
  - Solve the subproblems
  - Combine solutions to subproblems
- Dynamic programming이라 이름이 흠 어울리는것 같지 않은데

### Requirements for Dynamic Programming

Dynamic Programming is a very general solution method for problems which have two properties:

- Optimal substructure
  - *Principle of optimality* applies
  - Optimal solution can be decomposed into subproblems
- Overlapping subproblems
  - Subproblems recure many times
  - Solutions can be cached and reused
- Markov decision processes satisfy both properties
  - Bellman equation gives recursive decomposition
  - Value function stores and reuses solutions

### Planning by Dynamic Programming

- Dynamic programming assumes full knowledge of the MDP
- It is used for *planning* in an MDP
- For prediction:
  - Input: MDP $\mathcal{\langle S, A, P, R, \gamma \rangle} $ and policy $\pi$
  - &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or: MRP $\mathcal{\langle S, P^{\pi}, R^{\pi}, \gamma \rangle}$
  - Output: value function $\mathsf{v_{\pi}}$
- Or for control:
  - Input: MDP $\mathcal{\langle S, A, P, R, \gamma \rangle} $
  - Output: optimal value function $\mathsf{v_*}$
  - &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and: optimal policy $\pi_*$

## Iterative Policy Evaluation

To evaluate given policy $\pi$, iterative application of Bellman expectation backup is applied

Reward를 모르는 것(0)으로 간주하고 backup을 반복하여 Reward 계산 - Policy 변경 없음

### synchronous backup

- At each iteration $k+1$, For all states $\mathsf{s} \in \mathcal{S}$
- Update $\mathsf{v_{k+1}(s)}$ from $\mathsf{v_{k}(s')}$
- where $\mathsf{s'}$ is a successor state of s

$$\begin{aligned}
\mathsf{v_{k+1}(s)} &= \mathsf{\displaystyle\sum_{a \in \mathcal{A}}\pi(a \vert s)\left( \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'}v_k(s')\right)} \\
\mathsf{\boldsymbol{v}^{k+1}} &= \mathsf{\mathcal{\boldsymbol{R}}^{\pi} + \gamma \mathcal{\boldsymbol{P} }^{\pi} v^k}
\end{aligned}$$

## Policy Iteration

- Given a policy $\pi$
  - Evaluate the policy $\pi$

$$ \mathsf{v_{\pi} (s) = \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \dots \vert S_t = s \right]} $$

  - Improve the policy by acting greedily with respect to $\mathsf{v_{\pi}}$

$$ \mathsf{\pi ' = greedy(v_{\pi})} $$

- This process of policy iteration always converges to $\pi_*$

### Policy Improvement

- Consider a determinisitc policy, $\mathsf{a = \pi (s)}$
- Improve the policy by acting greedily

$$ \mathsf{\pi ' (s) = \underset{a \in \mathcal{A}}{argmax}\; q_{\pi} (s,a)} $$

- This improves the value from any state $\mathsf{s}$ over one step

$$ \mathsf{q_{\pi} (s, \pi ' (s)) = \underset{a \in \mathcal {A}}{mas} \; q_{\pi}(s,a) \geq q_{\pi}(s,\pi (s)) = v_{\pi} (s)} $$

- This imporves the value from any state s over one step

$$ \mathsf{q_{\pi} (s, \pi '(s)) = \underset{a \in \mathcal{A}}{max} \; q_{\pi} (s,a) \geq q_\pi (s,\pi(s)) = v_\pi (s)}$$

- It therefor imporves the value function, $\mathsf{v_{\pi'}(s) \geq v_\pi(s)}$

$$\begin{aligned}
\mathsf{v_\pi(s)} &\leq \mathsf{q_\pi(s,\pi'(s)) = \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma v_\pi(S_{t+1})\;\vert\;S_t = s \right]} \\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma q_\pi (S_{t+1}, \pi '(S_{t+1}))\;\vert\;S_t = s \right]}\\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} +\gamma ^2q_\pi (S_{t+2}, \pi '(S_{t+2}))\;\vert\;S_t = s \right]}\\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \dots \;\vert\;S_t = s \right] = v_{\pi '} (s)}
\end{aligned}$$

- If improvements stop,

$$ \mathsf{q_{\pi} (s, \pi '(s)) = \underset{a \in \mathcal{A}}{max} \; q_{\pi} (s,a) = q_\pi (s,\pi(s)) = v_\pi (s)}$$

- Then the Bellman optimality equation has been satisfied

$$ \mathsf{ v_\pi(s) = \underset{a \in \mathcal{A}}{max} \; q_\pi(s,a)} $$

- Therefore $\mathsf{v_\pi(s)=v_*(s)}$ for all $\mathsf{s} \in \mathcal{S}$
- so $\pi$ is an optimal policy

### Modified Policy Iteration

- Does policy evaluation need to converge to $\mathsf{v_\pi}$?
- Or should we introduce  a stopping condition
  - e.g. $\epsilon$-convergence of value function
- Or simply stop after $\mathsf{k}$ iterations of iterative policy evaluation?
- Why not update policy every iteration?
  - This is equivalent to value iteration

### Principle of Optimality

An optimal policy can be subdivided into two components:

- An optimal first action $\mathsf{A_*}$
- Followed by an optimal policy from successor state $\mathsf{S'}$

|Theorem (Principle of Optimality)|
|---|
|A policy $\pi\mathsf{(a \vert s)} $ achieves the optimal value from state s, $\mathsf{v_\pi (s) = v_* (s)}$, if and only if<br>$\quad\bullet\;\;$ For any state $\mathsf{s'}$ teachable from $\mathsf{s}$<br>$\quad\bullet\;\;$ $\pi$ achieves the optimal value from state $\mathsf{s', v_\pi(s')=v_*(s')}$|

## Value Iteration

### Deterministic Value Iteration

- If we know the solution to subproblems $\mathsf{v_*(s')}$
- solution $\mathsf{v_*(s)}$ can be found by one-step lookahead

$$\mathsf{v_*(s) \leftarrow \underset{a \in \mathcal{A}}{max} \; \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}}\mathcal{P}^a_{ss'}v_*(s')}$$

- The idea of value iteration is to apply these updates iteratively
- Intuition: start with final rewards and work backwards
- Still works with loopy, stochastic MDPs
- 진짜 쓰면서 뭔 생각인지 몰라도 안맞는다는 걸 생각을 왜 못하는지 모르겠다... 식좀 검토좀 해줘 제발

### Value Iteration - Define

- Probleam: find optimal policy $\pi$
- Solution: ierative application of Bellman optimality backup
- $\mathsf{v_1 \rightarrow v_2 \rightarrow \dots \rightarrow v_*}$
- Using synchronous backups
  - At each iteration $\mathsf{k+1}$
  - For all states $\mathsf{s \in \mathcal{S}}$
  - Update $\mathsf{v_{k+1}(s)}$ from $\mathsf{v_k(s')}$
- Convergence to $\mathsf{v_*}$ will be proven later
- Unlike policy iteration, there is no explicit policy
- Intermediate value functions may not correspond to any policy

$$\begin{aligned}
\mathsf{v_{k+1}(s)} &= \mathsf{\underset{a \in \mathcal{A}}{max}\left( \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'}v_k(s')\right)} \\
\mathsf{\boldsymbol{v}_{k+1}} &= \mathsf{\underset{a \in \mathcal{A}}{max}\;\mathcal{\boldsymbol{R}}^a+ \gamma \mathcal{\boldsymbol{P} }^a v^k}
\end{aligned}$$

## Extensions to Dynamic Programming

### Synchronous Dynamic Programming Algorithms

| Problem | Bellman Equation | Algorithm |
| :--- | :--- | :---:|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation|
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

- Algorithms are based on state-value function $\mathsf{v_\pi(s)}$ or $\mathsf{v_*(s)}$
- Complexiy $\mathcal{O}\mathsf{mc^2}$ per iteration, for $\mathsf{m}$ actions and $\mathsf{n}$ states
- Could also apply to action-valuefunction $\mathsf{q_\pi(s,a)}$ or $\mathsf{q_*(s,a)}$
- Complexity $\mathcal{O}\mathsf{(m^2n^2)}$ per iteration

### ASynchronous Dynamic Programming

- DP methods described so far used synchronous backups
- i.e. all states are backed up in parallel
- Asynchronous DP backs up states individually, in any order
- For each selected state, apply the appropriate backup
- Can significantly reduce computation
- Guaranteed to converge if all states continue to be selected

Three simple ideas for asynchronous dynamic programming:

- In-placedynamic programming
- Prioritised sweeping
- Real-time dynamic programming

#### In-Place Dynamic Programming

- Synchronous value iteration stores two copies of value function

for all $\mathsf{s}$ in $\mathcal{S}$

$$\mathsf{v_{new}(s)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_{old}(s')\right)}$$

$$ \mathsf{ v_{old} \leftarrow v_{new} }$$

- In-place value iteration only stores one copy of value function

for all $\mathsf{s}$ in $\mathcal{S}$

$$\mathsf{v(s)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v(s')\right)}$$

#### Prioritised Sweeping

- Use magnitude of Bellman error to guide state selection, e.g.

$$ \mathsf{\left\vert \underset{a\in\mathcal{A}}{max}\; \left( \mathcal{R}^a_s+\gamma \displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v(s') \right) - v(s) \right\vert }$$

- Backup the state with the largest remaining Bellman error
- Update Bellman error of affected states after each backup
- Requires knowledge of reverse dynamics (predecessor states)
- Can be implemented efficiently by maintaining a priority queue

#### Real-Time dynamic Programming

- Idea: only states that are relevant to agent
- Use agent's experience to guide the selection of states
- After each time-step $\mathsf{S_t, A_t, R_{t+1}}$
- Backup the stae $\mathsf{S_t}$

$$\mathsf{v(S_t)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_{S_t}+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{S_t s'}v(s')\right)}$$

### Full-Width Backups

- DP uses full-width backups
- For each backup (sync or async)
  - Every succesor state and action is considered
  - Using knowledge of the MDP transitions and reward function
- DP is effective for medium-sized probelms (millons of states)
- For large problems DP suffers Bellman's curse of dimensionality
  - Number of states $\mathsf{n= \vert \mathcal{S} \vert}$ grows exponentially with number of state variables
- Even one backup can be too expensive

### Sample Backups

- Using sample rewards and sample transitions $\mathsf{ \langle S, A, R, S' \rangle}$
- Instead of reward function $\mathcal{R}$ and transition dynamics $\mathcal{P}$
- Advantages:
  - Model-free: no advance knowledge of MDP required
  - Break the curse of dimensionality through sampling
  - Cost of backup is constant, independent of $\mathsf{n} = \vert \mathcal{S} \vert$

### Approximate Dynamic Programming

- Approximate the value function
- Using a function approximator $\mathsf{\hat{v}(s,\boldsymbol{w})}$
- Apply dynamic programming to $\mathsf{\hat{v}(\cdot ,\boldsymbol{w})}$
- e.g. Fitted VAlue Iteration repeats at each iteration $\mathsf{k}$,
  - Sample states $\mathcal{\tilde{S} \subseteq S}$
  - For each state $\mathcal{s \in \tilde{S}}$, estimate target value using Bellman optimality equation,
$$\mathsf{\tilde{v}_k(s)=\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}\hat{v}(s', \boldsymbol{w_k})\right)}$$
  - Train next value function $\mathsf{\hat{v}(s', \boldsymbol{w_{k+1}})}$ using targets $\mathsf{ \lbrace \langle s,\tilde{v}_k(s)\rangle\\rbrace}$

## Contraction Mapping

contraction mapping theorem resolve convergence of value iteration $v_*$, policy evaluation $v_\pi$, policy iteration, uniqueness of solution, convergence speed.

### Value Function Space

- Consider the vector space $\mathcal{V} $ over value functions, $\mathcal{\vert S \vert}$ dimensions
- Each point in this space fully specifies a value function $\mathsf{v(s)}$
- Bellman backup brings value function closer
- therefore backup must converge on a unique solution

### Value Function $\infty$-Norm

- distance between state-value functions $\mathsf{u}$ and $\mathsf{v}$ by $\infty$-norm is largest difference

$$ \mathsf{\Vert u-v\Vert_\infty = \underset{s\in\mathcal{S}}{max}\;\vert u(s) -v(s)\vert}$$

### Bellman Expectation Backup is a Contraction

- Define the Bellman expectation backup operator $\mathsf{T^\pi}$,

$$ \mathsf{T^\pi(v) = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v} $$

- This operator is a $\gamma$-contraction, i/e/ it makes value functions closer by at least $\gamma$,

$$\begin{aligned}
\mathsf{\Vert T^\pi(u) - T^\pi (v) \Vert_\infty} &= \mathsf{\Vert (\mathcal{R}^\pi + \gamma \mathcal{P}^\pi u) - (\mathcal{R}^\pi + \gamma\mathcal{P}^\pi v)\Vert_\infty} \\
&= \mathsf{\Vert\gamma\mathcal{P}^\pi (u-v) \Vert_\infty} \\
&\leq \mathsf{\Vert\gamma\mathcal{P}^\pi \Vert u-v \Vert_\infty \Vert_\infty} \\
&\leq \mathsf{\gamma \Vert u-v \Vert_\infty} \\
\end{aligned}$$

### Contraction Mapping Theorem

|Theorem (Contraction Mapping Theorem)|
|---|
|For any metric space $\mathcal{V}$ that is complete (i.e. closed) under an operator $\mathsf{T(v)}$, where T is a $\gamma$-contraction,<br>$\quad\bullet\;\;\mathsf{T}$ converges to a unique fixed point<br>$\quad\bullet\;\;$ At a linear convergence rate of $\gamma$|

### Convergence of Iter. Policy Evaluation and Policy Iteration

- The Bellman expectation operator $\mathsf{T^\pi}$ has a unique fixed point
- $\mathsf{v_\pi}$ is a fixed point of $\mathsf{T^\pi}$ (by Bellman expectation equation)
- By contraction mapping theorem
- Iterative policy evaluation converges on $\mathsf{v_\pi}$
- Policy iteration converges on $\mathsf{v_*}$

### Bellman Optimality Backup is a Contraction

- Define the Bellman optimality backup operator $\mathsf{T^*}$,

$$ \mathsf{T^* (v) = \underset{a\in\mathcal{A}}{max}\;\mathcal{R}^a + \gamma \mathcal{P}^a v}$$

- This operator is a $\gamma$-contraction, i.e. it makes value functions closer by at least $\gamma$ (similar to previous proof)

$$\mathsf{\Vert T^*(u)-T^* (v) \Vert_\infty \leq \gamma \Vert u-v \Vert_\infty }$$

### Convergence of Value Iteration

- The Bellman optimality operator $\mathsf{T^*}$ has a unique fixed point
- $\mathsf{v_*}$ is a fixed point of $\mathsf{T^*}$ (by Bellman expectation equation)
- By contraction mapping theorem
- Value iteration converges on $\mathsf{v_*}$
