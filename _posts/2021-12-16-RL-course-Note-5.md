---
title: RLcourse note - Lecture 5 Model-Control
author: DS Jung
date: 2021-12-16 06:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec5_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/0g4j2k_Ggc4)
{: .text-center}

## Introduction of Model-Free Control

Optimise the Value function of an unknown MDP<br>
Model-free control can solve Some MDP problems which modelled:

- MDP model is unknown, but experience can be sampled
- MDP model is known, but is too big to use, except by samples

### 5강 소감

asfds

## On-Policy Monte-Carlo Control

- On-policy Learning
  - Learn on the job
  - Learn about policy $\pi$ from experience sampled from $\pi$
- Off-policy Learning
  - Look over someone's shoulder
  - Learn about poilcy $\pi$ from experience sampled from $\mu$

### Generalised Policy Iteration

#### Generalised Policy Iteration With Monte-Carlo Evaluation

Policy evaluation - Monte-Carlo policy evaluation<br>
Policy improvement - Greedy policy improvement?

#### Model-Free Policy Iteration Using Action-Value Function

- Greedy policy improvement over $\mathsf{V(s)}$ requires model of MDP

$$ \mathsf{\pi'(s) = \underset{a \in \mathcal{A}}{argmax}\;\mathcal{R}^a_s + \mathcal{P}^a_{ss'}V(s') } $$

- Greedy policy improvement over $\mathsf{Q(s,a)}$ is model-free

$$ \mathsf{\pi'(s) = \underset{a \in \mathcal{A}}{argmax}\;Q(s,a) } $$

### Exporation

#### $\epsilon$-Greedy Exploration

- Simplest idea for ensuring continual exploration
- All m actions are tried with non-zero probability
- With probability $1-\epsilon$ choose the greedy action
- With probability $\epsilon$ choose an action at random

$$ \mathsf{\pi(a\vert s)=} \begin{cases}
\epsilon/\mathsf{m} +1 - \epsilon & \mathsf{if\;a^*=\underset{a\in \mathcal{A}}{argmax}\;Q(s,a)} \\
\epsilon/\mathsf{m} & \mathsf{otherwise}
\end{cases} $$

또 또 식 이상하게 쓴다... policy 결과가 왜 숫자가 나와..... 이번엔 $\pi$ output이 action이 아니라 action을 선택할 확률이다 이거지.... 좀 문자 다른거 써라..

#### $\epsilon$-Greedy Policy Improvement

|Theorem|
|---|
|For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $\mathsf{q_\pi}$ is an improvement, $\mathsf{v_{\pi'}(s)\geq v_\pi(s)}$|

$$\begin{aligned}
\mathsf{q_\pi (s,\pi'(s))} &= \mathsf{\displaystyle\sum_{a\in\mathcal{A}}\pi'(a\vert s)q_\pi(s,a)}\\
&=\mathsf{\epsilon/m \displaystyle\sum_{a\in\mathcal{A}}q_\pi (s,a)+(1-\epsilon)\underset{a\in\mathcal{A}}{max}\;q_\pi(s,a)}\\
&\leq \mathsf{ \epsilon/m \displaystyle\sum_{a\in\mathcal{A}}q_\pi (s,a)+(1-\epsilon) \displaystyle\sum_{a\in\mathcal{A}} \frac{\pi(a\vert s)-\epsilon/m}{1-\epsilon}q_\pi(s,a) } \\
&=\mathsf{\displaystyle\sum_{a\in\mathcal{A}}\pi(a\vert s)q_\pi(s,a)=v_\pi(s)}
\end{aligned}$$

Therefor from policy improvement theorem, $\mathsf{v_{\pi'}(s)\geq v_\pi(s)}$

사실 greedy만 해도 improvement가 보장되었었는데 일부만 greedy 나머지는 random 할때도 보장되는 건 당연한거긴 하다. 굳이 이런건 또 증명해주네

#### Monte-Carlo Policy Iteration

Policy evaluation - Monte-Carlo policy evaluation $\mathsf{Q=q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

#### Monte-Carlo Policy Control

Every episode:<br>
Policy evaluation - Monte-Carlo policy evaluation $\mathsf{Q\approx q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

### GLIE

|Definition|
|---|
|Greedy in the Limit with Infinite Exploration(GLIE)<br>$\quad \bullet\;$ All state-action pairs are explored infinitely many times,<br><center>$$ \mathsf{ \underset{k\rightarrow\infty}{lim}\;\, N_k(s,a) = \infty } $$</center><br>$\quad \bullet\;$ The policy converges on a greedy policy,<br><center>$$ \mathsf{ \underset{k\rightarrow\infty}{lim}\;\, \pi_k(a\vert a) = 1(a=\underset{a'\in\mathcal{A}}{argmax}\;Q_k(s,a')) } $$</center>|

- For example, $\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\mathsf{\epsilon_k = \frac{1}{k}}$

#### GLIE Monte-Carlo Control

- Sample *k*th episode using $\pi:\mathsf{ \lbrace S_1,A_1,R_2,\dots,S_T \rbrace \sim \pi}$
- For each state $\mathsf{S_t}$ and action $\mathsf{A_t}$ in the episode,

$$\begin{aligned}
&\mathsf{ N(S_t,A_t)\leftarrow N(S_t,A_t)+1 }\\
&\mathsf{ Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\frac{1}{N(S_t,A_t)}(G_t - Q(S_t,A_t)) }
\end{aligned}$$

- Improve policy based on new action-value function

$$\begin{aligned}
\epsilon &\leftarrow \mathsf{ 1/k }\\
\pi &\leftarrow \mathsf{ \epsilon-greedy(Q) }
\end{aligned}$$

$\epsilon$ 값이 episode 마다 점점 작아짐 - exploration 감소

|Theorem|
|---|
|GLIE Monte-Carlo control converges to the optimal action-value functino, $\mathsf{ Q(s,a)\rightarrow q_*(s,a) }$

## On-Policy Temporal-Difference Learning

### MC vs. TD Control

- Temporal-difference (TD) learning has several advantages over Monte-Carlo (MC)
  - Lower variance
  - Online
  - Incomplete sequence
- Natural idea: use TD instead of MC in our control loop
  - Apply TD to $\mathsf{Q(S,A)}$
  - Use $\epsilon$-greedy policy improvement
  - Update every time-step

### Sarsa($\lambda$)

#### Updating Action-Value Functions with Sarsa

$$ \mathsf{ Q(S,A) \leftarrow Q(S,A) + \alpha (R+ \gamma Q(S',A') - Q(S,A)) } $$

#### On-Policy Control With Sarsa

Every time-step<br>
Policy evaluation Sarsa, $\mathsf{Q \approx q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

#### Sarsa Algorithm for On-Policy Control

|Initialize $\mathsf{Q(s,a), \forall s \in S, a \in A(s)}$, arbitrarily,and $\mathsf{Q}(terminal state, \cdot)=0$<br>Repeat (for each episode):<br>$\quad$Initialize S<br>$\quad$Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Take action A, observe R, S'<br>$\quad\quad$Choose A' from S' using policy derived from Q (e.g., $\epsilon$-greedy)<br>$$\quad\quad\mathsf{Q(S,A)\leftarrow Q(S,A) + \alpha[R+\gamma Q(S',Q') - Q(S,A)] }$$<br>$$\quad\quad\mathsf{ S\leftarrow S'; A \leftarrow A';} $$<br>$\quad$until S is terminal|

다음 state의 action 까지 포함하는 TD의 action 확장 버전

#### Convergence of Sarsa

|Theorem|
|---|
| Sarsa converges to the optimal action-value function,<br>$\mathsf{Q(s,a)\rightarrow q_\pi(s,a)}$, under the following conditions:<br>$\quad \bullet\;$ GLOE sequence of policies $\mathsf{ \pi_t(a\vert s) }$<br>$\quad \bullet\;$ Robbins-Monro sequence of step-sizes $\alpha_t$<br><center>$$\begin{aligned} &\mathsf{ \displaystyle\sum^\infty_{t=1} \alpha_t = \infty } \\ &\mathsf{ \displaystyle\sum^\infty_{t=1} \alpha^2_t < \infty } \end{aligned} $$</center>|

#### n-Step Sarsa

- Consider the following n-step returns for $n = 1, 2, \infty$:

$$\begin{aligned}
\mathsf{n=1\;\; (Sarsa)\;\;} & \mathsf{q^{(1)}_t = R_{t+1} + \gamma Q(S_{t+1})} \\
\mathsf{n=2\quad \quad\quad\quad\,} & \mathsf{q^{(2)}_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2 Q(S_{t+2})} \\
\vdots \quad\quad\quad\quad\quad\; & \;\, \vdots \\
\mathsf{n=\infty \;\, (MC)\quad} & \mathsf{q^{(\infty)}_t = R_{t+1} + \gamma R_{t+2} +\dots + \gamma ^{T-1} R_T}
\end{aligned}$$

- Define the n-step return

$$ \mathsf{q^{(n)}_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{n-1} R_{t+n} + \gamma ^n Q(S_{t+n})} $$

- n-step Sarsa updates $\mathsf{Q(s,a)}$ towards the n-step Q-return

$$ \mathsf{ Q(S_t, A_t) \leftarrow Q(S_t, A_t) +\alpha \left(q^{(n)}_t - Q(S_t,A_t)\right) }$$

#### Forward View Sarsa($\lambda$)

- The $q^\lambda$ return combines all n-step Q-reutrns $\mathsf{q^{(n)}_t}$
- Using weight $\mathsf{(1-\lambda)\lambda ^{n-1}}$

$$ \mathsf{q^\lambda_t = (1-\lambda) \displaystyle\sum^\infty_{n=1} \lambda^{n-1} q^{(n)}_t} $$

- Forward-view Sarsa($\lambda$)

$$ \mathsf{ Q(S_t,A_t) \leftarrow Q(S_t,A_t) +\alpha \left(q^\lambda_t - Q(S_t,A_t)\right)}$$

#### Backward View Sarsa($\lambda$)

- Just like TD($\lambda$), we use eligibility traces in an online algorithm
- But Sarsa($\lambda$) has one eligibility trace for each state-action pair

$$\begin{aligned}
&\mathsf{E_0(s,a) =0} \\
&\mathsf{E_t(s,a) =\gamma \lambda E_{t-1}(s,a) + 1(S_t=s, A_t=a)}
\end{aligned}$$

- $\mathsf{Q(s,a)}$ is updated for every state s and action a
- In proportion to TD-error $\delta_t$ and eligibility trace $\mathsf{E_t(s,a)}$

$$\begin{aligned}
&\delta_t = \mathsf{R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))}\\
&\mathsf{Q(s,a) \leftarrow Q(s,a) + \alpha\delta _t E_t(s,a)}
\end{aligned}$$

#### Sarsa($\lambda$) Algorithm

|Initialize $\mathsf{Q(s,a)}$ aribitrarily, for all $s \in S, a \in A(s)$<br>Repeat (for each episode):<br>$\quad$E(s,a) = 0, for all $s \in S, a \in A(s)$<br>$\quad$Initialize S, A<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Take action A, observe R, S'<br>$\quad\quad$Choose A' from S' using policy derived from Q (e.g., $\epsilon$-greedy)<br>$$\quad\quad \delta \leftarrow R+ \gamma Q(S', A') - Q(S,A)$$<br> $$\quad\quad E(S,A) \leftarrow E(S,A) + 1 $$<br> $ \quad\quad$ For all $s \in S, a \in A(s)$:<br>$$\quad\quad\quad Q(s,a)\leftarrow Q(s,a) + \alpha\delta E(s,a)$$<br>$$\quad\quad\quad E(s,a) \leftarrow \gamma\lambda E(s,a)$$<br>$$\quad\quad S\leftarrow S'; A \leftarrow A'; $$<br>$\quad$until S is terminal|

## Off-Policy Learning

- Evaluate target policy $\mathsf{ \pi(a \vert s)}$ to compute $\mathsf{v_\pi(s)}$ or $\mathsf{q_\pi(s,a)}$
- While following behaviour policy $\mathsf{\mu(a\vert s)}$

$$ \mathsf{ \lbrace S_1, A_1, R_2, \dots, S_T \rbrace \sim \mu } $$

Why is this important

- Learn from observing humans or other agents
- Re-use experience generated from old policies $\pi_1, \pi_2, \dots, \pi_{t-1}$
- Learn about optimal policy while following exploratory policy
- Learn about multiple policies while following one policy

### Importance Sampling

- Estimate the expectation of a different distribution

$$\begin{aligned}
\mathsf{ \mathbb{E}_{X \sim P}[f(X)] } &= \sum \mathsf{P(X)f(X)} \\
&= \sum \mathsf{Q(X) \frac{P(X)}{Q(X)} f(X) } \\
&= \mathsf{ \mathbb{E}_{X \sim Q} \left[ \frac{P(X)}{Q(X)}f(X)\right]}
\end{aligned}$$

#### Importance Sampling for Off-Policy Monte-Carlo

- Use returns generated from $\mu$ to evaluate $\pi$
- Weight return $\mathsf{G_t}$ according to similarity between policies
- Multiply importance sampling corrections along whole episode

$$ \mathsf{ G^{\pi/\mu}_t = \frac{\pi(A_t \vert S_t)}{\mu(A_t \vert S_t)} \frac{\pi(A_{t+1} \vert S_{t+1})}{\mu(A_{t+1} \vert S_{t+1})} \dots \frac{\pi(A_T \vert S_T)}{\mu(A_T \vert S_T)} G_t } $$

- Update value towards corrected return

$$ \mathsf{ V(S_t) \leftarrow V(S_T) + \alpha\left(G^{\pi/\mu}_t - V(S_t)\right) } $$

- Cannot use if $\mu$ is zero when $\pi$ is non-zero
- Importance sampling can dramatically increase variance

#### Importance Sampling for Off-Pollicy TD

- Use TD targets generated from $\mu$ to evaluate $\pi$
- Weight TD target $\mathsf{R+\gamma V(S')}$ by importance sampling
- Only need a single importance sampling correction

$$ \mathsf{ V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t\vert S_t)}{\mu(A_t\vert S_t)} (R_{t+1} + \gamma V(S_{t+1})) - V(S_t)\right) } $$

- Much lower variance than Monte-Carlo importance sampling
- Policies only need to be similar over a single step

### Q-Learning

- We now consider off-policy learning of action-values $\mathsf{Q(s,a)}$
- No importance sampling is required
- Next action ischosen using begaviour policy $\mathsf{ A_{t+1} \sim \mu(\cdot \vert S_t) }$
- But we consider alternative successor action $\mathsf{ A' \sim \pi(\cdot \vert S_t) }$
- And update $\mathsf{Q(S_t,A_t)}$ towards value of alternative action

$$ \mathsf{ Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha \left(R_{t+1} + \gamma Q(S_{t+1},A') - Q(S_t,A_t)\right) } $$

#### Off-Policy Control with Q-Learning

- We now allow both behaviour and target policies to improve
- The target policy $\pi$ is greedy w.r.t. $\mathsf{Q(s,a)}$

$$ \mathsf{ \pi (S_{t+1}) = \underset{a'}{argmax}\;Q(S_{t+1}, a') } $$

- The behaviour policy $\mu$ is e.g. $\epsilon$-greedy w.r.t. $\mathsf{Q(s,a)}$
- The Q-learning target then simplifies:

$$\begin{aligned}
& \mathsf{ R_{t+1} + \gamma Q(S_{t+1}, A') } \\
=& \mathsf{ R_{t+1} + \gamma Q(S_{t+1}, \underset{a'}{argmax}\; Q(S_{t+1}, a')) } \\
=& \mathsf{ R_{t+1} + \underset{a'}{max}\;\gamma Q(S_{t+1}, a') } \\
\end{aligned}$$

#### Q-Learning Control Algorithm

$$ \mathsf{ Q(S,A) \leftarrow Q(S,A) + \alpha \left( R + \gamma\; \underset{a'}{max}\; Q(S',a') - Q(S,A)\right) } $$

|Theorem|
|---|
|Q-learning control converges to the optimal action-value function, $\mathsf{Q(s,a)\rightarrow q_*(s,a)}$

#### Q-Learning Algorithm for Off-Policy Control

|Initialize $\mathsf{Q(s,a) \forall s \in S, a \in A(s)}$, arbitrarily, and $\mathsf{Q(terminal\, state,\cdot)=0}$<br>Repeat (for each episode):<br>$\quad$Initialize S<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)<br>$\quad\quad$Take action A, observe R, S'<br>$$\quad\quad Q(S,A)\leftarrow Q(S,A) + \alpha [ R + \gamma\; max_a Q(S',a)-Q(S,A)]$$<br>$$\quad\quad S\leftarrow S'; $$<br>$\quad$until S is terminal|

## Summary

### Relationship Between DP and TD

||Full Backup (DP)|Sample Backup (TD)|
|---|---|---|
|Bellman Expectation<br> Equation for $\mathsf{v_\pi(s)}$|Iterative Policy Evaluation<br>$$ \mathsf{ V(s) \leftarrow \mathbb{E} [R+\gamma V(S') \vert s] } $$ | TD Learning <br> $$ \mathsf{ V(S)\;\overset{\alpha}{\leftarrow}\; R+\gamma V(S')} $$|
|Bellman Expectation<br> Equation for $\mathsf{q_\pi(s,a)}$|Q-Policy Iteration<br>$$ \mathsf{ Q(s,a) \leftarrow \mathbb{E} [R+\gamma Q(S',A') \vert s,a] } $$ | Sarsa <br> $$ \mathsf{ Q(S,A)\;\overset{\alpha}{\leftarrow}\; R+\gamma Q(S',A')} $$|
| Bellman Optimality<br> Equation for $\mathsf{q_*(s,a)}$|Q-Value Iteration<br>$$ \mathsf{ Q(s,a) \leftarrow \mathbb{E} \left[ R+\gamma \;\underset{a'\in\mathcal{A}}{max}\;Q(S',a') \vert s,a \right] } $$ |Q-Learning <br> $$ \mathsf{ Q(S,A)\;\overset{\alpha}{\leftarrow}\; R+\gamma \;\underset{a'\in\mathcal{A}}{max}\; Q(S',a')} $$|

where $$ \mathsf{ x \overset{\alpha}{\leftarrow} y \equiv x \leftarrow x + \alpha(y-x) } $$
