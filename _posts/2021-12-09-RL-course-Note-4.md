---
title: RLcourse note4 - Lecture 4 Model-Free Prediction
author: DS Jung
date: 2021-12-09 21:20:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec4_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/PnHCvfgC_ZA)
{: .text-center}

## Introduction of Model-Free Reinforcement Learning

Estimate the Value function of an unknown MDP

### 4강 소감

asfds

### Monte-Carlo Reinforcement Learning

- MC methods learn directly from episodes of experience
- MC is model-free: no knowledge of MDP transitions / rewards
- MC learns from complete episodes: no booststrapping
- MC uses the simplest possible idea: value = mean return
- Caveat: can only apply MC to episodic MDPs
  - All episodes must terminate

### First-Vist Monte-Carlo Policy Evaluation

- Goal: learn $\mathsf{v_\pi}$ from episodes of experience under policy $\pi$

$$ \mathsf{ S_1, A_1, R_2, \dots, S_k \sim \pi } $$

- Monte-Carlo policy evaluation uses empirical mean return instead of expected return
*- state s 를 최초로 visit한 time t만 사용한다.*
- counter $\mathsf{N(s)}$ 세상에 뭔 counter가 입력변수를 가져 어이가 없네
- Total return $\mathsf{S(s) \leftarrow S(s) + G_t}$
- Estimated Value by mean return $\mathsf{V(s) = S(s) /N}$
- By law of large number, $\mathsf{V(s) \rightarrow v_\pi(s) \;as\; N \rightarrow \infty}$ V는 mean return이고 v는 state return인데 이것도 어이없네...

### Every-Visit Monte-Carlo Policy Evaluation

- Goal: learn $\mathsf{v_\pi}$ from episodes of experience under policy $\pi$

$$ \mathsf{ S_1, A_1, R_2, \dots, S_k \sim \pi } $$

- Monte-Carlo policy evaluation uses empirical mean return instead of expected return
- *state s를 방문하는 모든 step t를 이용한다.*
- counter $\mathsf{N(s)}$
- Total return $\mathsf{S(s) \leftarrow S(s) + G_t}$
- Estimated Value by mean return $\mathsf{V(s) = S(s) /N}$
- By law of large number, $\mathsf{V(s) \rightarrow v_\pi(s) \;as\; N \rightarrow \infty}$
- 예시로 Blackjack 이 주어졌는데 사실 이 예시는 MDP로 정의되지 않음 ㅋㅋ

### Incremental Mean

The mean $\mu_1, \mu_2, \dots$ of a sequence $\mathsf{x_1,x_2,\dots}$ can be computed incrementally, $\mu_k$ 는 sequence $\mathsf{x_1, \dots, x_k}$ 의 평균

$$\begin{aligned}
\mu _k &=\mathsf{\frac{1}{k} \displaystyle\sum^{k}_{j=1} x_j =\frac{1}{k} \left( x_k + \displaystyle\sum^{k-1}_{j=1} x_j \right)}\\
&=\mathsf{\frac{1}{k} (x_k + (k-1)\mu_{k-1})}\\
&=\mathsf{\mu_{k-1}+\frac{1}{k}(x_k-\mu_{k-1})}\\
\end{aligned}$$

별거아닌 식인데 귀찮게 써버렸네

### Incremental Monte-Carlo Updates

- For each state $\mathsf{S_t}$ with return $\mathsf{G_t}$

$$ \mathsf{V(S_t) \leftarrow V(S_t) + \frac{1}{N} (G_t - V(S_t))} $$

S_t가 같은 s 를 의미해서 둘다 t로 쓴건 알겠지만 그러면 첫번째 줄이랑 t 정의가 안맞는다 수식 정말 좀....

- In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes

$$ \mathsf{V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))} $$

## Temporal-Difference Learning

- TD methods learn directly from episodes of experience
- TD is model-free: no knowledge of MDP transitions / rewards
- TD learns from incomplete episodes, by *bootstrapping*
- TD updates a guess towards a guess

### MC and TD

- Goal: learn $\mathsf{v_\pi}$ online from experience under policy $\pi$
- Incremental every-visit Monte-Carlo
  - Update value $\mathsf{V(S_t)}$ toward actual return $\mathsf{G_t}$

$$ \mathsf{ V(S_t) \leftarrow V(S_t) +\alpha (G_t - V(S_t))}$$

- Simplest temporal-difference learning algorithm: TD(0)
  - Update value $\mathsf{V(S_t)}$ toward estimated return $\mathsf{R_{t+1} +\gamma V(S_t)}$

$$ \mathsf{ V(S_t) \leftarrow V(S_t) +\alpha (R_{t+1} +\gamma V(S_t) - V(S_t))}$$

- $\mathsf{R_{t+1} +\gamma V(S_{t+1})} $ is called the TD target
- $\mathsf{ \delta _t = R _{t+1} + \gamma V ( S _{t+1})-V(S _t)}$ is called the TD error

### Advantages and Disadvantages of MC vs. TD

- TD can learn before knowing the final outcome
  - TD can learn online after every step
  - MC must wait until end of episode before reutrn is known
- TD can learn without the final outcome
  - TD can learn from incomplete sequences
  - MC can only learn from complete sequences
  - TD works in continuing (non-terminating) environmnets
  - MC only works for episodic (terminating) environments

### Bias/Variance Trade-Off

- Return $\mathsf{G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{T-1} R_T}$ is unbiased estimate of $\mathsf{v_\pi(S_t)}$
- True TD target $\mathsf{R_{t+1} + \gamma v-\pi(S_{t+1})}$ is unbiased estimate of $\mathsf{v_\pi(S_t)}$
- TD target  $\mathsf{R_{t+1} + \gamma V(S_{t+1})}$ is biased estimate of $\mathsf{v_\pi(S_t)}$
- TD target is much lower variance than the return:
  - Return depends on many random actions, transitions, rewards
  - TD target depends on one random action, transition, reward

### Advantages and Disadvantages of MC vs. TD 2

- MC has gigh variance, zero bias
  - Good convergence properties
  - (even with function approximation)
  - Not very sensitive to initial value
  - Very simple to understand and use
- TD has low variance, some bias
  - Usually more efficient than MC
  - TD(0) converges to $\mathsf{v_\pi(s)}$
  - (but not always with function approximation)
  - More sensitive to initial value

### Batch MC and TD

- MC and TD converge: $\mathsf{V(s) \rightarrow v_\pi(s)}$ as experience $\rightarrow \infty$
- But what about batch solution for finite experience?

$$\begin{aligned}
&\mathsf{s^1_s, a^1_1, r^1_2, \dots , s^1_{T_1}} \\
&\quad\quad\vdots \\
&\mathsf{s^K_s, a^K_1, r^K_2, \dots , s^K_{T_K}}
\end{aligned}$$

- e.g Repeatedly sample episode $\mathsf{k\in[1,K]}$
- Apply MC or TD(0) to episode $\mathsf{k}$

### Certainty Equivalence

- EC converges to solution with minimum mean-squared error
  - Best fit to the observed returns<br>
  $$
   \displaystyle\sum^K_{k=1} \sum^{T_k}_{t=1}(G^k_t - V (s^k_t))^2
   $$
  - In the AB example, $\mathsf{V(A) = 0}$
- TD(0) converges to solution of max likelihood Markov model
  - Solution to the MDP $\mathcal{\langle S, A, \hat{P}, \hat{R}, \gamma \rangle}$ that best fits the data<br>
  $$\begin{aligned}
  \mathsf{\hat{\mathcal{P}}^a_{s,s'}} &= \mathsf{\frac{1}{N(s,a)} \displaystyle\sum^K_{k=1}\sum^{T_k}_{t=1}1(s^k_t, a^k_t, s^k_{t+1} = s, a, s')} \\
   \mathsf{\hat{\mathcal{R}}^a_s} &= \mathsf{\frac{1}{N(s,a)} \displaystyle\sum^K_{k=1}\sum^{T_k}_{t=1}1(s^k_t, a^k_t = s, a)r^k_t}
  \end{aligned}$$
  - In the AB example, $\mathsf{V(A) = 0.75}$

### Advantages and Disadvantages of MC vs. TD 3

- TD exploits Markov property
  - Usually more efficient in Markov environments
- MC does not exploit Markov property
  - Usually more effective in non-Markov environmnets

### Bootstrapping and Sampling

- Bootstrapping: update involves an estimate
  - MC does not bootstrap
  - DP bootstraps
  - TD bootstraps
- Sampling: update samples an expectation
  - MC samples
  - DP does not sample
  - TD samples

![thumb1](/assets/pic/Note3_figure1.JPG){: width="400px" height="400px"}

## TD ($\lambda$)

### n-Step Prediction

- TD target look n steps into the future
- $\infty$-step prediction goes to MC

### n-Step Return

- Consider the following n-step returns for n=1,2,$\infty$:

$$\begin{aligned}
\mathsf{n=1\quad (TD)\quad} & \mathsf{G^{(1)}_t = R_{t+1} + \gamma V(S_{t+1})} \\
\mathsf{n=2\quad \quad\quad\quad\,} & \mathsf{G^{(2)}_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2 V(S_{t+2})} \\
\vdots \quad\quad\quad\quad\quad\; & \;\, \vdots \\
\mathsf{n=\infty \;\, (MC)\quad} & \mathsf{G^{(\infty)}_t = R_{t+1} + \gamma R_{t+2} +\dots + \gamma ^{T-1} R_T}
\end{aligned}$$

- Define the n-step return

$$ \mathsf{G^{(n)}_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{n-1} R_{t+n} + \gamma ^n V(S_{t+n})} $$

- n-step temporal-difference learning

$$ \mathsf{ V(S_t) \leftarrow V(S_t) +\alpha (G^{(n)}_t - V(S_t))}$$

### Averaging n-Step Returns

- We can average n-step returns over different n
- e.g. average the 2-step and 4-step returns

$$ \mathsf{\frac{1}{2} G^{(2)} + \frac{1}{2} G^{(4)} } $$

- Combines information from two different time-steps
- Can we efficiently combine information from all time-steps?

### $\lambda$-return

- The $\lambda$-return $\mathsf{G^\lambda _t }$ combines all n-step reutrns $\mathsf{G^{(n)}_t}$
- Using weight $\mathsf{(1-\lambda)\lambda ^{n-1}}$

$$ \mathsf{G^\lambda_t = (1-\lambda) \displaystyle\sum^\infty_{n=1} \lambda^{n-1} G^{(n)}_t} $$

- discount $\lambda$를 주면서 평균을 구하는 방법임

- Forward-view TD($\lambda$)

$$ \mathsf{ V(S_t) \leftarrow V(S_t) +\alpha (G^\lambda_t - V(S_t))}$$

- 정체 평균 G 에 대하여 update 함

### TD($\lambda$) Weighting Function

$$ \mathsf{G^\lambda_t = (1-\lambda) \displaystyle\sum^\infty_{n=1} \lambda^{n-1} G^{(n)}_t} $$

### Forward-view TD($\lambda$)

- update value function towards the $\lambda$-return
- Forward-view looks into the future to compute $mathsf{G^\lambda_t}$
- Like MC, can only be computed from episodes

### Backward View TD($\lambda$)

- Forward view provides theory
- Backward view provides mechanism
- Update online, every step, from incomplete sequences

- Keep an eligibility trace for every state s
- Update value $\mathsf{V(s)}$ for every state $\mathsf{s}$

$$\begin{aligned}
\delta_t &= \mathsf{R_{t+1} + \gamma V(S_{t+1} - V(S_t))}\\
\mathsf{V(s)} &\leftarrow V(s) + \alpha\delta _t E_t(s)
\end{aligned}$$

### TD($\lambda$) and TD(0)

- When $\lambda = 0$, only current state is updated

$$\begin{aligned}
\mathsf{E_t(s)} &= \mathsf{1(S_t = s)}\\
\mathsf{V(s)} &\leftarrow V(s) + \alpha\delta _t E_t(s)
\end{aligned}$$

- This is exactly equivalent to TD(0) update

$$ \mathsf{V(s) \leftarrow V(s) + \alpha\delta _t} $$

### TD($\lambda$) and MC

- When $\lambda =1$, credit is deferred until end of episode
- Consider episodic environmnets with offline updates
- Over the course of an episode, total update for TD(1) is the same as total update for MC

|Theorem|
|---|
|The sum of offline updates is identical for forward-view and backward-view TD($\lambda$)<br><center>$$ \mathsf{\displaystyle\sum ^T_{t=1} \alpha\delta_t E_t(s) = \sum^T_{t=1} \alpha \left( G^\lambda_t - V(S_t)\right)1(S_t=s)} $$</center>|

### MC and TD(1)

- Consider an episode where $\mathsf{s}$ is visited once at time-step k,
- TD(1) eligibility trace discounts time since visit,

$$ \begin{aligned}
\mathsf{E_t(s)} &= \mathsf{ \gamma E_{t-1}(s) + 1(S_t =s)} \\
&= \begin{cases}\; 0 & \mathsf{if\; t< k} \\ \;\mathsf{\gamma ^{t-k}} & \mathsf{if\; t\geq k} \end{cases}
\end{aligned} $$

- TD(1) updates accumulate error online

$$ \mathsf{\displaystyle \sum^{T-1}_{t=1} \alpha\delta_t E_t(s) = \alpha \sum^{T-1}_{t=k} \gamma^{t-k}\delta_t = \alpha(G_k - V(S_k))} $$

- By end of episode it accumulates total error

$$ \mathsf{\delta_k + \gamma\delta_{k+1} + \gamma^2\delta_{k+2} + \dots + \gamma^{T-1-k} \delta_{T-1}} $$

### Telescoping in TD(1)

When $\lambda = 1$, sum of TD errors telescopes into MC error,

$$\begin{aligned}
& \mathsf{\delta_k + \gamma\delta_{k+1} + \gamma^2\delta_{k+2} + \dots + \gamma^{T-1-k} \delta_{T-1}} \\
=& \mathsf{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)} \\
+& \gamma \mathsf{R_{t+2} + \gamma^2 V(S_{t+2}) - \gamma V(S_{t+1})} \\
+& \gamma^2 \mathsf{R_{t+3} + \gamma^3 V(S_{t+3}) - \gamma^2 V(S_{t+2})} \\
& \quad \vdots \\
+& \gamma^{T-1-t} \mathsf{R_T + \gamma^{T-t} V(S_T) - \gamma^{T-1-t} V(S_{T-1})} \\
=& \mathsf{R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-1-t} R_T - V(S_t)} \\
=& \mathsf{G_t - V(S_t)}
\end{aligned}$$

### TD($\lambda$) and TD(1)

- TD(1) is roughly equivalent to every-visit Monte-Carlo
- Error is accumulated online, step-by-step
- If value function is only updated offline at end of episode
- Then total update is exactly the same as MC

### Telescoping in TD($\lambda$)

For general $\lambda$, TD errors also telescope to $\lambda$-error, $\mathsf{G^\lambda_t - V(S_t)} $

$$ \begin{aligned}
\mathsf{G^\lambda _t - V(S_t) }=& \mathsf{-V(S_t) + (1-\lambda)\lambda^0(R_{t+1} + \gamma V(S_{t+1})) } \\
& \quad\quad\quad\;\,\mathsf{+\, (1-\lambda)\lambda^1 (R_{t+1}+\gamma R_{t+2} + \gamma^2 V(S_{t+2})) } \\
& \quad\quad\quad\;\,+\,\dots\\
=& \mathsf{-V(S_t) + (\gamma\lambda)^0(R_{t+1} + \gamma V(S_{t+1}) -\gamma\lambda V(S_{t+1})) } \\
& \quad\quad\quad\;\,\mathsf{+\, (\gamma\lambda)^1 (R_{t+1}+ \gamma V(S_{t+2}) -\gamma\lambda V(S_{t+2})) } \\
& \quad\quad\quad\;\,+\,\dots\\
=& \quad\quad\quad\quad\;\, \mathsf{ (\gamma\lambda)^0(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)) } \\
& \quad\quad\quad\;\,\mathsf{+\, (\gamma\lambda)^1 (R_{t+1}+ \gamma V(S_{t+2}) -V(S_{t+1})) } \\
& \quad\quad\quad\;\,+\,\dots\\
=&\;\mathsf{\delta _t + \gamma\delta _{t+1} + \gamma\lambda)^2 \delta_{t+2} + \dots}
\end{aligned}$$

### Forward and Backwards TD($\lambda$)

- Consider an episode where s is visited once at time-step k
- TD($\lambda$) eligibility trace discounts time since visit,

$$ \begin{aligned}
\mathsf{E_t(s)} &= \mathsf{ \gamma\lambda E_{t-1}(s) + 1(S_t =s)} \\
&= \begin{cases}\; 0 & \mathsf{if\; t< k} \\ \;\mathsf{(\gamma\lambda)^{t-k}} & \mathsf{if\; t\geq k} \end{cases}
\end{aligned} $$

- Backward TD($\lambda$) updates accumulate error online

$$ \mathsf{\displaystyle \sum^T_{t=1} \alpha\delta_t E_t(s) = \alpha \sum^T_{t=k} (\gamma\lambda)^{t-k}\delta_t = \alpha(G_k - V(S_k))} $$

- By end of episode it accumulates total error for $\lambda$-return
- For multiple visits to $\mathsf{s, E_t(s)}$ accumulates many errors

### Offline Equivalence of Forward and Backward TD

Offline updates

- Updates are accumulated within episode
- but applied in batch at the end of episode

Online updates

- TD($\lambda$) updates are applied online at each step within episode
- Forward and backward-view TD($\lmabda$) are slightly different
- NEW: Exact online TD($\lambda$) achieves perfect equivalene
- By using a slightly different form of eligibility trace

### Summary of Forward and Backward TD($\lambda$)

| Offline updates | $\lambda=0$ | $\lambda \in (0,1)$ | $\lambda =1$ |
| :---: | :---: | :---: | :---: |
|Backward view <br><br>Forward view | TD(0) <Br> $\parallel$ <br> TD(0) | TD($\lambda$) <br> $\parallel$ <br> Forward TD($\lambda$) | TD(1) <Br> $\parallel$ <br> MC |

| Online updates | $\lambda=0$ | $\lambda \in (0,1)$ | $\lambda =1$ |
| :---: | :---: | :---: | :---: |
|Backward view <br><br>Forward view <br><br> Exact Online | TD(0) <Br> $\parallel$ <br> TD(0) <br> $\parallel$ <br> TD(0) | TD($\lambda$) <br> $\nparallel$ <br> Forward TD($\lambda$) <br> $\parallel$ <br> Exact Online TD($\lambda$) | TD(1) <Br> $\nparallel$ <br> MC <br> $\nparallel$ <br> Exact Online TD(1) |

here indicates equivalence in total update at end of episode.
