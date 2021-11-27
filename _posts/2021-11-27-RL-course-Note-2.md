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

### Markov Reward Process

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
