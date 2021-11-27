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

