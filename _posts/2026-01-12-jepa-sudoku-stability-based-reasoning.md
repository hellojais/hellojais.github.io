---
layout: post
title: "Stability-Based Reasoning: A JEPA Approach to Sudoku"
date: 2026-01-12 00:00:00 +0000
categories: machine-learning representation-learning reasoning
author: Santosh Jaiswal
description: Most reasoning systems traverse invalid states and repair them. I built a JEPA-style Sudoku solver that never enters an invalid state at all. Reasoning emerges from maintaining latent stability rather than fixing errors.
---

> **Author:** Santosh Jaiswal ([@hellojais](https://github.com/hellojais))
> **Paper:** [Stability-Based Reasoning with Joint Embedding Predictive Architectures: A Case Study on Sudoku](#) *(submitted, awaiting endorsement)*
> **Related work:** [TRM (Creswell & Shanahan, 2024)](https://arxiv.org/abs/2402.03654)

---

## TL;DR

Most neural reasoning systems (chain-of-thought, tree-of-thought, Transformer Reasoning Machines) work by generating candidate solutions and repairing violations. I explored a different paradigm: a JEPA-style encoder trained on solved Sudoku boards using masked-view consistency objectives, where inference proceeds by selecting moves that minimally perturb the latent representation. The result: zero constraint violations across all inference steps, latent changes bounded between 10^-9 and 10^-7, no backtracking, no explicit repair. Constraint satisfaction emerges from latent stability rather than error correction.

---

## Two Ways to Reason

There is a dominant paradigm in neural reasoning: generate a trajectory, detect violations, repair them, repeat. Chain-of-Thought prompting, Tree-of-Thought search, and Transformer Reasoning Machines (TRM) all share this structure. They are powerful. They also all assume that reasoning trajectories may pass through invalid or inconsistent states before converging.

This assumption is so common it is rarely examined. But it has a cost. A system that is willing to enter invalid states needs a separate mechanism to detect and escape them. That mechanism can fail. And when it fails, the system produces a fluent, confident wrong answer.

What if reasoning never entered invalid states in the first place?

This is the question this experiment explores. Not as an abstract philosophical point, but concretely, on Sudoku, a constraint satisfaction problem clean enough to make the dynamics visible.

---

## The JEPA Foundation

JEPA (Joint Embedding Predictive Architecture), proposed by Yann LeCun, learns representations by predicting masked or future latent states rather than reconstructing raw inputs. The key property is that the model is trained to capture invariant structure (what is consistent across different views of the same underlying reality) rather than surface-level patterns.

For this experiment, I trained a JEPA-style encoder on fully solved Sudoku boards using masked-view consistency objectives. Different masked views of the same solved board are presented to the encoder, and the objective encourages their latent representations to be consistent. No digit prediction, no explicit constraint supervision. Just: these two views come from the same valid board, their representations should agree.

The result is an encoder that has internalized what a valid Sudoku configuration looks like in latent space, without ever being told the rules explicitly.

---

## Inference via Latent Stability

Inference works as follows. Given a partially filled board, at each step all legal candidate digit placements are enumerated. Each candidate is evaluated by measuring how much it perturbs the latent representation:

$$\Delta_t = \|z_t - z_{t-1}\|_2^2$$

where $z_t = f(x_t)$ is the encoder's representation of the board at step $t$. The move that minimizes $\Delta_t$ is selected, the one that keeps the representation closest to where it already is.

There is no backtracking. There are no explicit constraint checks. The solver never enters an illegal state because it only selects moves that preserve latent stability, and the encoder has learned that stable representations correspond to valid configurations.

The GIF below shows the solver filling a board in real time:

![JEPA Sudoku solver inference]({{ "/assets/images/sudoku_solution.gif" | relative_url }})
*Each digit placement is chosen to minimally perturb the latent representation. The solver never backtracks.*

---

## Results

The results across all evaluated puzzles were clean:

| Metric | Value |
|---|---|
| Maximum constraint violations | 0 |
| Mean latent change (∆t) | ~10^-8 |
| Maximum latent spike | ~10^-7 |
| Backtracking required | No |
| Explicit repair steps | None |

Zero constraint violations throughout. Latent changes bounded between 10^-9 and 10^-7 with no abrupt spikes, indicating smooth evolution through the latent space rather than the sharp corrective jumps you would see in a repair-based system.

---

## Stability vs. Repair: What the Dynamics Look Like

The contrast with TRM is worth spelling out precisely.

A repair-based reasoner like TRM explicitly traverses invalid intermediate states. It proposes a candidate, checks for constraint violations, identifies them, and corrects them. The reasoning trajectory is a sequence of: invalid state, correction, less invalid state, correction, valid state. The corrections are doing the work.

The JEPA-based solver does not do this. Its trajectory is: valid state, valid state, valid state, valid state. Every intermediate state is legal. There are no corrections because there is nothing to correct. Constraint satisfaction is not enforced after the fact. It emerges from staying within the stable region of the latent space the encoder learned.

This is what the paper calls stability-based reasoning: reasoning as proximity to a fixed point in latent space rather than navigation away from errors.

---

## Why Sudoku

Sudoku is a deliberately conservative choice for testing this idea. It has hard, discrete, fully enumerable constraints. Every state is either valid or invalid with no ambiguity. Legal moves are easy to enumerate. This makes the dynamics legible. You can see exactly what the solver is doing at each step and verify that it never violates a constraint.

The harder and more interesting question is whether stability-based reasoning generalizes beyond clean constraint satisfaction. In a domain with softer constraints, continuous state spaces, or irreducible ambiguity, what does "latent stability" even mean? I do not have an answer. But Sudoku is the right place to establish that the mechanism works at all before asking whether it scales.

---

## Limitations

A few honest constraints on what this work actually shows:

**Legal move enumeration does not scale.** At each step, the solver enumerates all legal candidate placements and evaluates each one. For Sudoku this is tractable. For more complex domains it is not.

**The encoder was trained on fully solved boards.** The masked-view consistency objective gives the encoder a strong signal about what valid configurations look like, but it is trained only on complete solutions. Partial boards during inference are out-of-distribution to varying degrees depending on how many cells remain unfilled.

**Solution completeness is not guaranteed.** The stability criterion selects the move that minimally perturbs the representation, but there is no guarantee that following this criterion at every step leads to a complete solution. The solver can get stuck in a low-perturbation region that is not fully solved.

---

## What This Points Toward

The broader claim this experiment supports is modest but concrete: constraint satisfaction can emerge from learned latent structure without explicit constraint enforcement or error repair. The system does not need to know the rules of Sudoku explicitly. It needs to have internalized what valid Sudoku configurations feel like in representation space.

This is close in spirit to Karl Friston's free-energy principle: behavior as the minimization of surprise relative to a learned model of the world, applied to discrete symbolic reasoning. Stay close to what the model expects valid reality to look like, and validity emerges.

Whether this scales to harder reasoning problems is the open question. The answer probably depends on how well the latent space can be structured to make "valid" and "invalid" regimes geometrically separable. For Sudoku, that separation is clean. For natural language reasoning or scientific hypothesis generation, it almost certainly is not. But that is a problem for future work.

---

## Resources

| Resource | Link |
|---|---|
| 📄 JEPA (LeCun, 2022) | [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) |
| 📄 TRM (Creswell & Shanahan, 2024) | [arXiv:2402.03654](https://arxiv.org/abs/2402.03654) |
| 📄 Chain-of-Thought (Wei et al., 2022) | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) |
| 📄 Free-Energy Principle (Friston, 2010) | [Nature Reviews Neuroscience](https://www.nature.com/articles/nrn2787) |

---

*Reach out at hellojais@gmail.com or [@hellojais](https://github.com/hellojais) on GitHub if you have thoughts on scaling stability-based reasoning beyond constraint satisfaction.*
