---
layout: post
title: "Seeing Quantization in Action: Training ResNet-18 at FP32, FP16, FP8, and FP4"
date: 2025-10-06 00:00:00 +0000
categories: machine-learning quantization training
author: Santosh Jaiswal
description: I trained ResNet-18 on a chest X-ray dataset across four precision levels on a MacBook Pro M2 Max and a rented NVIDIA B200. FP4 training completed in 12 minutes with 83-84% accuracy. FP32 took 50 minutes at 87%. Here is what the precision journey looks like end to end.
---

> **Author:** Santosh Jaiswal ([@hellojais](https://github.com/hellojais))\
> **Dataset:** [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)\
> **Hardware:** MacBook Pro M2 Max (FP32/FP16) + NVIDIA B200 on Northflank (FP8/FP4)

---

## TL;DR

I trained ResNet-18 on a chest X-ray pneumonia dataset across four precision levels: FP32, FP16, FP8, and FP4. FP32 on a MacBook Pro M2 Max took 50 minutes at 87% accuracy. FP4 on a rented NVIDIA B200 took 12 minutes at 83-84%. The B200's native NVFP4 support and CUDA tensor cores made the lower precisions practical. Mixed precision (FP4 forward pass, FP8/FP16 gradients) was the key to keeping training stable at ultra-low precision. FP4 training is not the future. It is already here.

---

## The Starting Point

It started with a paper on ultra-low precision training. The idea of training deep neural networks in FP4 felt almost unreal: four bits to represent a weight, an activation, a gradient. The dynamic range is tiny. The rounding errors should compound catastrophically. And yet the paper showed it working.

So I decided to try it myself.

I picked ResNet-18 and the Chest X-Ray Pneumonia dataset from Kaggle: roughly 5,000 images, binary classification (normal vs pneumonia). A well-understood architecture on a well-understood task, clean enough that precision-related degradation would be visible rather than hidden in model complexity.

The first obstacle: my MacBook Pro M2 Max could only take me so far. FP32 and FP16 ran fine. FP8 and FP4 require hardware with native low-precision tensor core support, which the M2 does not have. So I rented an NVIDIA B200 on Northflank for 8 hours at approximately $20 and ran the lower precision experiments there.

---

## Why Precision Matters

Neural network training is dominated by matrix multiplications. Roughly 90% of the compute in a forward and backward pass is MATMUL operations. The precision of those multiplications determines three things: memory footprint, arithmetic throughput, and numerical stability.

Lower precision means smaller numbers: FP32 uses 32 bits per value, FP16 uses 16, FP8 uses 8, FP4 uses 4. Halving the bit width roughly doubles the number of values you can fit in a given memory budget, and modern tensor cores can execute lower-precision MATMULs significantly faster than higher-precision ones.

The cost is numerical range and resolution. FP32 can represent values from roughly 10^-38 to 10^38 with about 7 decimal digits of precision. FP4 has a tiny representable range and can only distinguish 16 distinct values. Training a neural network in FP4 naively would cause gradients to vanish or explode almost immediately.

The solution is mixed precision: use low precision where it is safe (the forward pass, where rounding errors are bounded by the loss function) and higher precision where it is not (gradients and optimizer states, where accumulated rounding errors can destabilize training).

---

## The Precision Journey

| Configuration | Hardware | Training time (5 epochs) | Accuracy |
|---|---|---|---|
| FP32 | MacBook Pro M2 Max | ~50 minutes | ~87% |
| FP16 | MacBook Pro M2 Max | ~27 minutes | ~86.5% |
| FP8 | NVIDIA B200 (Northflank) | ~18 minutes | ~85% |
| FP4 | NVIDIA B200 (Northflank) | ~12 minutes | ~83-84% |

A few things stand out in this table.

**FP16 is nearly free.** Going from FP32 to FP16 on the same hardware cuts training time almost in half (50 min to 27 min) while losing less than half a percentage point of accuracy. For any training run that does not require FP32 for numerical reasons, FP16 should be the default.

**FP8 on a B200 is faster than FP16 on an M2 Max.** This is partly a hardware comparison rather than a pure precision comparison, but it illustrates the combined effect of lower precision and better tensor core support. 18 minutes at 85% accuracy is a strong result.

**FP4 is surprisingly viable.** 12 minutes at 83-84% accuracy is a 4x speedup over FP32 with a 3-4 percentage point accuracy drop. On a medical imaging task that is not a trivial tradeoff, but for many applications it is entirely acceptable. The key enabler is the B200's native NVFP4 support.

---

## Mixed Precision Strategy

Training in FP4 is not as simple as setting a flag. The forward pass can run in FP4 safely: activations flow through the network, rounding errors accumulate, but the loss function provides a corrective signal that keeps the model on track.

The backward pass is a different story. Gradients are small, can be negative, and need to be accumulated across many operations. In FP4, the representable range is so narrow that most gradient values would round to zero or overflow. The optimizer state (momentum, variance in Adam) needs even more precision to track slow-moving statistics accurately.

The strategy I used:

- **Forward pass:** FP4, using the B200's native NVFP4 tensor cores
- **Gradients:** FP8/FP16, preserving enough range for stable backpropagation
- **Optimizer states:** FP16, maintaining numerical stability in Adam's momentum and variance terms

This is the standard mixed precision pattern, extended down to FP4 for the forward pass. CUDA's tensor cores handle the precision transitions efficiently: the B200 can execute FP4 MATMULs natively and accumulate results in higher precision before writing them back.

---

## What the B200 Makes Possible

The NVIDIA B200 is the first consumer-accessible GPU with native NVFP4 support. On earlier hardware, FP4 training required software emulation: representing FP4 values in higher-precision containers and doing the arithmetic at higher precision. That eliminates most of the speed benefit.

On the B200, FP4 MATMULs run natively on dedicated tensor cores. The hardware handles the quantization, accumulation, and dequantization in silicon. The result is that the theoretical speedup of lower precision translates into actual wall-clock speedup rather than being consumed by emulation overhead.

At $20 for 8 hours on Northflank, running the FP8 and FP4 experiments cost less than a lunch. That accessibility matters: researchers and practitioners who cannot justify large cloud GPU budgets can now experiment with ultra-low precision training on real hardware.

---

## Accuracy vs Speed Tradeoff

The four data points trace a clear curve: as precision decreases, training time drops and accuracy drops with it.

| Precision | Speedup vs FP32 | Accuracy drop vs FP32 |
|---|---|---|
| FP16 | 1.85x | 0.5pp |
| FP8 | 2.8x | 2pp |
| FP4 | 4.2x | 3-4pp |

The FP16 point is the most efficient on this curve: nearly 2x speedup for half a percentage point of accuracy loss. FP8 and FP4 trade larger accuracy drops for larger speedups, and whether that tradeoff is acceptable depends entirely on the application.

For pneumonia detection from chest X-rays, a 3-4pp accuracy drop at FP4 is probably not acceptable in a clinical setting. For a research prototype, a data preprocessing pipeline, or a non-critical classification task, it may well be.

The broader point is that this choice now exists. A year ago, FP4 training was a research curiosity. On a B200, it is a practical option with known tradeoffs.

---

## Caveats

**Five epochs is a short run.** I trained for 5 epochs to keep costs manageable and make the precision comparison clean. Final accuracy numbers after full convergence (30-50 epochs) would likely show different relative gaps between precision levels, as lower precision models may benefit more from additional training.

**Single dataset, single architecture.** ResNet-18 on chest X-rays is one data point. The accuracy-precision tradeoffs on larger models, more complex datasets, or different tasks could look quite different. Transformer architectures in particular may respond differently to ultra-low precision training due to their attention mechanisms.

**The M2 vs B200 comparison is not apples to apples.** The FP32/FP16 numbers come from an M2 Max and the FP8/FP4 numbers come from a B200. The B200 would be faster at FP32 and FP16 too. The table shows what I actually ran, not a controlled hardware comparison.

---

## What This Points Toward

Mixed precision training with FP4 forward passes and FP8/FP16 gradients and optimizer states is not just feasible. It is practical on available hardware at accessible cost.

The implication for anyone training models: precision is now a design choice rather than a fixed constraint. FP32 is no longer the default you fall back to when you are unsure. FP16 should be the baseline for most training runs. FP8 is viable for throughput-sensitive applications where a small accuracy drop is acceptable. FP4 is available for cases where speed matters more than the last few percentage points.

CUDA's tensor cores and native hardware support on the B200 are what make this practical. Roughly 90% of neural network training is MATMUL, and those MATMULs can now run at 4-bit precision on real hardware. That changes the cost structure of training in ways that will compound as the hardware becomes more widely available.

FP4 training is not the future. It is already here.

---

## Resources

| Resource | Link |
|---|---|
| Dataset | [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Cloud GPU | [Northflank](https://northflank.com) |
| Paper | Ultra-low precision training (arXiv link to be added) |

---

*Reach out at hellojais@gmail.com or [@hellojais](https://github.com/hellojais) on GitHub if you have run quantization experiments across precision levels.*
