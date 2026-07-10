---
layout: post
title: "From Laptop to Blackwell: Scaling an AV World Model with Omnissa Horizon VDI"
date: 2026-07-09
tags: [machine-learning, autonomous-vehicles, jepa, gpu, vdi, horizon]
---

*This post is Part 2 of a series. Part 1 covers the minDrive-JEPA paper, a
transformer-based latent world model for autonomous vehicles that I trained on a
MacBook and published on arXiv. If you haven't read it, the short version: instead
of predicting raw pixels or coordinates, the model learns to predict in **latent
space** using a JEPA objective. The result is a "surprise score", a real-valued
signal per scenario that tells you how unexpected a driving situation was. Part 1
is at [arXiv:2606.28383](https://arxiv.org/abs/2606.28383). This post is about what
happened when I tried to scale it.*

---

## The Wall

The paper model was trained on nuPlan mini, roughly 1,000 driving scenarios. That
was enough to validate the idea. But autonomous vehicle systems need to handle the
full distribution of real-world driving: dense urban intersections, highway merges,
pedestrian crossings across cities on two continents. The full nuPlan dataset
is 63,000 scenarios. Add Argoverse 2 and you're at **263,000**.

My MacBook M5 Max is a genuinely powerful machine. But it has hard limits for this
kind of work:

- **No FP8 tensor cores.** Apple Silicon uses MPS, not CUDA. FP8, the precision
  that gives Blackwell GPUs 2× the throughput of BF16, doesn't exist on it.
  Training at FP32 on MPS for 200 epochs over 263K scenarios would have taken days,
  not hours. And FP8 isn't just a software flag. It's a capability that lives in
  the silicon. You can't emulate it on a laptop.
- **No FlashAttention.** The memory-efficient attention kernel that makes large
  transformer batches tractable is CUDA-only.
- **Storage.** The raw datasets are hundreds of gigabytes. That's NFS territory,
  not a local SSD.
- **Batch size.** Chinchilla scaling tells you that a 38.7M parameter model needs
  roughly 263K training samples to avoid overfitting. Getting there requires a batch
  size of 512 and 48GB of GPU memory, a configuration that doesn't fit on any
  laptop.

The model I wanted to train was physically impossible on the hardware I had.

---

## Why Horizon VDI Was the Right Answer

I've been engineering Omnissa Horizon VDI deployments for several years. When I hit
this wall, the solution was already in front of me.

Horizon VDI with **NVIDIA AI-Ready vWorkstation (vWS)** lets you provision a
GPU-backed virtual machine on demand, connected to shared NFS storage, running a
full CUDA stack, accessible from any endpoint. Critically, when the job is done you
**release the GPU back to the shared pool**. You don't own dedicated hardware. You
borrow exactly what you need, for exactly as long as you need it.

For this project, that meant:

| Component | Spec |
|---|---|
| GPU | RTX Pro 6000 Blackwell, MIG 2g.48gb slice, 51.2GB vRAM |
| Architecture | SM 12.0 (Blackwell) |
| Driver | 580.126.09 |
| CUDA | 12.8 |
| cuDNN | 9.19.0 |
| PyTorch | 2.11.0+cu128 |
| Storage | 2TB NFS volume (datasets + checkpoints) |

The NVIDIA AI vWS SDK unlocked the full Blackwell stack out of the box: CUTLASS BF16
kernels, FlashAttention, and FP8 tensor cores. These are capabilities tied to SM 12.0
silicon and would require significant infrastructure investment to replicate any other
way.

One important point on hypervisor flexibility: **Horizon is hypervisor-agnostic**.
This deployment ran on vSphere, but the same setup can work identically on Nutanix,
OpenStack, and OpenShift. The developer experience is the same regardless of what
infrastructure is underneath.

For enterprise customers thinking about AI developer productivity: this is exactly the
model. Data scientists and ML engineers get Blackwell-class compute on demand through
their standard VDI solution. IT provisions it, monitors it, and reclaims it. No one
buys a $10,000 workstation that sits idle between experiments.

---

## What the Project Did

### Data Pipeline

The first task was converting raw data into a format the model could consume. nuPlan
stores scenarios as SQLite `.db` files across four cities (Boston, Pittsburgh,
Singapore, mini split). Argoverse 2 uses Parquet. A parallel preprocessing pipeline
converted both formats into normalized tensors of shape `[50, 21, 6]`:

- **50** timesteps (subsampled from raw data)
- **21** agents (ego vehicle + 20 nearest agents)
- **6** features per agent: position (x, y), velocity (vx, vy), heading, agent type

The pipeline ran at **2,238 scenarios per second** on the Blackwell VM, processing
all 263,000 scenarios in under two minutes.

### Model Architecture

The model is a scaled-up version of the minDrive-JEPA paper architecture:

| Parameter | Paper (minDrive-JEPA) | This project |
|---|---|---|
| d_model | 128 | 512 |
| Attention heads | 4 | 8 |
| Encoder layers | 4 | 8 |
| Predictor layers | 2 | 4 |
| Total parameters | ~1.3M | **38.7M** |
| Training data | ~1K scenarios | **263K scenarios** |
| Training precision | FP32 (MPS) | **FP8/BF16 (Blackwell)** |

The model size was chosen using Chinchilla scaling: 38.7M parameters requires roughly
263K training samples to converge without overfitting.

### Training and FP8

Training ran for 200 epochs at batch size 512.

Here is where FP8 matters in practice:

> *FP8 tensor cores on Blackwell perform roughly 2× more floating-point operations
> per second than BF16, and use half the memory per weight. That's not a software
> optimization. It's physics.*

This is why the model trained in **5 hours and 42 minutes** at **11.5 TFLOPS**
sustained throughput, using CUTLASS BF16 kernels and FlashAttention. The equivalent
FP32 run on the same hardware would have taken over a day. On an M5 Max it wouldn't
have been feasible at all.

The end-user implication: FP8 let us train a model that would have taken 2 days on
FP32 in under 6 hours, not by changing the algorithm, just by using the right
hardware precision.

### Cross-Dataset Transfer Study

Once training was done, I ran a controlled experiment: train on nuPlan only (63K
scenarios, one geography), then evaluate on held-out Argoverse 2 validation scenarios
(cities the model had never seen). Then repeat with a combined nuPlan + AV2 model
trained on the same 63K scenario budget, so the only variable that changes is
geographic diversity, not data volume.

| Model | Training data | Mean surprise (AV2 val) |
|---|---|---|
| nuPlan-only | 63K, one geography | 0.273 |
| Combined (nuPlan + AV2) | 63K, two geographies | **0.228** |

At identical data scale, combined training reduced mean surprise by **16.5%** (mean of
three seeds). Training on geographic diversity makes the model a better predictor of
driving behaviour it hasn't seen before.

### Demo

The surprise score is interpretable; it's not just a training metric. The animations
below show the model running on real held-out AV2 scenarios. The score bar turns red
when the model is surprised. It spikes at complex intersections and unexpected agent
behaviour, and stays low on routine highway driving.

![Surprise score on a held-out AV2 scenario](/assets/images/demo_final_172538.gif)

![Surprise score on a held-out AV2 scenario](/assets/images/demo_final_57378.gif)

![Surprise score on a held-out AV2 scenario](/assets/images/demo_final_187081.gif)

![Surprise score on a held-out AV2 scenario](/assets/images/demo_final_139387.gif)

![Surprise score on a held-out AV2 scenario](/assets/images/demo_final_57843.gif)

---

## Who This Is For

This project sits at the intersection of three audiences:

**AV researchers** who want to study scene complexity and model generalization across
datasets. The surprise score gives you a cheap, differentiable proxy for "how hard
is this scenario."

**ML platform engineers** who need a concrete, reproducible example of what enterprise
GPU-on-demand looks like end-to-end: data on NFS, model in PyTorch, training on a
vWS-provisioned Blackwell VM, results reproduced by anyone with:
```bash
source local.env
python scripts/train.py --config configs/combined.yaml
```

**Enterprise IT and VDI architects** evaluating whether Horizon + NVIDIA AI vWS can
support serious AI workloads for internal developer teams. This project is evidence
that it can. Not a proof-of-concept toy, but a published research result with
263K training samples and a 38.7M parameter model trained to convergence.

The code is open-source at [github.com/hellojais/av-worldmodel](https://github.com/hellojais/av-worldmodel).  
The paper is at [arXiv:2606.28383](https://arxiv.org/abs/2606.28383).
