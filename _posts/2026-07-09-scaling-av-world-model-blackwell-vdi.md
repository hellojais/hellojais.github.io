---
title: "From Laptop to Blackwell: How Omnissa Horizon Let Me Scale an AV World Model on GPU-on-Demand VDI"
date: 2026-07-09
tags: [machine-learning, autonomous-vehicles, jepa, gpu, vdi]
---

Earlier this year I published minDrive-JEPA on arXiv
([arXiv:2606.28383](https://arxiv.org/abs/2606.28383)), a transformer-based world model
for autonomous vehicles that I designed and trained on a laptop. The idea is simple.
Instead of predicting raw pixels or coordinates, the model predicts in latent space
using a JEPA objective, and the prediction error becomes a "surprise score": one number
per driving scenario that tells you how unexpected the situation was. That paper
validated the idea on about 1,000 scenarios. This post is about what happened when I set
out to scale the architecture up by two orders of magnitude, and the infrastructure
that made it possible.

## The wall

Production AV systems have to handle the full distribution of real driving: dense urban
intersections, highway merges, pedestrians, across many cities. The full nuPlan dataset
is 63,000 scenarios. Add Argoverse 2 and you are at 263,000.

My laptop is a fast machine, but for this kind of work it runs into hard limits. Apple
Silicon uses MPS, not CUDA, so there are no FP8 tensor cores and no FlashAttention. The
raw datasets are hundreds of gigabytes, which is not something you keep on a local SSD.
And to fit a 38.7M-parameter model at batch size 512 you need roughly 48GB of GPU
memory. None of that exists on a laptop. The model I wanted to train just was not
trainable on the hardware I owned.

I did not want to buy a workstation either. A dedicated GPU box sits idle between
experiments, and I only needed the compute for the length of a training run.

## How I actually ran it

I have spent years building virtual desktop infrastructure, so the answer was already
sitting in front of me. I provisioned a GPU-backed virtual workstation on demand using
Omnissa Horizon with NVIDIA AI-ready vWS. A few things about that setup are worth
spelling out, because they are the reason the whole project was practical for one
person.

**The GPU comes as a MIG slice, allocated on the fly.** The physical card is an
RTX Pro 6000 Blackwell. I did not get the whole card. I got a MIG 2g.48gb slice of it,
51.2GB of vRAM, presented to my VM as a PCI device. That slice is carved out of a
shared card and handed to me when I need it. When my run is done I release it, and it
goes back into the pool for the next person. Nobody is sitting on a whole Blackwell card
waiting for me to finish.

**Storage lives on NFS, not on the VM.** The datasets and every checkpoint sit on an
NFS volume that is mounted into the VM. My data does not care which VM I happen to be
running on. The compute is one thing and the storage is another, and they are wired
together at mount time. That separation is the quiet hero of this whole story.

**Checkpoints on NFS make the VM disposable.** Training writes a checkpoint to the NFS
volume at a regular cadence. If a run dies, or the VM goes away for any reason, I do not
lose the training. I bring up compute again, the same NFS volume mounts, and I resume
from the last checkpoint exactly where I left off. The VM is cattle, not a pet. The GPU
slice is something I attach and detach. The only thing I actually care about, the state
of the model, lives on storage that outlives any single VM.

**It is not tied to one hypervisor.** I ran this on Horizon on vSphere. That was my
environment, but nothing in the workflow depends on it. Horizon vWS also runs on
Nutanix, on the hyperscalers like Azure and AWS, and on OpenStack and OpenShift. The
GPU slice, the NFS mount, the resume-from-checkpoint pattern, all of it works the same
way regardless of what is underneath. If you already run one of those, you already have
a place to do this.

Here is the environment I ended up with:

| Component | Spec |
|---|---|
| GPU | RTX Pro 6000 Blackwell, MIG 2g.48gb slice, 51.2 GB vRAM |
| Architecture | SM 12.0 (Blackwell) |
| CUDA / cuDNN | 12.8 / 9.19.0 |
| PyTorch | 2.11.0+cu128 |
| Storage | NFS volume for datasets and checkpoints |

The NVIDIA AI vWS stack gave me the full Blackwell capability set out of the box:
CUTLASS kernels, FlashAttention, and FP8 tensor cores. Those are tied to the SM 12.0
silicon, so getting them any other way would mean buying and racking hardware.

## What the project did

### Data pipeline

nuPlan stores scenarios as SQLite files across four cities. Argoverse 2 uses Parquet. I
wrote a parallel preprocessing pipeline that normalizes both into tensors of shape
`[50, 21, 6]`: 50 timesteps, 21 agents (ego plus the 20 nearest), and 6 features per
agent (position x and y, velocity vx and vy, heading, and agent type). On the Blackwell
VM it ran at about 2,238 scenarios per second and converted all 263,000 scenarios in
under two hours.

### Model

The model is a scaled-up version of the minDrive-JEPA architecture.

| Parameter | Paper (minDrive-JEPA) | This project |
|---|---|---|
| d_model | 128 | 512 |
| Attention heads | 4 | 8 |
| Encoder layers | 4 | 8 |
| Predictor layers | 2 | 4 |
| Total parameters | ~1.3M | 38.7M |
| Training data | ~1K scenarios | 263K scenarios |
| Training precision | FP32 (MPS) | FP8 (Blackwell) |

### Training

Training ran for 200 epochs at batch size 512. FP8 on Blackwell does roughly twice the
floating-point operations per second of BF16 and uses half the memory per weight. That
is a property of the hardware, not a trick in the code. In practice it turned a run that
would have taken more than a day at FP32 into one that finished in under six hours. On
the laptop it would not have run at all.

### The finding: geographic diversity beats data volume

Once the model trained, I ran a controlled transfer study. Train on one data mix, then
evaluate on held-out Argoverse 2 scenarios from cities the model never saw. I ran three
seeds per condition and reported mean and standard deviation.

| Model | Training data | Mean surprise (AV2 val) |
|---|---|---|
| nuPlan-only | 63K, 1 geography | 0.273 +/- 0.008 |
| Combined-63K | 63K, 2 geographies | 0.228 +/- 0.015 |
| AV2-only | 200K, 1 geography | 0.264 |
| Combined-full | 263K, 2 geographies | 0.236 +/- 0.009 |

At the same data scale of 63K scenarios, adding a second geography cut mean surprise by
16.5%. The result I did not expect is the AV2-only row. Training on three times more
data from a single geography still generalized worse than the diverse 63K model. It was
the diversity of the data that mattered, not the volume.

### Demo

The surprise score is something you can watch, not just a training metric. Running the
model on real held-out AV2 scenarios, the score climbs at complex intersections and odd
agent behaviour and stays low on routine highway driving.

![Surprise score on a held-out AV2 scenario]({{ "/assets/images/demo_final_172538.gif" | relative_url }})

![Surprise score on a held-out AV2 scenario]({{ "/assets/images/demo_final_57378.gif" | relative_url }})

![Surprise score on a held-out AV2 scenario]({{ "/assets/images/demo_final_57843.gif" | relative_url }})

![Surprise score on a held-out AV2 scenario]({{ "/assets/images/demo_final_139387.gif" | relative_url }})

![Surprise score on a held-out AV2 scenario]({{ "/assets/images/demo_final_187081.gif" | relative_url }})

## Takeaways

For AV and ML researchers, the surprise score is a cheap, label-free, differentiable
proxy for how hard a scenario is. It is useful for mining interesting scenes and for
studying how well a model generalizes across datasets. The code and configs are
open-source.

For engineering and platform teams, the part I keep coming back to is that the
minDrive-JEPA architecture I published, scaled up to 38.7M parameters and trained to
convergence on 263K scenarios, ran end to end on infrastructure I did not own and gave
back when I was done. Compute came as a MIG slice from a shared card. State lived on
NFS, so any VM would do and a dead run just resumed from its last checkpoint. And
because Horizon vWS runs on vSphere, Nutanix, the hyperscalers, OpenStack and
OpenShift, the same pattern drops into whatever platform a team already runs. That is a
realistic way to give data scientists Blackwell-class compute without buying everyone a
workstation that sits idle between experiments.

If your team wants to build something like this, your Omnissa account team can walk you
through how to stand up a similar Horizon environment with NVIDIA GPUs and put that
setup to work for high-end AI development.

The code is at
[github.com/hellojais/av-worldmodel](https://github.com/hellojais/av-worldmodel), and
the minDrive-JEPA research paper is at
[arXiv:2606.28383](https://arxiv.org/abs/2606.28383).
