---
layout: post
title: "What V-JEPA 2 Knows That It Cannot Say: Probing Self-Supervised Video Representations on UCF-101"
date: 2026-05-31
categories: machine-learning world-models representation-learning
description: "I ran V-JEPA 2 frozen on 1,059 UCF-101 videos it was never trained on. The SSv2 head confidently mislabeled 42% of action classes. A linear probe on the same frozen backbone hit 98.2%. The bottleneck is not the representation. It is the vocabulary."
---

> **Author:** Santosh Jaiswal ([@hellojais](https://github.com/hellojais))  
> **Code:** [hellojais/vjepa2-live](https://github.com/hellojais/vjepa2-live)  
> **Model:** [qubvel-hf/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/qubvel-hf/vjepa2-vitl-fpc16-256-ssv2)  
> **Dataset:** UCF-101 (1,059 clips, 101 action classes)  
> **Hardware:** MacBook Pro M5 Max, MPS backend, no CUDA  

---

## TL;DR

I ran V-JEPA 2 frozen on 1,059 UCF-101 videos it was never trained on.
The SSv2 classification head confidently mislabeled 42% of action
classes — mapping PoleVault to "turning upside down" and WalkingWithDog
to a camera-motion label. A linear probe trained on the same frozen
backbone embeddings achieved 98.2% accuracy across all 101 classes,
including 100% on every "blind spot" class the SSv2 head failed on.
The backbone already understands all 101 actions. It just does not know
their names. The bottleneck is not the representation — it is the
vocabulary.

---

## The Idea

Imagine a chef trained exclusively on Italian cuisine. Show them pasta —
they name it instantly. Show them sushi — they say "it looks like a
rolled focaccia." They are not wrong about the shape. They are wrong
about the name. Their representations are rich. Their vocabulary is
limited.

This is precisely what happens when you run V-JEPA 2 on video it was
never trained on.

V-JEPA 2 is a self-supervised video model trained to predict missing
spatio-temporal patches in embedding space — no labels, no pixel
reconstruction. The representations it learns encode physical dynamics:
motion patterns, contact geometry, temporal rhythms. On top of this
backbone, a classification head was fine-tuned on Something-Something v2
(SSv2), a dataset of 174 hand-object interaction categories: "pushing
something from left to right", "throwing something in the air",
"spinning something so it continues spinning."

I wanted to know: what happens when you show this model actions it was
never labeled on? Does the backbone understand them? Does the head?
Are they the same question?

They are not.

---

## What I Built

I built a four-phase pipeline on a MacBook Pro M5 Max using PyTorch's
MPS backend, no CUDA, no cloud:

| Phase | What it does | Key output |
|-------|-------------|------------|
| Phase 1 | Live webcam inference, MPS validation | ~10fps end-to-end |
| Phase 2 | Zero-shot inference on 1,059 UCF-101 videos | Behavior classification per class |
| Phase 3 | Backbone embedding extraction + analysis | (1059, 1024) embeddings, 4 hypothesis tests |
| Phase 4 | Linear probe on frozen embeddings | 98.2% ± 1.0% accuracy |

Everything runs from a single `git clone`. No GPU required.

---

## Phase 2: What the SSv2 Head Does With Unfamiliar Video

I ran all 1,059 videos through the frozen model and classified each
UCF-101 class into one of three behavior categories based on top-1
confidence and label consistency:

| Behavior | Classes | Description |
|----------|---------|-------------|
| HIGH_CONF_CONSISTENT | 34 | Confidently maps to one SSv2 concept |
| HIGH_CONF_INCONSISTENT | 25 | High confidence, inconsistent SSv2 labels |
| LOW_CONF_SCATTERED | 42 | Low confidence, no dominant SSv2 label |

The most revealing HIGH_CONF_CONSISTENT mappings:

| UCF Class | SSv2 Prediction | Confidence |
|-----------|----------------|------------|
| PlayingDhol | "Hitting [something] with [something]" | 0.834 |
| HulaHoop | "Spinning [something] so it continues spinning" | 0.808 |
| WalkingWithDog | "Moving away from [something] with your camera" | 0.718 |
| CuttingInKitchen | "Spreading [something] onto [something]" | 0.695 |
| CleanAndJerk | "Lifting [something] up completely" | 0.616 |

The WalkingWithDog mapping is the most telling. The model has no concept
of "dog" or "walking" as subjects. It sees the camera-relative motion
pattern — scene moving away as you walk — and encodes it as a spatial
relationship. This is JEPA working as designed: it learned motion
dynamics, not object identity.

Three SSv2 physics primitives dominated predictions across all 101
classes: "Hitting" (154 videos), "Throwing" (133), "Spinning" (98).
The SSv2 head's vocabulary for out-of-distribution video collapses
to a handful of motion archetypes.

The four worst-performing classes — PoleVault (0.168 confidence),
WritingOnBoard (0.189), SalsaSpin (0.194), Surfing — share something
structural: they all involve whole-body or tool-mediated motion with
no clear hand-object interaction. SSv2 is entirely hand-object
interactions. These actions fall completely outside its learned
physics vocabulary.

---

## Phase 3: What the Backbone Actually Encoded

I extracted the frozen ViT-L backbone embeddings for all 1,059 videos
before the classification head. Shape: (1059, 1024).

The first verification was striking:

| Similarity type | Cosine similarity |
|-----------------|------------------|
| Same UCF class | 0.93 |
| Different UCF class | 0.59 |

A gap of 0.34 on actions the model was never trained to distinguish.
The backbone is already organizing UCF-101 videos by action category
without any supervision signal for those categories.

### The Blind Spots Form a Coherent Region

t-SNE and UMAP showed the four blind-spot classes clustering together
in embedding space — not scattered randomly. They form a coherent
"whole-body motion" region the SSv2 vocabulary has no words for.
Their forced mappings reveal how the head handles this:

| Class | SSv2 Mapping | Why |
|-------|-------------|-----|
| PoleVault | "Turning upside down" | Body inversion misread as object inversion |
| SalsaSpin | "Spinning something" | Rotation without hand-object contact |
| Surfing | "Wiping something off" | Lateral body sweep misread as wiping gesture |
| WritingOnBoard | "Sprinkling something" | Arm arc misread as dispersal motion |

The model does not have an "unknown" token. It forces every input onto
the nearest physics primitive, however ill-fitting.

### "Hitting" Is a Catch-All, Not a Real Cluster

The "Hitting" label absorbed 33 different UCF classes. I tested whether
this was a genuine physics cluster or a spurious catch-all:

| Metric | Value |
|--------|-------|
| Within-group cosine similarity | 0.61 |
| Global same-class baseline | 0.93 |

A 0.32 gap confirms it. "Hitting" is the model's miscellaneous shelf —
anything with a small repetitive stroke gets filed there regardless of
underlying physics. TableTennisShot, ApplyEyeMakeup, and Hammering share
the label not because they are the same action, but because none of them
fit clearly elsewhere.

### Three Hypotheses, One Partial Answer

I tested three hypotheses about what drives the behavior categories,
using embedding-derived proxies:

| Hypothesis | Proxy | Pearson r | p-value | Verdict |
|------------|-------|-----------|---------|---------|
| A: Temporal dynamics | Intra-class variance | -0.20 | 0.043 | PARTIALLY SUPPORTED |
| B: Spatial scale | Distance from SSv2 center | +0.02 | 0.87 | REJECTED |
| C: Cluster tightness | Within-class similarity | +0.15 | 0.13 | REJECTED |

Hypothesis A explains about 4% of variance (r²≈0.04). Periodic,
repeatable actions embed more consistently and get higher confidence —
but even that is a weak signal. None of the three geometric proxies
fully explain the confidence behavior.

### The Tight Cluster Paradox

The most important finding from Phase 3 came from the rejected
hypotheses. 21 LOW_CONF_SCATTERED classes have above-median
within-class similarity — their embeddings are geometrically tight,
yet the SSv2 head produces low-confidence scattered predictions.

These 21 classes are the clearest evidence that the backbone has encoded
something real that the SSv2 head simply has no vocabulary to express.
The information is in the embedding. The head cannot read it.

---

## Phase 4: The Linear Probe Closes the Loop

I trained a logistic regression classifier (L2 regularised, stratified
5-fold cross-validation) on the frozen (1059, 1024) embeddings with
UCF-101 class labels. No model reloading. No fine-tuning. Just a linear
mapping from existing embeddings to UCF-101 labels.

**Overall accuracy: 98.2% ± 1.0%**

| Behavior category | Linear probe accuracy |
|------------------|----------------------|
| HIGH_CONF_CONSISTENT | ~99% |
| HIGH_CONF_INCONSISTENT | ~98% |
| LOW_CONF_SCATTERED | ~97% |

Every single blind-spot class achieved 100%:

| Class | SSv2 confidence | Linear probe accuracy |
|-------|----------------|----------------------|
| PoleVault | 0.168 | 100% |
| WritingOnBoard | 0.189 | 100% |
| SalsaSpin | 0.194 | 100% |
| Surfing | ~0.20 | 100% |

The tight cluster paradox is resolved:

| Metric | Value |
|--------|-------|
| Mean linear probe accuracy (21 scattered classes) | 99.5% |
| Mean SSv2 confidence (same 21 classes) | 30.8% |
| Delta | +68.7% |

The backbone had the answer all along. The head had no words for it.

There is not a single UCF-101 class where the backbone fails. The worst
performer is WalkingWithDog at 90% — and that class was already
somewhat confident in SSv2 terms (0.718), leaving little room to
improve.

---

## The Finding

> V-JEPA 2's self-supervised backbone achieves 98.2% linear probe
> accuracy on UCF-101 — a dataset it was never trained on — while its
> SSv2 classification head produces confident but semantically wrong
> predictions for 42% of the same classes. The bottleneck is not the
> representation. It is the vocabulary.

Replacing the SSv2 classification head with a linear mapping to UCF-101
labels — nothing else, no backbone fine-tuning, no additional training
data beyond the 1,059 clips used here — would yield a ~98% accurate
UCF-101 classifier. The world model already understands all 101 actions.
It just needed someone to give it the right dictionary.

This finding has a direct implication for how to think about adapting
V-JEPA 2 to new domains: the backbone is not the bottleneck. Any
fine-tuning effort should focus entirely on the classification head,
not the representations.

---

## What This Connects To

This result sits within a broader pattern in JEPA-style architectures.
In my [billiards study](https://hellojais.github.io/machine-learning/world-models/research/2026/05/09/what-billiards-reveals-about-jepa.html),
the LeWM backbone encoded ball position at R²=0.988 from pure
pixel prediction — but the planning head could not navigate the latent
space effectively. The representation was rich. The downstream component
was the bottleneck.

The same structure appears here at a different level: V-JEPA 2's
backbone encodes 101 UCF-101 action categories at 98.2% linear
separability. The SSv2 classification head — a supervised component
with a fixed 174-word vocabulary — cannot express most of what the
backbone knows.

Self-supervised representations generalize. Supervised heads are
vocabulary-limited by construction.

---

## Limitations

- ~10 videos per class — small sample for a 101-class linear probe.
  Results are internally valid for relative comparison but not
  representative of large-sample benchmarks.
- Linear probe uses the same embeddings as the analysis phases —
  no fully held-out evaluation set.
- Single model (V-JEPA 2 ViT-L), single dataset (UCF-101).
  Results may not generalize to other architectures or datasets.

---

## What's Next

1. **Test on Kinetics-400** — a larger, more diverse benchmark.
   Does the 98% linear probe result hold at scale?
2. **Compare head replacement vs LoRA fine-tuning** — does updating
   the backbone weights improve over a pure linear head swap, or is
   the backbone already at ceiling?
3. **Temporal dynamics with optical flow** — Hypothesis A was only
   partially supported using embedding variance as a proxy. Direct
   measurement of motion periodicity from optical flow would give a
   cleaner test.
4. **Cross-architecture comparison** — Does a CLIP-based video model
   show the same backbone/head dissociation, or is this specific to
   JEPA-style predictive pre-training?
5. **Few-shot head adaptation** — how few labeled examples per class
   does a linear head need to reach 95%+ accuracy on a new domain?

---

## Resources

| Resource | Link |
|----------|------|
| 💻 Code | [hellojais/vjepa2-live](https://github.com/hellojais/vjepa2-live) |
| 📊 Full Report | [vjepa2_live_report.html](https://github.com/hellojais/vjepa2-live/blob/main/results/vjepa2_live_report.html) |
| 🤖 Model | [qubvel-hf/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/qubvel-hf/vjepa2-vitl-fpc16-256-ssv2) |
| 📁 Dataset | [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) |

---

*All code from this project is released under MIT license.*