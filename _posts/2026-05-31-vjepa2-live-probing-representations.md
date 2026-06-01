---
layout: post
title: "V-JEPA 2 on UCF-101: Backbone Generalizes, SSv2 Head Is the Bottleneck"
date: 2026-05-31
categories: machine-learning world-models representation-learning
description: "I ran V-JEPA 2 frozen on 1,059 UCF-101 videos it was never trained on. A linear probe on the frozen backbone hit 98.2%. The SSv2 classification head fails on 42% of classes. The representations are not the problem."
---

> **Author:** Santosh Jaiswal ([@hellojais](https://github.com/hellojais))  
> **Code:** [hellojais/vjepa2-live](https://github.com/hellojais/vjepa2-live)  
> **Model:** [qubvel-hf/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/qubvel-hf/vjepa2-vitl-fpc16-256-ssv2)  
> **Dataset:** UCF-101 (1,059 clips, 101 action classes)  
> **Hardware:** MacBook Pro M5 Max, MPS backend, no CUDA

---

## TL;DR

V-JEPA 2's frozen backbone achieves 98.2% ± 1.0% linear probe accuracy
on UCF-101 — a dataset it was never trained on. The SSv2 classification
head, by contrast, produces low-confidence or semantically mismatched
predictions for 42% of UCF-101 classes. All four "blind spot" classes
that the SSv2 head fails on completely (PoleVault, SalsaSpin,
WritingOnBoard, Surfing) achieve 100% linear probe accuracy on the same
frozen embeddings. The backbone is not the bottleneck. The classification
head's fixed 174-word SSv2 vocabulary is.

---

## Motivation

V-JEPA 2 is pre-trained self-supervised — it learns by predicting
missing spatio-temporal patches in embedding space, with no labels.
A classification head is then fine-tuned on Something-Something v2
(SSv2), a dataset of 174 hand-object interaction categories.

The pre-training and the fine-tuning are separate stages with
separate objectives. This raises a practical question for anyone
considering adapting V-JEPA 2 to a new domain: when the model
underperforms on out-of-distribution video, is the problem in the
backbone representations or in the classification head's limited
vocabulary? The answer determines whether you need to fine-tune the
backbone (expensive, requires data) or simply replace the head
(cheap, requires very little data).

This study answers that question systematically on UCF-101.

---

## Setup

**Model:** qubvel-hf/vjepa2-vitl-fpc16-256-ssv2 — ViT-L backbone,
325M parameters, SSv2 classification head (174 classes). Used frozen
throughout. No fine-tuning at any stage.

**Dataset:** UCF-101, 101 action categories, ~10 videos per class,
1,059 clips total. UCF-101 has no overlap with SSv2's label space.

**Hardware:** MacBook Pro M5 Max, 68GB unified memory, PyTorch 2.12
MPS backend. All phases run locally, no cloud, no CUDA.

**Pipeline:** Four phases, each building on the previous:

| Phase | What it does | Output |
|-------|-------------|--------|
| 1 | Live webcam inference, MPS validation | ~10fps end-to-end |
| 2 | Zero-shot inference on 1,059 UCF-101 videos | Per-class behavior classification |
| 3 | Backbone embedding extraction + 4 hypothesis tests | (1059, 1024) float32 embeddings |
| 4 | Linear probe on frozen embeddings | Per-class accuracy across 101 classes |

---

## Phase 2: Zero-Shot Behavior on UCF-101

Running the frozen model on all 1,059 UCF-101 videos and classifying
each UCF class by its prediction pattern produced three behavior
categories:

| Behavior | Classes | Description |
|----------|---------|-------------|
| HIGH_CONF_CONSISTENT | 34 | Confidently maps to one SSv2 concept |
| HIGH_CONF_INCONSISTENT | 25 | High confidence, inconsistent SSv2 labels |
| LOW_CONF_SCATTERED | 42 | Low confidence, no dominant SSv2 label |

Three SSv2 physics primitives dominated predictions across all 101
classes: "Hitting" (154 videos), "Throwing" (133), "Spinning" (98).
The SSv2 head's response to out-of-distribution video collapses
to a small set of motion archetypes.

The HIGH_CONF_CONSISTENT mappings are the most informative. The
model is confident and consistent — but the SSv2 labels it assigns
reveal what it actually encoded:

| UCF Class | SSv2 Prediction | Confidence |
|-----------|----------------|------------|
| PlayingDhol | "Hitting [something] with [something]" | 0.834 |
| HulaHoop | "Spinning [something] so it continues spinning" | 0.808 |
| WalkingWithDog | "Moving away from [something] with your camera" | 0.718 |
| CuttingInKitchen | "Spreading [something] onto [something]" | 0.695 |
| CleanAndJerk | "Lifting [something] up completely" | 0.616 |

WalkingWithDog → "Moving away from something with your camera" is
a clean example of JEPA encoding working as intended. The model
has no concept of "dog" as a subject. It encodes the camera-relative
motion pattern — scene receding as you walk forward — and maps it
to the nearest SSv2 spatial-motion concept. Object identity is absent.
Relational motion is present.

The four worst-performing classes share a structural property: they
all involve whole-body or tool-mediated motion with no clear
hand-object interaction. SSv2 is entirely tabletop hand-object
interactions. These classes fall outside its learned physics
vocabulary entirely.

---

## Phase 3: Embedding Analysis

### Baseline: Same-Class vs Different-Class Similarity

Extracting the frozen ViT-L backbone embeddings (before the
classification head) for all 1,059 videos gives a (1059, 1024)
float32 array. The first diagnostic:

| Similarity type | Cosine similarity |
|-----------------|------------------|
| Same UCF class | 0.93 |
| Different UCF class | 0.59 |

A 0.34 gap on action classes the backbone was never trained to
distinguish. The backbone is organizing UCF-101 videos by action
category without any UCF supervision signal.

### Embedding Space Structure

t-SNE and UMAP on the 1024-dim embeddings show behavior categories
occupying distinct regions of embedding space. 94 unique SSv2 labels
were predicted across 101 UCF classes, confirming the backbone
separates most UCF classes into distinct embedding regions.

![Behavior clusters in embedding space]({{ "/assets/images/vjepa2_behavior_clusters.png" | relative_url }})
*t-SNE and UMAP colored by behavior category. GREEN = HIGH_CONF_CONSISTENT, YELLOW = HIGH_CONF_INCONSISTENT, RED = LOW_CONF_SCATTERED. The three categories occupy partially separable regions, consistent with the 0.93 same-class cosine similarity baseline.*

The SSv2 label regions plot shows which physics primitives dominate
which areas of embedding space:

![SSv2 label regions]({{ "/assets/images/vjepa2_ssv2_label_regions.png" | relative_url }})
*Top 15 SSv2 predicted labels overlaid on UMAP. "Hitting", "Throwing", and "Spinning" dominate large contiguous regions. Gray points are classes predicted as less frequent SSv2 labels.*

### The Blind Spot Cluster

The four lowest-confidence UCF classes cluster together in embedding
space rather than scattering randomly. They form a coherent region
with no close SSv2 centroid neighbors — a structurally isolated
"whole-body motion without hand-object interaction" region:

| Class | SSv2 Mapping | Structural reason |
|-------|-------------|------------------|
| PoleVault | "Turning upside down" | Body inversion misread as object inversion |
| SalsaSpin | "Spinning something" | Rotation with no hand-object contact |
| Surfing | "Wiping something off" | Lateral body sweep misread as wiping |
| WritingOnBoard | "Sprinkling something" | Arm arc misread as dispersal |

![Blind spot neighborhood analysis]({{ "/assets/images/vjepa2_blind_spot_analysis.png" | relative_url }})
*Each blind-spot class shown in red with its 20 nearest embedding neighbors in blue. Lines connect each class centroid to its nearest SSv2 concept centroids (stars). All four blind-spot classes cluster together and share nearest neighbors — they form a coherent isolated region, not random noise.*

The model has no "unknown" output. It forces every input onto the
nearest SSv2 physics primitive regardless of fit quality.

### "Hitting" Is a Spurious Catch-All

"Hitting [something] with [something]" absorbed 33 different UCF
classes. Testing whether this is a genuine physics cluster:

| Metric | Value |
|--------|-------|
| Within-group cosine similarity | 0.61 |
| Global same-class baseline | 0.93 |

The 0.32 gap below baseline confirms spurious absorption. The label
functions as a miscellaneous bin for any repetitive contact motion.
TableTennisShot, ApplyEyeMakeup, and Hammering share it not because
their underlying physics is similar but because none fit cleanly into
any other SSv2 category.

![Hitting cluster analysis]({{ "/assets/images/vjepa2_hitting_cluster.png" | relative_url }})
*Left: UMAP showing all 33 UCF classes predicted as "Hitting", colored by UCF class. The spread confirms they are not a tight geometric cluster. Right: 8×8 pairwise cosine similarity heatmap for the top 8 UCF classes in the hitting group. TableTennisShot and Hammering are geometrically distant despite sharing the same SSv2 label.*

### Hypothesis Testing: What Drives Behavior Categories?

Three hypotheses tested using embedding-derived proxies:

| Hypothesis | Proxy | Pearson r | p-value | Verdict |
|------------|-------|-----------|---------|---------|
| A: Temporal dynamics | Intra-class embedding variance | -0.20 | 0.043 | PARTIALLY SUPPORTED |
| B: Spatial scale | Distance from SSv2 distribution center | +0.02 | 0.87 | REJECTED |
| C: Cluster tightness | Within-class cosine similarity | +0.15 | 0.13 | REJECTED |

Hypothesis A is the strongest signal: periodic, repeatable actions
embed more consistently across video instances and tend to receive
higher confidence predictions. But r²≈0.04 means it explains only
4% of variance. Hypotheses B and C are not supported.

The geometry of where a class sits in embedding space, and how
tightly it clusters, does not reliably predict whether the SSv2
head will be confident. Something more nuanced — likely the degree
of overlap between the action's motion vocabulary and SSv2's physics
primitive vocabulary — is the actual driver, but that is not
directly measurable from frozen embeddings alone.

### The Tight Cluster Paradox

21 LOW_CONF_SCATTERED classes have above-median within-class cosine
similarity. Their embeddings are geometrically tight — the backbone
encodes them consistently — yet the SSv2 head produces low-confidence
scattered predictions. These 21 classes are the clearest pre-linear-probe
evidence that the backbone has encoded structure the SSv2 head lacks
vocabulary to express.

---

## Phase 4: Linear Probe

A logistic regression classifier (L2 regularised, `lbfgs` solver,
stratified 5-fold cross-validation, StandardScaler fit on train folds
only) trained on the frozen (1059, 1024) embeddings with UCF-101 labels.

**Overall accuracy: 98.2% ± 1.0%**

| Behavior category | Linear probe accuracy |
|------------------|----------------------|
| HIGH_CONF_CONSISTENT | ~99% |
| HIGH_CONF_INCONSISTENT | ~98% |
| LOW_CONF_SCATTERED | ~97% |

All four blind-spot classes:

| Class | SSv2 confidence | Linear probe accuracy | Delta |
|-------|----------------|----------------------|-------|
| PoleVault | 0.168 | 100% | +83.2% |
| WritingOnBoard | 0.189 | 100% | +81.1% |
| SalsaSpin | 0.194 | 100% | +80.6% |
| Surfing | ~0.20 | 100% | ~+80% |

The tight cluster paradox resolution:

| Metric | Value |
|--------|-------|
| Mean linear probe accuracy (21 scattered classes) | 99.5% |
| Mean SSv2 confidence (same 21 classes) | 30.8% |
| Delta | +68.7% |

There is no UCF-101 class where the backbone fails. The worst
performer is WalkingWithDog at 90% — the only class where the SSv2
head was already somewhat confident (0.718), leaving less room to
improve.

![Linear probe results]({{ "/assets/images/vjepa2_linear_probe.png" | relative_url }})
*Top left: per-behavior accuracy with error bars. Top right: scatter of SSv2 confidence (x) vs linear probe accuracy (y) per class — points above the diagonal have more linearly separable embeddings than the SSv2 head's confidence suggests. Bottom: top 15 most and least improved classes by delta.*

---

## Conclusion

The linear probe result closes the causal question the embedding
analysis raised. The backbone encodes all 101 UCF-101 action categories
at 98.2% linear separability. The SSv2 classification head — a
supervised component with a fixed 174-word vocabulary — cannot express
42% of what the backbone knows.

For practitioners adapting V-JEPA 2 to a new action domain: the
backbone is not the bottleneck. Replacing the SSv2 head with a linear
mapping trained on even a small labeled set from the target domain
is sufficient to recover near-ceiling accuracy. Full backbone
fine-tuning is unlikely to produce meaningful gains over head
replacement alone, at substantially higher cost.

This finding connects to a broader pattern in JEPA-style architectures.
In my earlier [billiards study](https://hellojais.github.io/machine-learning/world-models/research/2026/05/09/what-billiards-reveals-about-jepa.html),
the LeWM backbone encoded ball position at R²=0.988 from pure pixel
prediction, but the downstream planning component was the bottleneck.
The same separation appears here: self-supervised pre-training produces
representations that generalize well beyond the training distribution.
The supervised components built on top of them are the limiting factor.

---

## Limitations

- ~10 videos per class. Results are internally valid for relative
  comparison across classes but not representative of large-sample
  linear probe benchmarks.
- The linear probe uses the same embeddings as the analysis phases.
  There is no fully held-out evaluation set.
- Single model (V-JEPA 2 ViT-L), single dataset (UCF-101). Generalization
  to other architectures or datasets is not established here.

---

## What's Next

1. **Kinetics-400** — does the 98.2% linear probe result hold on a
   larger, more diverse benchmark?
2. **Head replacement vs LoRA** — does fine-tuning the backbone improve
   over a pure linear head swap, or is the backbone already at ceiling?
3. **Optical flow for temporal dynamics** — Hypothesis A was only
   partially supported using embedding variance as a proxy. Direct
   measurement of motion periodicity from optical flow would give a
   cleaner test of whether temporal periodicity drives confidence.
4. **Cross-architecture comparison** — does a CLIP-based video model
   show the same backbone/head dissociation, or is this specific to
   JEPA-style predictive pre-training?
5. **Few-shot head adaptation** — how few labeled examples per class
   does a linear head need to reach 95%+ on a new action domain?

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