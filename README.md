# Personalized Reward Features from Preference Feedback

### Few-Shot Adaptation and Downstream Offline Reinforcement Learning in Meta-World

## Overview

This project investigates personalized reward modeling from pairwise preference feedback. It explores whether a shared reward representation can be learned across multiple users and adapted to individual preferences using only a small number of preference queries.

Rather than training an independent reward model for every user, the proposed approach learns a shared temporal state-action encoder that produces low-dimensional reward features. Each user is represented by a lightweight preference-weight vector, allowing personalization without retraining the shared encoder.

The framework is evaluated using synthetic users in two Meta-World environments:

- Reach-v3
- Pick-Place-v3

The project covers shared reward representation learning, few-shot personalization, architecture and feature-dimension ablations, preference acquisition strategies, reward recovery, and downstream offline reinforcement learning.

## Research Objectives

The project investigates five central questions:

1. Can shared reward features support few-shot personalization across users with different preferences?
2. How does the dimensionality of the learned reward representation affect adaptation?
3. How important are temporal modeling and encoder architecture?
4. Can active preference querying improve adaptation efficiency over random querying?
5. Do improvements in preference prediction translate into better downstream reinforcement-learning policies?

## Methodology

### 1. Personalized Reward Representation

The model learns a shared trajectory representation:

\[
\phi(\tau)\in\mathbb{R}^{K}
\]

where \(\tau\) represents a trajectory segment and \(K\) is the reward-feature dimension.

A personalized reward is computed as:

\[
R_u(\tau)=w_u^\top\phi(\tau)
\]

where:
- \(\phi(\tau)\) is the shared learned representation.
- \(w_u\) is the preference-weight vector for user \(u\).
- \(R_u(\tau)\) is the predicted personalized reward.

The encoder processes state-action sequences using separate state and action embeddings, followed by feature fusion and temporal modeling.

A per-step feature head produces \(\phi_t\), and segment features are obtained through temporal aggregation:

\[
\phi(\tau)=\sum_t\phi_t
\]

This decomposition allows personalized rewards to be assigned at the individual time-step level for downstream offline RL.

### 2. Shared Training

The shared encoder and preference weights for training users are learned jointly using pairwise preference comparisons and a Bradley-Terry preference loss.

The training setup uses:
- 50 training users.
- Held-out episodes for validation.
- AdamW optimization.
- Learning rate of \(3\times10^{-4}\).
- Batch size of 256.
- Up to 6,000 training steps.
- Validation-based early stopping.

### 3. Few-Shot User Adaptation

For an unseen user, the shared encoder is frozen.

The user-specific preference vector is estimated using ridge-regularized logistic regression over differences between trajectory feature vectors:

\[
\Delta\phi=\phi(\tau_i)-\phi(\tau_j)
\]

Adaptation uses a damped Newton optimization procedure.

Two prior configurations are investigated:

- **Population prior:** Shrinkage toward the mean training-user preference vector.
- **Zero prior:** Shrinkage toward zero.

The default regularization strength is \(l_2=0.1\).

## Experimental Setup

### Environments

| Environment | Description |
|---|---|
| Reach-v3 | Robotic reaching task |
| Pick-Place-v3 | Robotic object manipulation task |

### Synthetic User Preferences

User preferences are generated from six handcrafted behavioral statistics.

Users are divided into three heterogeneity groups:

- Low heterogeneity
- Medium heterogeneity
- High heterogeneity

The high-heterogeneity group represents users whose preferences differ substantially from the training-user population.

Preferences are defined over 25-step trajectory segments, with duration based on time to first success.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Preference accuracy (`acc_true`) | Measures agreement with noiseless ground-truth preferences |
| Reward recovery (Spearman correlation) | Measures correlation between learned and ground-truth rewards |
| Ground-truth policy reward | Evaluates downstream policy quality |
| Learned-versus-true reward gap | Examines exploitation of learned reward-model errors |

## Experiments

### 1. Baseline Comparison

The proposed shared reward-feature model is compared against:

**PrefMMT**
- A pooled preference-model baseline.
- Adaptation through last-layer fine-tuning.
- Full-network fine-tuning is not implemented.

**Trajectory-Statistics MLP**
- Receives the behavioral statistics used to generate preference labels.
- Serves as a privileged, near-oracle reference rather than a directly comparable learned-model baseline.

**Psi-Linear**
- A linear reference operating directly on the ground-truth behavioral statistics.

The comparison evaluates how effectively the learned state-action representation recovers the information required for personalized preference prediction.

### 2. Reward-Feature Dimension Ablation

The effect of reward-feature dimensionality is investigated using:

\[
K\in\{4,8,16,32\}
\]

The experiment evaluates whether a smaller or larger reward representation improves few-shot adaptation.

Key observation:
- \(K=4\) performs substantially worse.
- \(K=8,16,32\) achieve broadly comparable results.
- \(K=16\) performs particularly well for high-heterogeneity users.
- Smaller representations can be beneficial when preference labels are scarce.

These findings are empirical observations for the evaluated task and dataset.

### 3. Temporal Encoder Ablation

The importance of temporal modeling is investigated by comparing:

- Transformer encoder.
- GRU encoder.
- MLP without temporal context.
- PrefMMT-style encoder within the proposed reward-feature framework.

The experiments examine whether the gains come primarily from the shared reward-feature personalization mechanism or from a particular temporal architecture.

The evaluated architectures achieve broadly similar performance, while removing temporal context generally reduces accuracy.

### 4. Personalization Prior and Regularization Ablation

The adaptation mechanism is evaluated using different regularization strengths:

\[
l_2\in\{0.1,1,10,100\}
\]

The experiments compare:
- Population-centered regularization.
- Zero-centered regularization.

The purpose is to investigate the trade-off between adapting to individual preferences and retaining information from the training-user population.

Key observations:
- The two priors perform similarly at \(l_2=0.1\).
- Stronger population priors can benefit low- and medium-heterogeneity users when labels are scarce.
- Strong regularization can hinder adaptation for high-heterogeneity users.
- No single regularization strength dominates across all groups.

### 5. Random vs. Active Preference Querying

The project investigates whether selecting informative preference comparisons improves adaptation efficiency.

Three strategies are evaluated:

**Random Querying**
- Selects preference pairs randomly.

**Ensemble Disagreement**
- Uses disagreement between independently trained reward-model ensemble members.
- Selects pairs with high variance in predicted preference probabilities.

**Predictive Entropy**
- Selects pairs with uncertain mean preference predictions.

The experiments use nested preference budgets and a shared initial set of random queries.

Key observation:

Active querying does not demonstrate a consistent advantage over random querying in the evaluated setup. Ensemble disagreement sometimes performs worse, while entropy is generally comparable to or slightly better than random querying.

The comparison is preliminary because the active strategies use fewer query orders than random querying, and the original ensemble included undertrained checkpoints.

### 6. Reward Recovery Analysis

Preference prediction accuracy alone does not establish whether the learned representation recovers the underlying reward structure.

Reward recovery is therefore evaluated using Spearman rank correlation between:

\[
w_u^\top\phi(\tau)
\]

and

\[
v_u^\top\psi(\tau)
\]

where \(\psi(\tau)\) represents the ground-truth behavioral statistics.

The evaluation considers:
- Population reward without personalization.
- Personalized reward with 5 labels.
- Personalized reward with 20 labels.
- Personalized reward with 100 labels.
- A privileged linear reference using ground-truth features.

The results show that reward recovery improves with additional preference feedback, particularly for high-heterogeneity users.

### 7. Downstream Offline Reinforcement Learning

The project evaluates whether personalized reward models can improve downstream policy learning.

Implicit Q-Learning (IQL) is used to train policies on an offline dataset containing 1,200 collected episodes per task.

The experiment compares reward variants:

| Reward variant | Description |
|---|---|
| Personalized reward | User-specific reward estimated from preference feedback |
| Population reward | Mean training-user reward without additional preference labels |
| Pooled PrefMMT | Reward predicted by the pooled preference baseline |
| Environment reward | Original Meta-World task reward |
| Oracle reward | Ground-truth personalized reward |

IQL uses:
- Expectile parameter: 0.7
- Inverse temperature: 3
- Discount factor: 0.99
- Two hidden layers of 256 units
- 100,000 training steps

Policies are evaluated using ground-truth user rewards over 10 rollout episodes.

**Key findings:**
- Personalized rewards improve policy performance over the population-reward baseline.
- The personalized reward does not establish superiority over pooled PrefMMT.
- Increasing the preference budget does not consistently improve downstream policy quality.
- The oracle reward provides a useful reference, although IQL does not fully optimize even the exact reward in this setup.

### 8. Learned Reward Exploitation Analysis

An additional experiment investigates whether policies exploit errors in the learned reward model.

Policies are evaluated using both:
- The learned reward used during training.
- The ground-truth reward of the corresponding user.

The difference between these values measures the extent to which policies achieve high predicted rewards without achieving correspondingly high true rewards.

The analysis shows that policies trained on learned rewards can obtain substantially higher learned reward values while retaining poor ground-truth performance.

This demonstrates that improved preference prediction does not automatically guarantee reliable downstream policy optimization.

Extrapolation beyond the offline dataset is a plausible explanation, although the experiments do not conclusively establish the cause.

## Main Results

The five-training-seed Reach-v3 Transformer evaluation uses \(K=16\) and the population prior.

| User heterogeneity | B=0 | B=5 | B=20 | B=100 |
|---|---:|---:|---:|---:|
| Low | 0.866 | 0.804 | 0.849 | **0.886** |
| Medium | 0.808 | 0.781 | 0.850 | **0.890** |
| High | 0.513 | 0.689 | 0.803 | **0.855** |

Values are provisional five-seed means; standard deviations are reported in the accompanying technical report.

The four initially undertrained Transformer runs (seeds 1–4) were subsequently retrained for 6,000 steps using separate repair-tagged checkpoints. Their regenerated adaptation results were included in the updated analysis.

## Repository Structure

```text
.
├── configs/
│   └── model.yaml
├── src/
│   ├── models/
│   ├── training/
│   └── ...
├── checkpoints/
├── results/
│   └── tables/
├── tests/
├── scripts/
├── main.py
├── HANDOFF.md
└── README.md
```

### Main Components

| Component | Responsibility |
|---|---|
| `configs/` | Model and experiment configurations |
| `src/models/` | Reward encoder and model implementations |
| `src/training/` | Training, adaptation and evaluation routines |
| `checkpoints/` | Saved model checkpoints |
| `results/tables/` | Generated experiment results |
| `scripts/` | Diagnostic and supporting scripts |
| `tests/` | Project tests |
| `main.py` | Experiment command-line interface |
| `HANDOFF.md` | Detailed project implementation and module documentation |

## Running the Project

### Requirements

- Python
- PyTorch
- Meta-World
- Dependencies specified by the project environment

Install the dependencies using the project's dependency configuration.

### Available Commands

Run commands from the repository root.

```bash
python main.py --help
```

The experiment pipeline is organized into the following stages:

```bash
python main.py collect
python main.py pairs
python main.py users
python main.py train
python main.py stats
python main.py adapt
python main.py iql
python main.py report
```

Additional experiment modules:

```bash
python main.py ablations
python main.py querying
python main.py heterogeneity
python main.py baselines
python main.py audit
```

The available commands and their configuration overrides can be inspected through the CLI help and `configs/`.

### Run Tests

```bash
python tests/run_tests.py
```

## Reproducibility

- Experiments are configuration-driven.
- Training checkpoints are saved under `checkpoints/`.
- Generated result tables are saved under `results/tables/`.
- Training seeds and adaptation/query seeds represent different sources of variability.
- Transformer seeds 1–4 were retrained after an initial checkpoint audit identified that the original runs had completed only 1–4 steps instead of the intended 6,000.
- Repaired checkpoints use a separate `repair` tag to distinguish them from the original checkpoints.
- The active-querying experiment retains a qualification because its original ensemble included undertrained checkpoints.

## Limitations

The project has several important limitations:

1. **Synthetic preferences:** Users are generated from six handcrafted behavioral statistics. The results do not establish personalization of arbitrary real human preferences.
2. **Reward structure:** Ground-truth preferences are linear in the underlying behavioral statistics.
3. **User distribution:** High-heterogeneity users deliberately differ from the training-user population, making adaptation an extrapolation challenge.
4. **Baseline scope:** The pooled PrefMMT baseline uses last-layer fine-tuning rather than full-network fine-tuning.
5. **Privileged reference:** The trajectory-statistics MLP receives the label-generating features and is therefore an upper reference.
6. **Active querying:** Query-order counts are not fully balanced, and the original ensemble included undertrained checkpoints.
7. **Offline RL coverage:** IQL evaluation is limited to Reach-v3, five users per group, one IQL seed and ten evaluation episodes.
8. **Reward exploitation:** The cause of learned-reward exploitation is not conclusively established.
9. **Ablation coverage:** Several architecture and feature-dimension experiments use limited training-seed coverage and are primarily evaluated on Reach-v3.

## References

1. Zhao et al. *PrefMMT: Modeling Human Preferences in Preference-based Reinforcement Learning with Multimodal Transformers.* IROS 2025.
2. Barreto et al. *Capturing Individual Human Preferences with Reward Features.* NeurIPS 2025.
3. Kostrikov, Nair and Levine. *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022.
4. Yu et al. *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning.* CoRL 2019.

## Project Report

Refer to the accompanying technical report for detailed mathematical formulations, experiment configurations, numerical comparisons, ablation tables, reward recovery results, and downstream policy evaluations.
