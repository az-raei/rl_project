# Personalized Reward Features from Preference Feedback

**Few-Shot Adaptation and Downstream Offline Reinforcement Learning in Meta-World**

## Overview

This project investigates learning shared reward representations from preference feedback and adapting them to individual users with limited preference queries.

Instead of training a separate reward model for every user, we learn a shared temporal state-action encoder that produces a low-dimensional reward representation. Individual user preferences are then represented using lightweight user-specific weight vectors.

The framework is evaluated on the **Meta-World Reach-v3 and Pick-Place-v3 tasks**, using synthetic users with different behavioral preferences and levels of preference heterogeneity.

## Methodology

The proposed model learns a shared feature representation:

\[
\phi(\tau)\in\mathbb{R}^{K}
\]

where \(\tau\) denotes a trajectory and \(K\) is the reward feature dimension.

A user-specific reward is calculated as:

\[
R_u(\tau)=w_u^\top\phi(\tau)
\]

where:
- \(\phi(\tau)\) is the shared learned trajectory representation.
- \(w_u\) is the user-specific preference vector.
- \(R_u(\tau)\) is the predicted reward for user \(u\).

The shared encoder is frozen during few-shot user adaptation. Only the user-specific preference parameters are adapted using a limited number of preference comparisons.

### Main model configuration

| Parameter | Configuration |
|---|---|
| Encoder | Transformer |
| Feature dimension (K) | 16 |
| Model dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Maximum sequence length | 128 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Training steps | 6,000 |
| Gradient clipping | 1.0 |

## Experimental Setup

### Environments
- Meta-World Reach-v3
- Meta-World Pick-Place-v3

### User heterogeneity

Synthetic users are generated using six behavioral statistics. Users are grouped into:
- Low heterogeneity
- Medium heterogeneity
- High heterogeneity

### Preference adaptation

The model is evaluated under different preference-feedback budgets:

- B = 0
- B = 5
- B = 20
- B = 100

The primary evaluation metric is `acc_true`, measuring accuracy against the ground-truth preference labels.

### Baselines

The experiments include:
- **PrefMMT:** A pooled preference-model baseline using last-layer fine-tuning.
- **StatsMLP:** A model receiving the behavioral statistics used in preference generation. This serves as a privileged, near-oracle reference rather than a directly comparable learned-model baseline.

### Additional experiments

- Reward representation dimension ablations.
- Temporal encoder ablations.
- Encoder architecture comparisons.
- Active versus random preference querying.
- Downstream offline reinforcement learning using Implicit Q-Learning (IQL).

## Results

The five-training-seed Reach-v3 Transformer evaluation produced the following provisional aggregate results:

| User group | B=0 | B=5 | B=20 | B=100 |
|---|---:|---:|---:|---:|
| Low | 0.866 | 0.804 | 0.849 | **0.886** |
| Medium | 0.808 | 0.781 | 0.850 | **0.890** |
| High | 0.513 | 0.689 | 0.803 | **0.855** |

Values represent mean accuracy across five training seeds. Standard deviations are reported in the accompanying research report.

At B=100, the proposed Transformer outperforms the pooled PrefMMT last-layer fine-tuning baseline across all three user groups.

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
├── main.py
└── README.md
```

The repository is organized around configuration-driven experiments, model implementations, training and adaptation routines, checkpoints, and generated results.

## Running the Experiments

Run commands from the project root.

### Display available commands

```bash
python main.py --help
```

### Train the proposed model

```bash
python main.py train "tasks=[reach-v3]" seed=0 train.steps=6000
```

### Adapt to a user

```bash
python main.py adapt "tasks=[reach-v3]" seed=0
```

### Train multiple seeds

```powershell
foreach ($s in 0, 1, 2, 3, 4) {
    python main.py train "tasks=[reach-v3]" seed=$s train.steps=6000
    if ($LASTEXITCODE -ne 0) { break }

    python main.py adapt "tasks=[reach-v3]" seed=$s
    if ($LASTEXITCODE -ne 0) { break }
}
```

### Run other experiment modules

The CLI exposes the following experiment commands:

```bash
python main.py ablations
python main.py audit
python main.py baselines
python main.py collect
python main.py heterogeneity
python main.py iql
python main.py pairs
python main.py querying
python main.py report
python main.py stats
python main.py users
```

Use `python main.py <command> --help` where supported, and consult the configuration files for experiment-specific settings.

## Checkpoints and Outputs

- `checkpoints/` stores trained model checkpoints.
- `results/tables/` stores generated experimental result tables.
- `configs/` contains experiment and model configurations.

Training runs use configuration-dependent identifiers, including task, feature dimension, encoder, pooling configuration and random seed.

## Reproducibility Notes

- Training seeds and adaptation/query seeds represent different sources of variability.
- The original Transformer checkpoints for seeds 1–4 were discovered to be severely undertrained. These runs were subsequently retrained for the intended 6,000 steps using separate `repair`-tagged checkpoints.
- The repaired checkpoints were used to regenerate the corresponding random-query adaptation results.
- The original undertrained checkpoints should not be used as valid trained-model replications.
- Five-seed aggregate results should be interpreted alongside the checkpoint and dataset provenance qualifications documented in the report.

## Limitations

- Active-querying results require qualification because the original ensemble included undertrained checkpoints.
- Downstream IQL experiments have limited evaluation coverage, including a single training seed in the reported setup.
- Several architecture and representation ablations use limited seeds and task coverage.
- StatsMLP receives privileged behavioral statistics and should not be interpreted as an equivalent information-access baseline to the proposed model.
- The PrefMMT comparison uses pooled last-layer fine-tuning rather than full-network fine-tuning.

## Research Report

See the accompanying research report for:
- Detailed mathematical formulation.
- Experimental methodology.
- Baseline comparisons.
- Few-shot adaptation results.
- Ablation studies.
- Active preference querying.
- Downstream offline reinforcement learning.
- Discussion and limitations.

## Authors

Developed as an academic research project on personalized reward modeling and preference-based reinforcement learning by Bonta Aalaya and Vaishnavi Nitin Pawar
