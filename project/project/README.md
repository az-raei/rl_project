# Personalized reward features from preferences (Meta-World -> IQL)

```bash
pip install -r requirements.txt
python main.py collect tasks=[reach-v3] collect.episodes_per_behavior=3   # smoke test
python main.py collect                                                    # full run, tasks from configs/default.yaml
python main.py pairs                                                      # segments + psi + candidate pairs
python main.py users                                                      # synthetic users + BT labels
python main.py train tasks=[reach-v3] model.K=16 train.steps=20000        # needs torch (GPU strongly advised)
python main.py adapt tasks=[reach-v3]                                     # few-shot adaptation of unseen users

# (defaults already use stop_after_success=10, 200 episodes/behavior, L=25, stride=5, max_per_behavior=1000)
python tests/run_tests.py                                                 # or: pytest
```

Implemented: configs, utils, collector, psi features, segments + pair builder, synthetic users + BT labels,
model (encoder / feature head / personalized reward), shared-feature pretraining, eval metrics,
few-shot adaptation with random querying, tests.
Stubs (raise NotImplementedError): baselines, ensemble querying, rl, experiments.

Model input is state + action only. Images are a planned later ablation.

Note on K: synthetic rewards are linear in the 6 psi features, so any K >= 6 can represent every user.
Expect the K in {8, 16, 32} ablation to be flat; add K in {4, 6} to see where capacity actually binds.

## Kaggle (GPU training)
1. Upload your local `data/` folder (raw, trajectories, pairs, users) as a private Kaggle Dataset.
2. New notebook, Accelerator = GPU, Internet = On, Add Input = that dataset. Import `scripts/kaggle_train.ipynb`
   (or paste its cells). `scripts/kaggle_run.py` finds the data and sets all `paths.*` overrides for you.
3. Order: tests -> 2000-step smoke run -> full train -> adapt -> bundle (`outputs.zip` in the Output panel).
Use the same `--K/--temporal/--tag` for train and adapt (they form the checkpoint name).
