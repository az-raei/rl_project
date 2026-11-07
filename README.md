# Personalized reward features from preferences (Meta-World -> IQL)

```bash
pip install -r requirements.txt
python main.py collect tasks=[reach-v3] collect.episodes_per_behavior=3   # smoke test
python main.py collect                                                    # full run, tasks from configs/default.yaml
python main.py pairs                                                      # segments + psi + candidate pairs
python main.py users                                                      # synthetic users + BT labels
python main.py train tasks=[reach-v3] model.K=16 train.steps=6000        # needs torch (GPU strongly advised); stops early when val acc plateaus (train.early_stop_patience)
python main.py adapt tasks=[reach-v3]                                     # few-shot adaptation of unseen users

# (defaults already use stop_after_success=10, 200 episodes/behavior, L=25, stride=5, max_per_behavior=1000)
python tests/run_tests.py                                                 # or: pytest
```

Everything in the plan is implemented (no stubs left): data, synthetic users, shared reward-feature model,
PrefMMT / stats-MLP baselines, few-shot adaptation, random + ensemble active querying, IQL, ground-truth policy
evaluation, ablations, report. Torch-dependent code was written without a local GPU/torch; the numpy parts are
unit-tested, the torch parts are covered by tests that run on Kaggle (first run: `python tests/run_tests.py`).

Model input is state + action only. Images are a planned later ablation.

Note on K: synthetic rewards are linear in the 6 psi features, so any K >= 6 can represent every user.
Expect the K in {8, 16, 32} ablation to be flat; add K in {4, 6} to see where capacity actually binds.

## Kaggle (GPU training)
1. Upload your local `data/` folder (raw, trajectories, pairs, users) as a private Kaggle Dataset.
2. New notebook, Accelerator = GPU, Internet = On, Add Input = that dataset. Import `scripts/kaggle_train.ipynb`
   (or paste its cells). `scripts/kaggle_run.py` finds the data and sets all `paths.*` overrides for you.
3. Order: tests -> 2000-step smoke run -> full train -> adapt -> bundle (`outputs.zip` in the Output panel).
Use the same `--K/--temporal/--tag` for train and adapt (they form the checkpoint name).

## Baselines
`model.encoder=prefmmt` swaps in a PrefMMT-style hierarchical state/action transformer (see
`src/models/prefmmt_baseline.py` for what is from the paper vs assumed).
  * PrefMMT encoder inside our scheme:  `python main.py train model.encoder=prefmmt model.K=16`
  * Original single-reward PrefMMT:     `python main.py train model.encoder=prefmmt model.K=1 train.pooled=true`

## Kaggle workflow (no zips)
One-time: Kaggle notebook -> Add-ons -> Secrets -> add `GH_TOKEN` (fine-grained GitHub token, *Contents: read & write*, this repo only).

Cell 1 (every session):
```python
import os
REPO = "https://github.com/<user>/<repo>.git"; D = "<repo>"
if os.path.isdir(D): !git -C {D} pull
else: !git clone {REPO}
%cd {D}
```
Cell 2 (one experiment, end to end):
```python
from kaggle_secrets import UserSecretsClient
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
!bash scripts/kaggle_run.sh baseline-k16 model.K=16 train.steps=5000     # tests -> train -> adapt, both tasks
!bash scripts/push_results.sh baseline-k16                               # prints a link to paste to Claude
```
Results land on branch `results-<name>` (never on main): `runs/<name>/tables/*_summary.txt` is the compact table.

## Reproduce everything
```bash
# data (only if data/ is not already in the repo)
python main.py collect && python main.py pairs && python main.py users

bash scripts/reproduce.sh try1 quick tests          # 1) unit tests incl. torch + an end-to-end fake-data pipeline test
bash scripts/reproduce.sh try1 quick baselines      # 2) ours vs PrefMMT (encoder / pooled) vs stats-MLP   (train+adapt)
bash scripts/reproduce.sh try1 quick ablations      # 3) K, temporal module, prior strength
bash scripts/reproduce.sh try1 quick querying       # 4) random vs active querying (ensemble of 5)
bash scripts/reproduce.sh try1 quick iql            # 5) IQL + ground-truth evaluation (needs baselines first)
bash scripts/reproduce.sh try1 quick report         # 6) markdown tables + figures
bash scripts/reproduce.sh final full                # final numbers, all stages (hours)
```
Single commands behind each stage: `python main.py {train|stats|adapt|baselines|ablations|querying|iql|report}`.
Outputs: `results/tables/*.csv` (+ `*_summary.txt`, `summary_*.md`), `results/figures/*.png`, `checkpoints/*.pt`.
What each result table means:
  * `*_adapt_l2*.csv` / `summary_adapt_<task>.md`: preference accuracy vs the TRUE preference on test pairs, by
    heterogeneity group, budget, method (population / ours_zero_prior / ours_pop_prior / pooled_* / psi_oracle) and query strategy.
  * `*_iql.csv`: per policy, the GROUND-TRUTH reward of the user it is meant for, env success and env return. Read it as:
    does `ours` beat `population` (gain column), and how far are `ref_scripted_expert` / `env_reward`?
Limits (by design / time): the stats-MLP is segment-level, so it has no IQL variant; PrefMMT fine-tuning is
last-layer only; active querying scores one fixed random candidate subset per user; IQL uses a compact in-repo implementation.

### Windows PowerShell
```powershell
.\scripts\reproduce.ps1 try1 quick tests          # same stages and profiles as reproduce.sh
.\scripts\reproduce.ps1 try1 quick baselines
.\scripts\reproduce.ps1 final full
$env:GH_TOKEN = "<token>"; .\scripts\push_results.ps1 try1
```
Quote bracketed lists in PowerShell: `python main.py adapt "tasks=[reach-v3,pick-place-v3]"` (an unquoted comma splits the argument).
If scripts are blocked: `Set-ExecutionPolicy -Scope Process Bypass`.
