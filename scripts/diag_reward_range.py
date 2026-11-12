import numpy as np
from src.dataset.dataset import SegmentStore, load_pairs, load_segments
from src.training.adapt_user import load_model, prepare_features
from src.training.evaluate_reward_model import embed_segments
from src.utils.io import load_config

cfg = load_config(["default", "users", "model", "query", "adapt"])
task = "reach-v3"
store, pr, seg = SegmentStore.load(cfg, task, "cpu"), load_pairs(cfg, task), load_segments(cfg, task)
model, _ = load_model(cfg, task)
Phi = embed_segments(model, store).astype(np.float64)
pool = np.flatnonzero(pr["pair_split"] == cfg["adapt"]["query_pool_split"])
_, Phi_s, w_pop = prepare_features(Phi, pr["pair_i"][pool], pr["pair_j"][pool],
                                   model.user_w.weight.detach().cpu().numpy())
r = Phi_s @ w_pop
tr = store.split == 0
z = (r - r[tr].mean()) / r[tr].std()
print("population reward (z-scored), ALL dataset segments")
print("  percentiles 50/90/99/99.9/max:", np.round(np.percentile(z, [50, 90, 99, 99.9, 100]), 2))
print("  the IQL population policy reached a mean of 2.70 on its rollouts")
for b in np.unique(seg["seg_behavior"]):
    zb = z[seg["seg_behavior"] == b]
    print(f"  {b:18s} mean {zb.mean():6.2f}  p99 {np.percentile(zb, 99):6.2f}  max {zb.max():6.2f}")