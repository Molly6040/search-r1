"""CPU-only synthetic event smoke test; NEVER mix with actual training runs."""
import argparse
import importlib.util
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument('--logdir', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tracking_smoke', root / 'verl/utils/tracking.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path = Path(a.logdir) / 'SYNTHETIC_SMOKE_NOT_TRAINING'
if path.exists():
    raise SystemExit('Use a fresh logdir; refusing to append synthetic data')
w = SummaryWriter(str(path))
adapter = module._TensorBoardAdapter(w)
for step in range(3):
    adapter.log({'train/score_mean': step / 10, 'val/test_score/nq': step / 20,
                 'policy/entropy_pre_token_mean': 2 - step / 10,
                 'grpo/zero_variance_group_fraction': .5,
                 'actor/grad_norm': .2, 'actor/lr': 1e-6,
                 'throughput/action_tokens_per_s_step': 100}, step)
w.close()
e = EventAccumulator(str(path)); e.Reload()
assert len(e.Scalars('train/score_mean')) == 3
assert e.Scalars('optim/grad_norm_preclip_minibatch_mean')[0].value > 0
assert 'custom_scalars__config__' in e.Tags()['tensors']
print('PASS synthetic event roundtrip and custom dashboard:', path)
