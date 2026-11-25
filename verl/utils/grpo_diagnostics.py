"""Read-only diagnostics. Masks exclude padding and tool observations."""
from collections import defaultdict
import torch

@torch.no_grad()
def batch_metrics(batch, timing):
    b = batch.batch
    width = b['responses'].shape[-1]
    context = b['attention_mask'][:, -width:].bool()
    action = b.get('loss_mask', b.get('info_mask', b['attention_mask'])[:, -width:]).bool() & context
    scores = b['token_level_scores'].sum(-1).float()
    rewards = b['token_level_rewards'].sum(-1).float()
    counts = action.sum(-1)
    m = {'train/score_mean': scores.mean().item(), 'train/reward_mean': rewards.mean().item(),
         'tokens/action_mean': counts.float().mean().item(),
         'tokens/observation_mean': (context.sum(-1)-counts).float().mean().item(),
         'tokens/action_total': counts.sum().item(),
         'tokens/context_mean': context.sum(-1).float().mean().item(),
         'data/empty_action_fraction': (counts == 0).float().mean().item()}
    groups = defaultdict(list)
    for uid, value in zip(batch.non_tensor_batch['uid'], rewards.tolist()):
        groups[uid].append(value)
    stds = [torch.tensor(v).std(unbiased=True).item() if len(v)>1 else 0. for v in groups.values()]
    m.update({'grpo/group_count': len(groups),
              'grpo/group_size_min': min(map(len, groups.values())),
              'grpo/group_size_max': max(map(len, groups.values())),
              'grpo/reward_std_mean': sum(stds)/len(stds),
              'grpo/zero_variance_group_fraction': sum(s == 0 for s in stds)/len(stds),
              'grpo/singleton_group_fraction': sum(len(v)==1 for v in groups.values())/len(groups)})
    binary = bool(((scores==0)|(scores==1)).all())
    if binary:
        sg = defaultdict(list)
        for uid, value in zip(batch.non_tensor_batch['uid'], scores.tolist()): sg[uid].append(value)
        m['train/binary_score_success_fraction'] = scores.mean().item()
        m['grpo/all_zero_score_group_fraction'] = sum(not any(v) for v in sg.values())/len(sg)
        m['grpo/all_one_score_group_fraction'] = sum(all(v) for v in sg.values())/len(sg)
    seq_adv = (b['advantages'] * action).sum(-1) / counts.clamp_min(1)
    valid = counts > 0
    if valid.any():
        m['grpo/sequence_advantage_mean'] = seq_adv[valid].mean().item()
        m['grpo/zero_advantage_sequence_fraction'] = (seq_adv[valid]==0).float().mean().item()
    for name in ('step', 'gen'):
        if timing.get(name, 0)>0:
            m['throughput/action_tokens_per_s_'+name] = counts.sum().item()/timing[name]
    if timing.get('step',0)>0: m['throughput/trajectories_per_s'] = len(scores)/timing['step']
    return m

@torch.no_grad()
def update_metrics(logp, old, advantage, mask, clip):
    valid = mask.bool()
    delta = (logp-old).detach().float()
    ratio = delta.exp()
    result = {}
    for label, selected in [('all', valid), ('positive', valid & (advantage>0)), ('negative', valid & (advantage<0))]:
        n = selected.sum().item()
        result[label+'/count'] = n
        if n:
            d=delta[selected]; r=ratio[selected]
            result[label+'/logratio_mean']=d.mean().item()
            result[label+'/ratio_mean']=r.mean().item()
            result[label+'/ratio_below_clip_fraction']=(r<1-clip).float().mean().item()
            result[label+'/ratio_above_clip_fraction']=(r>1+clip).float().mean().item()
            result[label+'/k3_old_current_sample_mean']=(r-1-d).mean().item()
    # Nonnegative sample estimator; not exact KL and not rollout-engine mismatch.
    return {'diagnostics/update_micro_mean/'+k:v for k,v in result.items()}


@torch.no_grad()
def entropy_fp32_chunked(logits, chunk_size=128):
    """T=1 full-vocabulary entropy, float32; bounded temporary allocation.

    Iterate slices without flattening the full possibly noncontiguous logits.
    Called only in no-grad pre-update diagnostics; never changes entropy loss.
    """
    shape = logits.shape[:-1]
    out = torch.empty(shape, device=logits.device, dtype=torch.float32)
    if logits.ndim == 2:
        for start in range(0, logits.shape[0], chunk_size):
            z = logits[start:start+chunk_size].float()
            lp = torch.log_softmax(z, dim=-1)
            out[start:start+chunk_size] = -(lp.exp()*lp).sum(-1)
    elif logits.ndim == 3:
        for i in range(logits.shape[0]):
            out[i] = entropy_fp32_chunked(logits[i], chunk_size)
    else:
        raise ValueError('Expected token/vocab or batch/token/vocab logits')
    return out

@torch.no_grad()
def entropy_metrics(entropy, mask):
    """Driver-side aggregation AFTER data-parallel gather, no rank-mean bias."""
    if entropy.shape != mask.shape:
        raise ValueError('Entropy/mask shape mismatch')
    mask = mask.bool()
    count = mask.sum(-1)
    nonempty = count > 0
    finite = torch.isfinite(entropy)
    result = {'policy/entropy_pre_action_token_count': count.sum().item(),
              'policy/entropy_pre_nonempty_sequence_count': nonempty.sum().item()}
    n = count.sum().item()
    if not n:
        return result
    result['policy/entropy_pre_nonfinite_fraction'] = ((~finite)&mask).sum().item()/n
    # Invalid action values must be visible; do not silently drop them.
    selected = torch.where(mask, entropy.double(), torch.zeros_like(entropy, dtype=torch.float64))
    sums = selected.sum(-1)
    result['policy/entropy_pre_token_mean'] = sums.sum().item()/n
    result['policy/entropy_pre_sequence_mean'] = (sums[nonempty]/count[nonempty]).mean().item()
    return result
