import importlib.util, tempfile, math, ast
from pathlib import Path
from types import SimpleNamespace
import torch
root=Path(__file__).parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('diag',root/'verl/utils/grpo_diagnostics.py')
b={'responses':torch.ones(4,4),'attention_mask':torch.ones(4,6),
'loss_mask':torch.tensor([[1,1,0,0],[1,0,0,0],[1,1,1,0],[0,0,0,0]]),
 'token_level_scores':torch.tensor([[0.,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),
 'advantages':torch.tensor([[1.,1,1,1],[-1.,-1,-1,-1],[0,0,0,0],[0,0,0,0]])}
b['token_level_rewards']=b['token_level_scores']
r=m.batch_metrics(SimpleNamespace(batch=b,non_tensor_batch={'uid':['a','a','b','b']}),{'step':2,'gen':1})
assert r['tokens/action_total']==6 and r['tokens/observation_mean']==2.5
assert r['grpo/zero_variance_group_fraction']==.5
assert r['grpo/all_zero_score_group_fraction']==.5
assert math.isclose(r['grpo/sequence_advantage_mean'],0)
assert r['data/empty_action_fraction']==.25
assert r['throughput/action_tokens_per_s_step']==3
u=m.update_metrics(torch.tensor([[math.log(.5),math.log(2),100.]]),torch.zeros(1,3),torch.tensor([[1.,-1.,1.]]),torch.tensor([[1,1,0]]),.2)
p='diagnostics/update_micro_mean/'
assert u[p+'positive/ratio_below_clip_fraction']==1
assert u[p+'negative/ratio_above_clip_fraction']==1
assert u[p+'all/count']==2
# TensorBoard real roundtrip, isolated temporary event directory.
t=load('tracking',root/'verl/utils/tracking.py')
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
with tempfile.TemporaryDirectory() as d:
 w=SummaryWriter(d);a=t._TensorBoardAdapter(w);a.step_offset=100
 a.log({'critic/rewards/mean':.5,'train/score_mean':.5,'actor/entropy_loss':1.2},3);w.close()
 ea=EventAccumulator(d);ea.Reload()
 assert ea.Scalars('legacy/critic/rewards/mean')[0].step==103
 assert ea.Scalars('train/score_mean')[0].value==.5
 assert 'policy/entropy_micro_mean' in ea.Tags()['scalars']
# Scheduler restore regression: load_state_dict clobbers scheduled LR, repair restores it.
x=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([x],lr=.01)
s=torch.optim.lr_scheduler.LambdaLR(o,lambda t:.5)
expected=s.get_last_lr();sd=o.state_dict();sd['param_groups'][0]['lr']=.0001
o.load_state_dict(sd)
for g,lr in zip(o.param_groups,expected):g['lr']=lr
assert o.param_groups[0]['lr']==.005
for f in root.rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
print('PASS: masked/group metrics, positive-negative ratios, real TensorBoard roundtrip, scheduler regression, all Python syntax')
# Pre-update entropy: exact reference, noncontiguous tensors, masks and partitions.
z=torch.randn(3,7,13,dtype=torch.float32)[:,1:6,:]
original=z.clone()
h=m.entropy_fp32_chunked(z,chunk_size=2)
ref=torch.distributions.Categorical(logits=z.double()).entropy()
assert torch.allclose(h.double(),ref,atol=1e-6)
assert torch.equal(z,original), 'diagnostics must not mutate logits'
assert torch.allclose(m.entropy_fp32_chunked(torch.zeros(2,4)),torch.full((2,),math.log(4)))
assert m.entropy_fp32_chunked(torch.tensor([[100.,-100.]])).item()<1e-6
# Unequal token counts: token mean=2, sequence mean=3 (not rank/micro mean).
e=torch.tensor([[1.,1,1],[5.,float('nan'),float('nan')],[9.,9,9]])
mask=torch.tensor([[1,1,1],[1,0,0],[0,0,0]],dtype=torch.bool)
a=m.entropy_metrics(e,mask)
assert a['policy/entropy_pre_token_mean']==2
assert a['policy/entropy_pre_sequence_mean']==3
assert a['policy/entropy_pre_nonfinite_fraction']==0
# Equivalent DP gather is partition invariant, including a rank with no valid actions.
parts=[m.entropy_metrics(e[i:i+1],mask[i:i+1]) for i in range(3)]
num=sum(p.get('policy/entropy_pre_token_mean',0)*p['policy/entropy_pre_action_token_count'] for p in parts)
assert num/sum(p['policy/entropy_pre_action_token_count'] for p in parts)==2
assert 'policy/entropy_pre_token_mean' not in m.entropy_metrics(e,torch.zeros_like(mask))
bad=m.entropy_metrics(torch.tensor([[float('nan')]]),torch.ones(1,1,dtype=torch.bool))
assert bad['policy/entropy_pre_nonfinite_fraction']==1 and math.isnan(bad['policy/entropy_pre_token_mean'])
# Raw T=1 entropy stays independent of the temperature used for policy logprobs.
raw=torch.tensor([[1.,2.,4.]])
h1=m.entropy_fp32_chunked(raw)
for temperature in (.5,1.,2.):
 temp=raw.clone();before=m.entropy_fp32_chunked(temp);temp.div_(temperature)
 assert torch.equal(before,h1)
 assert torch.allclose(torch.log_softmax(temp,-1),torch.log_softmax(raw/temperature,-1))
print('PASS: full-vocabulary entropy reference, T=1, masks, empty/nonfinite cases, DP partition invariance, nonmutation')
