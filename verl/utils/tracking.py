# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console', 'tensorboard']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            import os
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

        if 'tensorboard' in default_backend:
            # Search-R1 复现新增: tensorboard 后端, logdir 由 TENSORBOARD_LOGDIR 覆盖,
            # 默认 ./tb_logs/<experiment_name>; 指标 key 与 wandb/console 完全一致
            from torch.utils.tensorboard import SummaryWriter
            import os
            _logdir = os.environ.get('TENSORBOARD_LOGDIR', os.path.join('./tb_logs', experiment_name))
            os.makedirs(_logdir, exist_ok=True)
            self.logger['tensorboard'] = _TensorBoardAdapter(SummaryWriter(log_dir=_logdir))

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)


class _TensorBoardAdapter:
    """把 fit()/validate() 的 metrics dict 逐 key 写成 TB scalar; 非数值 key 跳过
    Search-R1 复现新增: TB_STEP_OFFSET 让续训曲线接上原曲线的 x 轴(全局步数)"""

    def __init__(self, writer):
        self.writer = writer
        import os
        self.step_offset = int(os.environ.get('TB_STEP_OFFSET', '0'))
        # Presentation only; no additional model forwards or loss changes.
        self.writer.add_custom_scalars({'01 Quality': {'Score': ['Multiline', ['train/score_mean', 'val/test_score/.*']], 'Completion': ['Multiline', ['env/finish_ratio', 'data/empty_action_fraction']]}, '02 Entropy': {'Pre-update entropy': ['Multiline', ['policy/entropy_pre_token_mean', 'policy/entropy_pre_sequence_mean']], 'Nonfinite': ['Multiline', ['policy/entropy_pre_nonfinite_fraction']]}, '03 GRPO': {'Inactive groups': ['Multiline', ['grpo/zero_variance_group_fraction', 'grpo/all_zero_score_group_fraction', 'grpo/all_one_score_group_fraction']], 'Group size': ['Multiline', ['grpo/group_size_min', 'grpo/group_size_max']], 'Reward spread': ['Multiline', ['grpo/reward_std_mean']]}, '04 Update': {'Positive ratios': ['Multiline', ['diagnostics/update_micro_mean/positive/ratio_.*fraction']], 'Negative ratios': ['Multiline', ['diagnostics/update_micro_mean/negative/ratio_.*fraction']], 'KL': ['Multiline', ['policy/reference_kl_micro_mean']], 'Gradient': ['Multiline', ['optim/grad_norm_preclip_minibatch_mean']], 'LR': ['Multiline', ['optim/lr_next_rollout_update']]}, '05 Runtime': {'Tokens': ['Multiline', ['tokens/action_mean', 'tokens/observation_mean']], 'Throughput': ['Multiline', ['throughput/action_tokens_per_s_.*']], 'Seconds': ['Multiline', ['timing_s/.*']]}})

    def log(self, data, step):
        for key, value in data.items():
            try:
                # Legacy names are preserved under legacy/, not silently reinterpreted.
                if key.startswith(('critic/score/', 'critic/rewards/', 'critic/advantages/', 'critic/returns/', 'response_length/', 'prompt_length/', 'state_tokens/', 'timing_per_token_ms/')):
                    key = 'legacy/' + key
                aliases = {'actor/entropy_loss': 'policy/entropy_micro_mean',
                           'actor/ppo_kl': 'policy/old_current_k1_micro_mean',
                           'actor/kl_loss': 'policy/reference_kl_micro_mean',
                           'actor/grad_norm': 'optim/grad_norm_preclip_minibatch_mean',
                           'actor/lr': 'optim/lr_next_rollout_update'}
                key = aliases.get(key, key)
                self.writer.add_scalar(key, float(value), global_step=step + self.step_offset)
            except (TypeError, ValueError):
                pass
        self.writer.flush()


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans
