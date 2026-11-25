# Search-R1 GRPO diagnostics

基于 Search-R1 官方 commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3` 和训练机器代码快照的修正版。

- [指标定义与看板说明](METRICS-GUIDE.txt)
- [代码审计与性能建议](AUDIT.txt)
- [相对训练快照的修改补丁](reviewed-changes.patch)
- [原项目文档](README.upstream.md)

## 主要改动

更新前 T=1 全词表熵，按全局有效动作 token 加权，并提供轨迹等权均值；排除工具和 padding。
增加 GRPO 组内奖励、概率比和吞吐监控，澄清 TensorBoard 标签。
修复训练步数边界、恢复 optimizer 后的学习率覆盖和 KL 日志覆盖；移除被丢弃的逐轮 logprob 重算。

## 本地计算测试

```bash
python -m venv .venv-test
.venv-test/bin/pip install torch numpy tensorboard
.venv-test/bin/python test_audit.py
```

测试不需要训练 GPU。已通过 CPU 数值检查和 TensorBoard 写入读取；尚未做 CUDA/FSDP/vLLM 多卡验收。
训练依赖按原项目文档安装。脚本中的模型/数据路径来自原训练机器，运行前须按本机环境调整。
审计文档中本机目录名用于说明来源；本仓库根目录即修正版代码，未包含原始压缩包、训练数据、日志或虚拟环境。

## TensorBoard

训练配置启用 `trainer.logger=['console','wandb','tensorboard']`。
设置 `TENSORBOARD_LOGDIR` 为每次恢复训练的独立目录，再运行 `tensorboard --logdir <日志根目录>`。
不要使用历史合并脚本拼接不同训练分支。新指标不会自动出现在旧事件文件中。

保留原项目 LICENSE 和 Notice.txt。详细算法和安装说明见原项目文档。
