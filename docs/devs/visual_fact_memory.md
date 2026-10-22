# 视觉事实模块：迁移说明

当前流程见 [从历史轨迹的标准答案重建图文经验库](visual_fact_supervised.md)。

## 当前推荐流程

1. `inventory-trajectories`：直接读取历史轨迹保存的标准答案、选项文字及来源，不需要原数据集。
2. `assemble`：依据已有标注时间或关键帧对齐，零 VLM 调用，不重新生成标签。
3. `build --allow-source-grounded`：使用已有 WeMM 服务建立新索引。
4. `visual_facts_supervised.yml` 开启参考注入，`visual_facts_supervised_off.yml` 关闭。

模型预测、最终答对标志和普通观察描述都不自动成为正确标签。缺少标准答案或对应证据的轨迹会在清单中说明原因。

## 已删除的在线功能

在线 support/conflict review、current-only 二次复看、`visual_memory.review` 配置及 `visual_facts_review.yml` / `visual_facts_current_only.yml` 已删除。新在线流程每次局部观察只调用一次 VLM。旧配置给出明确迁移提示；请从服务器命令和本地覆盖文件中移除这些设置。

新库标签带 `source_grounded` 来源，并默认排除旧 `model_reviewed` 事实。不要将旧机器生成记录直接改状态冒充有监督标签；应从原始来源重新组装。历史版本可以通过 Git 提交记录恢复。

## 保留的离线兼容入口

原先通过独立数据集标注生成来源清单的 `inventory --dataset ...`、显式选择案例的 `curate`、人工核验案例索引仍可用。

旧的 `mine` 会重新调用 VLM 从像素生成候选并逐条审核。这不属于当前推荐流程；为复现实验保留代码，调用时必须显式传 `--allow-generated-labels`：

```bash
python -m videospy.visual_memory.cli mine --allow-generated-labels \
  --sources /runs/legacy_sources.jsonl \
  --config config/videospy/visual_fact_mining.yml \
  --output output/visual_memory/legacy_generated_facts
```

其输出仍标记 `model_reviewed`，不会被新的监督事实配置接纳。保留这段离线代码不表示在线 review 仍然存在。
