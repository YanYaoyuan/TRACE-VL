# 2026-11-25 实验汇总数据

对应[近期研发说明](../../research_results_20261125.md)。这些文件保存已经完成的实验统计与独立复核记录；完整视频、权重和原始候选logit不装入Git。

| 文件 | 内容 |
|---|---|
| [summary.json](summary.json) | 统一主指标、区间、预算、运行身份及当前决策 |
| [causal_pool_analysis.json](causal_pool_analysis.json) | 原Plain/Causal两组分析及原门槛决定 |
| [static_pool_review.json](static_pool_review.json) | 独立Static训练后三组预测的复核；汇总采用此版本区间 |
| [longstream_states_review.json](longstream_states_review.json) | 77源/385查询/6臂的完整延迟分数与身份核验记录 |
| [writer_controls_review.json](writer_controls_review.json) | 77源/385查询/10臂的分数、状态事件与精确重放核验记录 |
| [bank_readout_review.json](bank_readout_review.json) | 77源/847预测的逐项复核结论 |
| [bank_readout_paired_sources.json](bank_readout_paired_sources.json) | 全77源各臂正确性和实际读取token数；顺序与冻结输入一致 |
| [validation.md](validation.md) | 本PR提交前全套CPU回归、环境限制及检查结果 |
| [artifacts.json](artifacts.json) | 上述JSON的SHA256及工作区来源 |

`causal_pool_analysis.json`与`static_pool_review.json`在相同预测上的bootstrap端点略有差异，来自重采样遍历顺序；汇总统一采用完整静态对照复核区间，没有挑选区间。全部区间为开发分析的探索性pointwise区间，不是多重比较校正或最终未见测试结果。

复算bank读出主比较（在仓库根目录、有NumPy的Python环境执行）：

```python
import json
from pathlib import Path
import numpy as np

root = Path("docs/core_vlm/results/20261125")
rows = json.loads((root / "bank_readout_paired_sources.json").read_text())["rows"]
assert len(rows) == len({r["source_group"] for r in rows}) == 77
for policy in ("uniform8", "all_bank"):
    differences = np.array([
        sum(r["correct_fp32"][s + "_" + policy]
            - r["correct_fp32"][s + "_latest8"]
            for s in ("original", "29", "43")) / 3
        for r in rows
    ])
    draws = np.random.default_rng(7).choice(differences, (4000, 77), replace=True).mean(1)
    print(policy, 100 * differences.mean(), 100 * np.quantile(draws, [.025, .975]))
```

先在源内平均三个记忆随机流，再重采样77个源。all_bank不是128token等预算比较；实际最大400、平均196.71视觉token，上限配置512。原始审计包的校验身份与A100输出位置见汇总说明；仅凭这些轻量文件不能重新执行视觉推理或独立认证所有原始RGB。
