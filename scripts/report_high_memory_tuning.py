"""Build the local high-memory tuning report from saved measurements (no GPU needed)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from report_ddp_extension import bootstrap_video_em, normalized

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/tuning-20261105"
DOC = ROOT / "docs/experiments/high_memory_tuning_2026-11-05"


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    rows = []
    truth = {
        r["id"]: r for r in lines(ROOT / "data/processed/llava-pilot-tuning-cfr.jsonl") if r["split"] == "val"
    }
    base_predictions = {r["id"]: r["answer"] for r in lines(RUN / "baseline-4frames/predictions.jsonl")}
    for name in ("baseline-4frames", "lr1e4", "lr3e5"):
        directory = RUN / name
        evaluation = directory if name == "baseline-4frames" else directory / "val"
        scores = read(evaluation / "scores.json")
        control = read(evaluation / "visual_dependency_full.json")
        predictions = {r["id"]: r["answer"] for r in lines(evaluation / "predictions.jsonl")}
        assert set(predictions) == set(base_predictions) == set(truth)
        assert {r["id"] for r in control["results"]} == set(truth)
        assert scores["provenance"]["checkpoint_sha256"] == control["checkpoint_sha256"]
        assert scores["provenance"]["manifest_sha256"] == control["manifest_sha256"]
        row = {
            "name": name,
            "em": scores["qa_exact_match"],
            "f1": scores["qa_token_f1"],
            "generation_control": control["generation"],
            "loss_increase": control["mean_loss_increase"],
            "visual_bootstrap": bootstrap_video_em(control["results"]),
            "baseline_bootstrap": bootstrap_video_em(
                [
                    {
                        "video": r["video"],
                        "matched_correct": normalized(predictions[k]) == normalized(r["answer"]),
                        "mismatched_correct": normalized(base_predictions[k]) == normalized(r["answer"]),
                    }
                    for k, r in truth.items()
                ]
            ),
            "checkpoint_sha256": control["checkpoint_sha256"],
            "manifest_sha256": control["manifest_sha256"],
        }
        if name != "baseline-4frames":
            run = read(directory / "run.json")
            metrics = lines(directory / "metrics.jsonl")
            resources = read(directory / "resources.json")
            assert metrics[-1]["step"] == 450 and len(metrics) == 450
            row.update(
                checkpoint=str((directory / "last.pt").relative_to(ROOT)),
                learning_rates=run["optimizer_learning_rates"],
                initialization=run["initialization"],
                resources=resources,
                last50_mean_loss=sum(r["losses"]["total"] for r in metrics[-50:]) / 50,
            )
        rows.append(row)
    assert len({r["manifest_sha256"] for r in rows}) == 1
    assert rows[1]["initialization"]["checkpoint_sha256"] == rows[2]["initialization"]["checkpoint_sha256"]
    best = max(rows[1:], key=lambda r: (r["em"], r["f1"], -r["learning_rates"][0]))
    checkpoint = ROOT / best["checkpoint"]
    link = RUN / "best.pt"
    target = os.path.relpath(checkpoint, RUN)
    if link.is_symlink():
        assert os.readlink(link) == target
    elif link.exists():
        raise FileExistsError(link)
    else:
        link.symlink_to(target)
    capacity = []
    for path in sorted(RUN.glob("capacity-*.json")):
        value = read(path)
        # Early probes predate these explicit fields; recover their unchanged config defaults.
        value.setdefault("normalize_memory", value["config"]["model"].get("normalize_memory", False))
        value.setdefault("precision", value["config"]["training"].get("precision", "bf16"))
        value.setdefault("detach_every", value["config"]["training"].get("detach_every", 0))
        capacity.append(
            {
                "file": str(path.relative_to(ROOT)),
                **{
                    k: value.get(k)
                    for k in (
                        "batch_size",
                        "frames",
                        "spatial_tokens",
                        "normalize_memory",
                        "precision",
                        "detach_every",
                        "status",
                        "peak_allocated_gib",
                        "peak_reserved_gib",
                        "seconds_per_step",
                    )
                },
            }
        )
    report = {
        "date": "2026-11-05",
        "rows": rows,
        "capacity": capacity,
        "best_new_arm": best["name"],
        "selection": "validation EM, token F1, then lower learning rate; not independent test",
        "checks": {"pytest": "46 passed, 2 skipped", "ruff": "passed"},
    }
    (RUN / "best_checkpoint.json").write_text(json.dumps(report, indent=2) + "\n")
    DOC.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")

    def pct(v):
        return f"{100 * v:.2f}%"

    def interval(r):
        return "[" + ", ".join(f"{100 * x:+.2f}" for x in r["ci95"]) + "] pp"

    text = [
        "# 双卡高显存调参：2026-11-05\n",
        "将每段输入从最多4帧提高到32帧，RGB token 从96提高到224，文本上限256提高到512。每卡 batch 从1提高到2，累积从2降到1，全局有效 batch 仍为4。保持384×384、BF16和两个冻结骨干，总参数2,104,675,891，可训练参数67,817,011。没有引入JEPA。\n",
        "## 数值修复与实验边界\n",
        "直接扩大序列时出现非有限梯度。激活审计显示旧2250步权重的工作状态最大绝对值由第1帧106.76增至第16帧约7.42e7；仅截断反向或提高新模块精度未解决。新增可选无参数LayerNorm，将门控工作状态和新胶囊归一化。两组新实验均开启normalize_memory，旧四帧基线保持关闭。因此新旧比较同时改变输入预算和归一化行为，不能归因于单个因素；两组新实验间仅学习率不同。\n",
        "两组均从同一个2250步checkpoint独立初始化，seed=7，重建AdamW、步数归零，各训练450步，即1800条QA的一次呈现。--init-from与--resume分离，避免恢复旧优化器时覆盖新学习率。\n",
        "![显存、验证与训练曲线](high_memory_tuning_2026-11-05.png)\n",
        "## 结果\n",
        f"新配置中最高标准EM为{pct(best['em'])}，相对四帧基线{100 * (best['em'] - rows[0]['em']):+.2f}个百分点。扩大输入预算没有在本轮获得更高验证EM。旧2250步checkpoint及其best链接保持原样；本目录best.pt仅在两组新配置之间选择。\n",
        "| 配置 | 标准EM | token F1 | 正确视频EM | 错配视频EM | 答案变化 | 正确−错配EM 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        g = row["generation_control"]
        text.append(
            f"| {row['name']} | {pct(row['em'])} | {pct(row['f1'])} | {pct(g['matched_exact_match'])} | {pct(g['mismatched_exact_match'])} | {pct(g['fraction_answer_changed'])} | {interval(row['visual_bootstrap'])} |"
        )
    text += [
        "\n| 新配置 | 相对四帧基线EM | 按视频bootstrap 95% CI | 错配−正确答案CE |",
        "|---|---:|---|---:|",
    ]
    for row in rows[1:]:
        b = row["baseline_bootstrap"]
        text.append(
            f"| {row['name']} | {100 * b['matched_minus_mismatched_em']:+.2f} pp | {interval(b)} | {row['loss_increase']:+.6f} |"
        )
    text += [
        f"\n两组新配置按验证EM、token F1、较低学习率依次选择，best.pt指向{best['name']}。这是验证集选择，不是独立测试结果。当前测得的分数与视觉错配差值应分别判断；不能由更高显存占用推断模型已学会看视频。\n",
        "本轮正确/错配视频得分仍接近，答案变化比例仅约几个百分点，视觉依赖依然很弱。末50步训练CE下降也未对应更高验证EM。下一轮应优先排查视觉对齐数据、语言先验和前缀融合训练，加入受控视觉监督与对应消融；现有结果不足以定位唯一原因。记忆归一化解决了长展开数值稳定性，但不是问答效果提升的证据。\n",
        "## 显存与容量边界\n",
        "| 新配置 | rank | allocated GiB | reserved GiB | 分配GB | 训练分钟 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[1:]:
        for rank in row["resources"]["ranks"]:
            text.append(
                f"| {row['name']} | {rank['rank']} | {rank['peak_allocated_bytes'] / 2**30:.2f} | {rank['peak_reserved_bytes'] / 2**30:.2f} | {rank['peak_allocated_bytes'] / 1e9:.2f} | {rank['seconds'] / 60:.2f} |"
            )
    text += [
        "\n| 容量探测 B×T×RGB tokens | 归一化 | 精度 | detach间隔（0=不截断） | 结果 | 峰值allocated GiB |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in capacity:
        peak = row["peak_allocated_gib"]
        text.append(
            f"| {row['batch_size']}×{row['frames']}×{row['spatial_tokens']} | {row['normalize_memory']} | {row['precision']} | {row['detach_every']} | {row['status']} | {peak:.2f} |"
            if peak is not None
            else f"| {row['batch_size']}×{row['frames']}×{row['spatial_tokens']} | {row['normalize_memory']} | {row['precision']} | {row['detach_every']} | {row['status']} | — |"
        )
    text += [
        "\n探测采用每进程92% allocator上限，OOM表示触及本次探测预算，不是理论硬件极限。表中fp32指关闭autocast，新模块采用FP32，冻结Qwen仍按存储dtype加载，并非全模型转为FP32。早期探测未显式记录的归一化、精度和detach设置由其保存配置的默认值恢复。正式DDP不设置allocator上限；资源表为各训练进程PyTorch峰值，不含CUDA context和其他进程。GB=10^9 bytes，GiB=2^30 bytes。\n",
        "## 数据与复现\n",
        "沿用180训练/20验证视频，1800/200 QA，来源视频隔离；只有QA监督，不含告警标签。更密集解码审计发现一条验证MP4帧索引失败，另存CFR修复副本，涉及同视频10题；原始媒体与旧manifest保留，新旧三组评测统一使用llava-pilot-tuning-cfr.jsonl。2000条记录的问题、答案、来源组、划分及观测窗口逐项核对保持一致；未增加训练数据。\n",
        "标准评测和错配控制均覆盖200题。控制仅更换另一个验证视频的像素，保持目标问题/答案/时间戳/帧数；生成不接收答案。相同观测前缀可复用问题无关状态，测试确认复用不改变答案和证据。所有置信区间按20个视频进行5000次成对bootstrap。这里是本地归一化词面EM/token F1，不是官方ActivityNetQA judge，也未评估监控告警或长期记忆。\n",
        "```bash\nenv -u PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl train --config configs/tuning/high_memory_lr3e5.yaml --init-from outputs/llava-2b-ddp-5epochs-20261105/best.pt --output outputs/high-memory-reproduction\nbash scripts/evaluate_ddp_milestone.sh outputs/high-memory-reproduction/last.pt outputs/high-memory-reproduction/val configs/tuning/high_memory_lr3e5.yaml\n```\n",
        "测试：46 passed、2 skipped；ruff通过。容量探测、激活审计、媒体修复与两个rank的完整训练日志均在outputs/tuning-20261105/。JSON报告保留checkpoint与manifest哈希。\n",
    ]
    DOC.with_suffix(".md").write_text("\n".join(text))
    plot(rows, capacity)
    print(
        json.dumps(
            {
                "best_new_arm": best["name"],
                "results": [{k: r[k] for k in ("name", "em", "f1", "generation_control")} for r in rows],
            },
            indent=2,
        )
    )


def plot(rows, capacity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = ["#96a4b5", "#2563eb", "#0d9488"]
    for i, row in enumerate(rows):
        g = row["generation_control"]
        axes[0].bar(
            i - 0.19, 100 * row["em"], 0.35, color=colors[i], label="Canonical EM" if i == 0 else None
        )
        axes[0].bar(
            i + 0.19,
            100 * g["mismatched_exact_match"],
            0.35,
            color=colors[i],
            alpha=0.35,
            label="Wrong-video EM" if i == 0 else None,
        )
    axes[0].set(
        xticks=range(3),
        xticklabels=["4-frame base", "32f / LR 1e-4", "32f / LR 3e-5"],
        ylabel="Exact match (%)",
        title="Held-out QA: 20 videos / 200 questions",
    )
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend(frameon=False, fontsize=8)
    for index, row in enumerate(rows[1:], 1):
        logs = lines(RUN / row["name"] / "metrics.jsonl")
        smooth = [
            sum(x["losses"]["total"] for x in logs[max(0, i - 24) : i + 1])
            / len(logs[max(0, i - 24) : i + 1])
            for i in range(len(logs))
        ]
        axes[1].plot([r["step"] for r in logs], smooth, color=colors[index], label=row["name"])
    axes[1].set(
        xlabel="Fresh optimizer step", ylabel="Answer CE, 25-step mean", title="Same step-2250 initialization"
    )
    axes[1].legend(frameon=False)
    for i, row in enumerate(rows[1:]):
        for rank in row["resources"]["ranks"]:
            x = i * 3 + rank["rank"]
            axes[2].bar(x, rank["peak_reserved_bytes"] / 2**30, color=colors[i + 1], alpha=0.25)
            axes[2].bar(x, rank["peak_allocated_bytes"] / 2**30, color=colors[i + 1], width=0.5)
    axes[2].axhline(31.37, color="#64748b", linestyle="--", linewidth=1, label="Reported device capacity")
    axes[2].set(
        xticks=[0, 1, 3, 4],
        xticklabels=["1e-4 / 0", "1e-4 / 1", "3e-5 / 0", "3e-5 / 1"],
        ylabel="GiB per training process",
        title="Allocated / reserved GPU peak",
        ylim=(0, 34),
    )
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "TRACE-VL 2.105B | RTX 5090 x2 | BF16 | 32 frames x 224 RGB tokens", fontsize=14, fontweight="bold"
    )
    for suffix in ("png", "svg"):
        fig.savefig(DOC.with_suffix("." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
