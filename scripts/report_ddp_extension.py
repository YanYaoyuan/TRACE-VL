"""Summarize the 450 -> 2250 step DDP continuation and matched-video controls."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "outputs/llava-2b-ddp-5epochs-20261105"
BASE = PROJECT / "outputs/llava-2b-baseline-cfr-epoch1-20261105"


def read(path):
    return json.loads(Path(path).read_text())


def normalized(value):
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def bootstrap_video_em(results, seed=7, resamples=5000):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["video"]].append(int(row["matched_correct"]) - int(row["mismatched_correct"]))
    units = [(sum(values), len(values)) for _, values in sorted(grouped.items())]
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        selected = rng.choices(units, k=len(units))
        estimates.append(sum(s for s, n in selected) / sum(n for s, n in selected))
    estimates.sort()
    return {
        "matched_minus_mismatched_em": sum(s for s, n in units) / sum(n for s, n in units),
        "ci95": [estimates[int((resamples - 1) * p)] for p in (0.025, 0.975)],
        "resampling_unit": "video",
        "videos": len(units),
        "resamples": resamples,
        "seed": seed,
    }


def main():
    locations = [
        (450, BASE / "val", BASE / "visual_dependency_full.json", BASE / "last.pt"),
        (
            900,
            RUN / "validation-step-00900",
            RUN / "validation-step-00900/visual_dependency_full.json",
            RUN / "step-00900.pt",
        ),
        (
            2250,
            RUN / "validation-step-02250",
            RUN / "validation-step-02250/visual_dependency_full.json",
            RUN / "step-02250.pt",
        ),
    ]
    rows = []
    question_ids = None
    truth = {
        row["id"]: row
        for row in [
            json.loads(line)
            for line in (PROJECT / "data/processed/llava-pilot-cfr.jsonl").read_text().splitlines()
        ]
        if row["split"] == "val"
    }
    base_predictions = {
        row["id"]: row["answer"]
        for row in [json.loads(line) for line in (BASE / "val/predictions.jsonl").read_text().splitlines()]
    }
    base_em = read(BASE / "val/scores.json")["qa_exact_match"]
    for step, directory, diagnostic_path, checkpoint in locations:
        scores = read(directory / "scores.json")
        diagnostic = read(diagnostic_path)
        ids = {row["id"] for row in diagnostic["results"]}
        question_ids = ids if question_ids is None else question_ids
        assert ids == question_ids and len(ids) == 200
        assert scores["provenance"]["checkpoint_sha256"] == diagnostic["checkpoint_sha256"]
        predictions = {
            row["id"]: row["answer"]
            for row in [
                json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()
            ]
        }
        assert set(predictions) == set(base_predictions) == set(truth)
        paired = bootstrap_video_em(
            [
                {
                    "video": row["video"],
                    "matched_correct": normalized(predictions[ident]) == normalized(row["answer"]),
                    "mismatched_correct": normalized(base_predictions[ident]) == normalized(row["answer"]),
                }
                for ident, row in truth.items()
            ]
        )
        assert abs(paired["matched_minus_mismatched_em"] - (scores["qa_exact_match"] - base_em)) < 1e-10
        rows.append(
            {
                "step": step,
                "checkpoint": str(checkpoint.relative_to(PROJECT)),
                "qa_exact_match": scores["qa_exact_match"],
                "qa_token_f1": scores["qa_token_f1"],
                "canonical_minus_control_matched_em": scores["qa_exact_match"]
                - diagnostic["generation"]["matched_exact_match"],
                "generation_control": diagnostic["generation"],
                "matched_answer_loss": diagnostic["matched_mean_answer_loss"],
                "mismatched_answer_loss": diagnostic["mismatched_mean_answer_loss"],
                "answer_loss_gap": diagnostic["mean_loss_increase"],
                "paired_video_bootstrap": bootstrap_video_em(diagnostic["results"]),
                "canonical_improvement_vs_step450": paired,
                "checkpoint_sha256": diagnostic["checkpoint_sha256"],
                "manifest_sha256": diagnostic["manifest_sha256"],
            }
        )
    assert len({r["manifest_sha256"] for r in rows}) == 1
    best = max(rows, key=lambda r: (r["qa_exact_match"], r["qa_token_f1"], -r["step"]))
    target = os.path.relpath(PROJECT / best["checkpoint"], RUN)
    link = RUN / "best.pt"
    if link.is_symlink():
        assert os.readlink(link) == target, "existing best.pt points to another file"
    elif link.exists():
        raise FileExistsError("refusing to replace an existing best.pt")
    else:
        link.symlink_to(target)
    selection = {
        "step": best["step"],
        "checkpoint": best["checkpoint"],
        "sha256": best["checkpoint_sha256"],
        "rule": "Highest local validation EM among 450/900/2250, then token F1, then earlier step; not independent test selection.",
    }
    (RUN / "best_checkpoint.json").write_text(json.dumps(selection, indent=2) + "\n")
    logs = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logs] == list(range(451, 2251))
    run = read(RUN / "run.json")
    resources = read(RUN / "resources.json")
    batch = (
        run["world_size"] * run["config"]["training"]["batch_size"] * run["config"]["training"]["grad_accum"]
    )
    assert batch == 4 and run["world_size"] == 2
    summary = {
        "date": "2026-11-05",
        "model_changed": False,
        "jepa": False,
        "start_step": 450,
        "end_step": 2250,
        "new_optimizer_steps": len(logs),
        "global_batch": batch,
        "world_size": 2,
        "runs": rows,
        "best_checkpoint": selection,
        "last50_training_loss": sum(r["losses"]["answer"] for r in logs[-50:]) / 50,
        "epoch_mean_answer_ce": {
            epoch: sum(r["losses"]["answer"] for r in logs if (r["step"] - 1) // 450 + 1 == epoch) / 450
            for epoch in range(2, 6)
        },
        "resources": resources,
        "run": run,
        "timing_note": "Intermediate and baseline diagnostic evaluations overlapped training on GPU0; elapsed time includes contention and is not a DDP throughput benchmark.",
        "resume_note": "Weights, AdamW state and optimizer step restored at a dataset-pass boundary. RNG/data iterator not restored; DDP uses its own seeded DistributedSampler.",
        "source_sha256": {
            str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "src").rglob("*.py"))
        },
    }
    output = PROJECT / "docs/experiments"
    output.mkdir(parents=True, exist_ok=True)
    name = "ddp_extension_2026-11-05"
    (output / (name + ".json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    base_logs = [json.loads(line) for line in (BASE / "metrics.jsonl").read_text().splitlines()]
    for label, sequence, color in [
        ("Single GPU", base_logs, "#2563eb"),
        ("DDP continuation", logs, "#d97706"),
    ]:
        values = [r["losses"]["answer"] for r in sequence]
        mean = [sum(values[max(0, i - 49) : i + 1]) / min(i + 1, 50) for i in range(len(values))]
        axes[0].plot([r["step"] for r in sequence], mean, color=color, label=label)
    axes[0].axvline(450, color="#64748b", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Cumulative optimizer step",
        ylabel="Answer CE (50-step mean)",
        title="Same architecture and global batch = 4",
    )
    axes[0].legend(frameon=False)
    for offset, key, color, label in [
        (-0.18, "matched_exact_match", "#2563eb", "Matched video"),
        (0.18, "mismatched_exact_match", "#94a3b8", "Mismatched video"),
    ]:
        values = [r["generation_control"][key] for r in rows]
        bars = axes[1].bar([i + offset for i in range(3)], values, width=0.34, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.1%}", ha="center", fontsize=9
            )
    axes[1].set_xticks(range(3), [str(r["step"]) for r in rows])
    axes[1].set(
        xlabel="Checkpoint step",
        ylabel="Local exact match",
        title="200 QA / 20 held-out videos",
        ylim=(0, max(r["qa_exact_match"] for r in rows) + 0.12),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(output / (name + "." + extension), dpi=180, bbox_inches="tight")
    plt.close(figure)
    lines = [
        "# 双卡延长训练：2026-11-05",
        "",
        "从现有 450 步模型与 AdamW 状态恢复，两张 RTX 5090 新增 1800 步，累计 2250 步；按样本呈现累计约五轮。保持原有架构，无 JEPA，冻结两骨干，每段四帧、384×384、BF16、学习率 1e-4。每卡 batch=1/累积2，全局有效 batch=4，与原单卡相同。",
        "",
        "数据仍为 180 个训练视频 / 1800 QA、20 个验证视频 / 200 QA，按视频隔离，没有加入新数据。恢复不包含 RNG 和数据迭代器状态，DDP 使用自己的固定种子采样顺序。",
        "",
        f"最终标准 EM 为 {rows[-1]['qa_exact_match']:.2%}，450 步基线为 {base_em:.2%}。但最终模型在视频错配后仍有 {rows[-1]['generation_control']['mismatched_exact_match']:.2%} EM，只有 {rows[-1]['generation_control']['fraction_answer_changed']:.2%} 的答案变化；增加训练轮数尚未建立可靠视觉依赖。下一步优先改进视觉对齐数据、训练策略与消融，而不把这次小幅点估计提升当作架构优势。",
        "",
        f"![训练与视觉错配对照]({name}.png)",
        "",
        "## 验证集与生成对照",
        "",
        "| 累计步数 | 标准评测 EM | 标准评测 token F1 | 控制实验正确视频 EM | 控制实验错配视频 EM | 答案变化比例 | 配对 EM 差值的按视频 bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        control, interval = row["generation_control"], row["paired_video_bootstrap"]["ci95"]
        lines.append(
            f"| {row['step']} | {row['qa_exact_match']:.2%} | {row['qa_token_f1']:.2%} | {control['matched_exact_match']:.2%} | {control['mismatched_exact_match']:.2%} | {control['fraction_answer_changed']:.2%} | [{interval[0]:+.2%}, {interval[1]:+.2%}] |"
        )
    lines += [
        "",
        "训练集多数答案 yes 的本地 EM 为 11%。这里是归一化词面完全匹配与 token F1，没有使用官方 ActivityNetQA judge，不与论文排行榜直接比较。",
        "",
        "标准评测与配对控制的执行路径分别记录，没有假定逐位一致。例如 450 步的历史标准 EM 为 28.5%，本次配对控制正确视频 EM 为 28.0%，相差一题；两者均保留，未强行合并。控制实验差值和置信区间只使用同次配对结果。",
        "",
        "视觉对照固定相同问题、答案、时间戳和帧数，将视频替换为另一验证视频；20 个视频按固定种子循环配对，三个 checkpoint 使用相同映射。生成不接收真实答案。置信区间按视频进行 5000 次成对 bootstrap，避免将同视频的十个问题当成十个独立视频；该区间仍不能支持跨数据集泛化结论。",
        "",
        "| 累计步数 | 正确视频答案 CE | 错配视频答案 CE | 错配 − 正确 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['matched_answer_loss']:.4f} | {row['mismatched_answer_loss']:.4f} | {row['answer_loss_gap']:+.4f} |"
        )
    lines += [
        "",
        "## 延长训练的配对比较",
        "",
        "使用标准评测保存的同一批预测，按视频成对 bootstrap，与 450 步基线比较：",
        "",
        "| 累计步数 | 相对450步 EM 变化 | 按视频 bootstrap 95% CI |",
        "|---|---:|---|",
    ]
    for row in rows:
        estimate = row["canonical_improvement_vs_step450"]
        interval = estimate["ci95"]
        lines.append(
            f"| {row['step']} | {estimate['matched_minus_mismatched_em']:+.2%} | [{interval[0]:+.2%}, {interval[1]:+.2%}] |"
        )
    lines += [
        "",
        "区间跨零时，不把点估计的提升称为稳定或显著提升。验证集仅有20个独立视频，也不将本轮数据上的选择推广到其他场景。",
        "",
        "## checkpoint 与资源",
        "",
        f"按上述三个点中的验证 EM、token F1、较早步数依次选择，当前 best.pt 指向 **{best['step']} 步**；选择依据见 outputs/llava-2b-ddp-5epochs-20261105/best_checkpoint.json。这是验证集选择，不是独立测试结果。",
        "",
        f"双卡新增训练作业耗时 {resources['seconds'] / 60:.1f} 分钟，末50步训练 CE 均值 {summary['last50_training_loss']:.4f}。中途只读评测与训练共享 GPU0，所以耗时包含资源竞争，不用于单双卡吞吐对比。",
        "",
        "| GPU rank | 峰值 allocated GiB | 峰值 reserved GiB |",
        "|---|---:|---:|",
    ]
    for rank in resources["ranks"]:
        lines.append(
            f"| {rank['rank']} | {rank['peak_allocated_bytes'] / 2**30:.2f} | {rank['peak_reserved_bytes'] / 2**30:.2f} |"
        )
    lines += [
        "",
        "PyTorch 显存统计按训练进程记录，不包括其他进程、CUDA context 等，不能直接等同于 nvidia-smi 总占用。",
        "",
        "主目录：outputs/llava-2b-ddp-5epochs-20261105/。包含 run.json、metrics.jsonl、resources.json、torchrun 两个 rank 的日志、step-00900.pt、step-02250.pt、last.pt、best.pt 及两次验证/错配结果。2250 步与 last.pt 是同一文件的硬链接，不重复占用模型空间。",
        "",
        "## 复现",
        "",
        "```bash",
        "env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked torchrun --standalone --nproc_per_node=2 -m trace_vl train --config configs/local_2b_ddp_5epochs.yaml --resume outputs/llava-2b-baseline-cfr-epoch1-20261105/last.pt --output outputs/ddp-extension-reproduction",
        "bash scripts/evaluate_ddp_milestone.sh outputs/ddp-extension-reproduction/step-02250.pt outputs/ddp-extension-reproduction/validation-final",
        "```",
        "",
        "本轮保持模型结构，只扩展 checkpoint 保留、双 rank 资源记录和诊断脚本。短视频结果不能证明长视频记忆或告警性能，数据和视觉对齐仍需后续专项实验。",
    ]
    (output / (name + ".md")).write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"report": str(output / (name + ".md")), "best_step": best["step"], "results": rows}, indent=2
        )
    )


if __name__ == "__main__":
    main()
