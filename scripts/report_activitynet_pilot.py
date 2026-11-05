"""Build a local pilot report from completed run artifacts; plotting needs matplotlib."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def collect(relative):
    root = PROJECT / relative
    run = read(root / "run.json")
    resources = read(root / "resources.json")
    scores = read(root / "val/scores.json")
    diagnostic = read(root / "visual_dependency.json")
    logs = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in logs] == list(range(1, len(logs) + 1))
    assert scores["provenance"]["checkpoint_sha256"] == diagnostic["checkpoint_sha256"]
    assert run["manifest_sha256"] == diagnostic["manifest_sha256"]
    return {
        "directory": str(relative),
        "steps": len(logs),
        "training_sample_presentations": len(logs) * run["config"]["training"]["grad_accum"],
        "first10_mean_loss": sum(r["losses"]["answer"] for r in logs[:10]) / 10,
        "last10_mean_loss": sum(r["losses"]["answer"] for r in logs[-10:]) / 10,
        "qa_exact_match": scores["qa_exact_match"],
        "qa_token_f1": scores["qa_token_f1"],
        "validation_samples": scores["samples"],
        "peak_allocated_gib": resources["rank0_peak_allocated_bytes"] / 2**30,
        "peak_reserved_gib": resources["rank0_peak_reserved_bytes"] / 2**30,
        "training_loop_seconds_including_checkpoint": resources["seconds"],
        "matched_mean_answer_loss": diagnostic["matched_mean_answer_loss"],
        "mismatched_mean_answer_loss": diagnostic["mismatched_mean_answer_loss"],
        "visual_diagnostic_loss_increase": diagnostic["mean_loss_increase"],
        "visual_diagnostic_videos": diagnostic["videos"],
        "checkpoint_sha256": diagnostic["checkpoint_sha256"],
        "manifest_sha256": run["manifest_sha256"],
        "parameters": run["parameters"],
        "config": run["config"],
    }, logs


def main():
    paths = [
        Path("outputs/llava-2b-baseline-cfr-20261105"),
        Path("outputs/llava-2b-baseline-cfr-epoch1-20261105"),
    ]
    collected = [collect(path) for path in paths]
    runs = [item[0] for item in collected]
    assert runs[0]["manifest_sha256"] == runs[1]["manifest_sha256"]
    assert runs[0]["config"]["model"] == runs[1]["config"]["model"]
    majority = read(PROJECT / "data/processed/llava-pilot-cfr.majority_baseline.json")
    source_hashes = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src").rglob("*.py"))
    }
    revisions = {}
    cache = Path.home() / ".cache/huggingface/hub"
    for repo in ("Qwen/Qwen3-1.7B", "google/siglip2-large-patch16-384"):
        reference = cache / ("models--" + repo.replace("/", "--")) / "refs/main"
        revisions[repo] = reference.read_text().strip() if reference.exists() else None
    summary = {
        "date": "2026-11-05",
        "scope": "Current architecture, no JEPA; local short-video alignment pilot, not a paper benchmark.",
        "dataset": {"videos": 200, "train_videos": 180, "val_videos": 20, "train_qa": 1800, "val_qa": 200},
        "media_audit": read(PROJECT / "data/processed/llava-pilot-cfr.media_audit.json"),
        "backbone_revisions": revisions,
        "source_sha256_after_experiments": source_hashes,
        "majority_answer_baseline": majority,
        "checks": {"pytest_passed": 41, "pytest_cuda_skipped_in_sandbox": 2, "ruff": "passed"},
        "runs": runs,
    }
    output = PROJECT / "docs/experiments"
    output.mkdir(parents=True, exist_ok=True)
    prefix = "activitynetqa_baseline_2026-11-05"
    (output / (prefix + ".json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    colors = ["#2563eb", "#d97706"]
    for (run, logs), color in zip(collected, colors):
        steps = [r["step"] for r in logs]
        losses = [r["losses"]["answer"] for r in logs]
        smooth = [sum(losses[max(0, i - 9) : i + 1]) / min(i + 1, 10) for i in range(len(losses))]
        axes[0].plot(steps, losses, color=color, alpha=0.16, linewidth=0.6)
        axes[0].plot(steps, smooth, color=color, label=f"{run['steps']} steps (10-step mean)")
    axes[0].set(xlabel="Optimizer step", ylabel="Answer cross-entropy", title="Training loss")
    axes[0].legend(frameon=False)
    values = [majority["qa_exact_match"]] + [r["qa_exact_match"] for r in runs]
    bars = axes[1].bar(["Constant\nyes", "100 steps", "450 steps"], values, color=["#94a3b8"] + colors)
    axes[1].set(title="Held-out exact match (200 QA / 20 videos)", ylim=(0, max(values) + 0.08))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.1%}", ha="center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("TRACE-VL frozen-backbone alignment pilot | 4 frames, one RTX 5090", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / (prefix + "." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 现有架构真实视频训练基线：2026-11-05",
        "",
        "使用现有 TRACE-VL 架构，没有接入 JEPA。以下均为真实本机运行结果，属于短视频训练试验，不是官方 benchmark 成绩。",
        "",
        f"![训练曲线与验证结果]({prefix}.png)",
        "",
        "## 配置与数据",
        "",
        "- Qwen3-1.7B + SigLIP2 large，实际总参数 2,104,675,891；requires_grad 参数 67,817,011。冻结视觉和语言骨干，本轮只有问答损失，告警/写入/utility 头没有对应监督。",
        "- 单张 RTX 5090，BF16，每段最多 4 帧，384×384，每卡 batch=1，梯度累积 4，学习率 1e-4，seed=7。",
        "- ActivityNetQA 短视频：180 个训练视频 / 1,800 QA，20 个验证视频 / 200 QA；按视频隔离。两次独立训练从同一随机种子开始，不通过恢复重放数据迭代器。",
        "- 100 步使用 400 次训练问答呈现；450 步完整遍历 1,800 条训练问答一轮。相同种子不保证 GPU 逐位确定性。",
        "- 原始 200 个视频中有一个 MKV 的容器估计帧数导致末帧读取失败。全部 7 个 MKV 生成 CFR H.264 副本，CRF18、源平均帧率、无音频；193 个 MP4 未变。200 个视频均完成当前采样配置的解码检查。原始媒体保留。",
        "",
        "## 实测结果",
        "",
        "| 试验 | 前10步均值 CE | 后10步均值 CE | 验证 EM | 验证 token F1 | 峰值 allocated / reserved GiB | 训练循环秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 训练集最常见答案：yes | — | — | {majority['qa_exact_match']:.2%} | {majority['qa_token_f1']:.2%} | — | — |",
    ]
    for run in runs:
        lines.append(
            f"| {run['steps']} 步 | {run['first10_mean_loss']:.3f} | {run['last10_mean_loss']:.3f} | "
            f"{run['qa_exact_match']:.2%} | {run['qa_token_f1']:.2%} | "
            f"{run['peak_allocated_gib']:.2f} / {run['peak_reserved_gib']:.2f} | "
            f"{run['training_loop_seconds_including_checkpoint']:.1f} |"
        )
    lines += [
        "",
        "CE 为每步四个 microbatch 的答案交叉熵均值。训练时间包含数据加载和最终 checkpoint 保存，不包含骨干加载；PyTorch allocated/reserved 不等同于 nvidia-smi 的进程总显存。",
        "",
        "EM 是归一化后的词面完全匹配，token F1 是词面重叠；同义词和数字表达差异也可能判错。200 个问题来自 20 个视频，不能当作 200 个独立视频样本。没有使用官方 ActivityNetQA judge，不与论文排行榜分数直接比较。",
        "",
        "## 视觉依赖诊断",
        "",
        "每个验证视频通过固定种子的 ID 哈希选择一道题，20 题循环交换视频像素，保持问题、答案、时间戳和帧数不变。比较 teacher-forced 答案 CE；没有改变上表的生成评测结果。",
        "",
        "| checkpoint | 正确视频 CE | 错配视频 CE | 错配 − 正确 |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['steps']} 步 | {run['matched_mean_answer_loss']:.4f} | {run['mismatched_mean_answer_loss']:.4f} | {run['visual_diagnostic_loss_increase']:+.4f} |"
        )
    lines += [
        "",
        "该诊断只有 20 题，正差值本身也不能证明可靠视觉推理，负差值或接近零则不能作为有效视觉对齐的证据。训练损失下降不能替代该检查或保留集评测。",
        "",
        "## 判断与下一步",
        "",
        f"完整一轮的本地 EM 为 {runs[-1]['qa_exact_match']:.2%}，训练集多数答案基线为 {majority['qa_exact_match']:.2%}。四帧冻结骨干训练已跑通，但视觉错配平均 CE 只变化 {runs[-1]['visual_diagnostic_loss_increase']:+.4f}，尚不能证明模型建立了可靠的视觉依赖。优先完善视觉对齐数据和训练策略，继续用视觉错配对照检验；这些结果不能用于证明时空记忆结构的优势。",
        "",
        "Ruff 通过；沙箱 pytest 为 41 passed、2 CUDA tests skipped。真实 GPU 路径另由本次实际训练、checkpoint 恢复和验证推理覆盖。",
        "",
        "## 复现与产物",
        "",
        "```bash",
        "env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked trace-vl train --config configs/local_2b_epoch1.yaml --output outputs/epoch1-reproduction",
        "env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked trace-vl evaluate --config configs/local_2b_epoch1.yaml --checkpoint outputs/epoch1-reproduction/last.pt --split val --output outputs/epoch1-reproduction/val",
        "env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked python scripts/diagnose_visual_dependency.py --config configs/local_2b_epoch1.yaml --checkpoint outputs/epoch1-reproduction/last.pt --output outputs/epoch1-reproduction/visual_dependency.json",
        "```",
        "",
        "复现需要已有的本地权重和处理后媒体。主结果目录：",
        "",
    ]
    lines += [
        f"- `{run['directory']}/`：last.pt、run.json、metrics.jsonl、resources.json、val/scores.json、val/predictions.jsonl、visual_dependency.json。"
        for run in runs
    ]
    lines += [
        "",
        "数据处理记录：data/processed/llava-pilot-cfr.preprocessing.json；解码记录：llava-pilot-cfr.media_audit.json。模型缓存 revision、manifest/checkpoint SHA256 和运行后源文件哈希保存在同名 JSON 报告。Git 仓库尚无提交，git_revision 为 null。",
        "",
        "早期 5 步检查保存在 outputs/llava-2b-baseline-20261105/；原始 MKV 导致的 42 步中断记录保存在 outputs/llava-2b-baseline-20261105-100steps/，不计为完成的 100 步试验。",
        "",
        "四帧短视频没有检验长视频记忆淘汰、实时摄像头吞吐、codec-native 路径或告警准确率；公开论文比较仍需单独接入 OVO/StreamingBench 等协议。",
    ]
    (output / (prefix + ".md")).write_text("\n".join(lines) + "\n")
    print(output / (prefix + ".md"))


if __name__ == "__main__":
    main()
