# 阿里云镜像安装训练环境

本入口用于 Linux x86_64 上的 Python 3.10、Torch 2.8.0+cu128 视频训练环境。使用 uv 0.9.21 验证。

```bash
bash scripts/setup_aliyun.sh --dry-run   # 审计镜像锁和安装计划，不修改 .venv
bash scripts/setup_aliyun.sh             # 实际安装
```

## 镜像、锁文件与 CUDA 源

仓库根目录的 `uv.lock` 是使用 PyPI 生成的基准锁文件，包含包文件的完整 URL。看到
`files.pythonhosted.org` 不代表镜像配置已经生效。`uv sync --frozen` 直接使用锁文件中的
下载地址，不会把它们改成阿里云地址；更换索引后再使用 `--locked` 则可能因锁文件需要变化而报错。

脚本在独立的临时项目中复制基准锁文件，替换以下地址，**不重新解析依赖**：

1. 默认索引使用 `https://mirrors.aliyun.com/pypi/simple`。
2. PyPI 包文件使用 `https://mirrors.aliyun.com/pypi/packages/`。
3. Torch、Torchvision 的 cu128 包文件使用 `https://mirrors.aliyun.com/pytorch-wheels/cu128/`，
   保留 `2.8.0+cu128` / `0.23.0+cu128` 构建。NVIDIA CUDA 运行库依赖使用阿里云 PyPI 镜像。
4. 安装前检查镜像锁文件与原锁完全一致，仅 URL 前缀可变；版本、依赖关系、条件标记和全部文件哈希均不变。
   遇到未映射的下载源、版本或哈希变化时退出。镜像缺包时报错，不自动回退国外源或升级、降级。
5. 对镜像锁执行 frozen sync 安装 dev、vlm、data 依赖，再将当前工程安装为 editable，运行 `uv pip check`，
   并核对实际安装版本均存在于基准锁。

阿里云 cu128 目录是平铺 wheel 目录，`/cu128/torch/` 不是可用的包索引。
这里直接使用锁定的完整 wheel 地址；不会把目录当作普通 PEP 503 索引重新求解。

根目录的 `pyproject.toml`、`uv.lock` 保持原样。生成的文件位于已被 Git 忽略的
`outputs/environments/aliyun/`：`uv.lock`、临时项目配置和成功时的 `provenance.json`。
成功报告记录源锁 SHA256、全部锁定版本，以及实际安装版本（预演时为空）。
源锁可能为不同 Python 分支锁定多个版本；实际安装固定选择 Python 3.10 分支。
进入安装阶段前会清除旧成功报告。

```bash
grep -m 3 'mirrors.aliyun.com' outputs/environments/aliyun/uv.lock
git diff --exit-code -- pyproject.toml uv.lock
```

普通 Python 包、CUDA wheel 和 NVIDIA 运行库均通过阿里云下载。Hugging Face 模型和视频数据
使用各自的下载入口。脚本给 UV 下载超时设置默认值 300 秒，可自行覆盖：

```bash
UV_HTTP_TIMEOUT=600 bash scripts/setup_aliyun.sh
```

## 已经修改了根目录 uv.lock

先确认 `git rev-parse --short HEAD` 是预期的训练或修复提交，再备份并恢复依赖文件。
脚本在 Git 工程中会拒绝有本地改动的 `pyproject.toml` / `uv.lock`，不会覆盖这些改动。

```bash
cp -p uv.lock "uv.lock.backup.$(date +%Y%m%d-%H%M%S)"
git restore --source=HEAD -- uv.lock
bash scripts/setup_aliyun.sh
```

如 `pyproject.toml` 也有改动，先另行备份和检查，不要直接丢弃。

## 安装后验证和训练

```bash
uv run --no-sync python - <<'PY'
import torch, torchvision, transformers, peft
print(torch.__version__, torchvision.__version__, transformers.__version__, peft.__version__)
print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())
PY
uv run --no-sync pytest -q
```

已验证骨干对应的核心版本为 Torch 2.8.0+cu128、Torchvision 0.23.0+cu128、
Transformers 5.20.2、PEFT 0.21.2。后续训练命令使用 `uv run --no-sync ...`；已有的
`uv run --locked --no-sync ...` 云端脚本也可继续使用。重新执行普通 `uv sync` 会回到根锁文件的
索引策略，需要镜像安装时再次使用本入口。

本入口安装依赖，不更改模型、数据、训练配置，也不自动开始训练。

参考：[UV 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)、
[UV 包索引](https://docs.astral.sh/uv/concepts/indexes/)、
[阿里云 PyPI 镜像](https://developer.aliyun.com/mirror/pypi)。
CUDA 镜像：[阿里云 PyTorch wheels](https://developer.aliyun.com/mirror/pytorch-wheels)。
