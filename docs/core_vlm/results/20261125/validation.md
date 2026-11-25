# 本PR提交前验证

- 完整CPU回归：538 passed,7 skipped，45.04秒；1条已有tensor-to-scalar warning。
- 环境：现有项目venv；`CUDA_VISIBLE_DEVICES=""`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`，`PYTHONPATH=src python -m pytest -q -o faulthandler_timeout=90`。
- 首次沙箱执行停在多进程训练用例：子进程创建本地Unix socket抛出PermissionError。保留中断日志，允许本地IPC后完整套件通过；没有删去训练用例。
- `ruff check src tests scripts`：通过。
- `git diff --check`、关键池化/状态/writer/bank启动器的`bash -n`、更新文档相对链接：通过。
- `artifacts.json`内所有证据JSON的SHA256一致；全部77源逐臂正确数与实际token独立核对；源内3种子平均、77源4000次bootstrap/seed7的两项差值及区间复算一致。

上述为本地回归和既有A100实验的验证，不代表GitHub Actions已运行或通过。真实模型完整实验与控制重放结果见总报告；本次文档整理没有重新训练或重跑已完成的A100实验。
