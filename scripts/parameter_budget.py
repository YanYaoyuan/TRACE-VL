"""Count new module parameters without downloading pretrained weights."""

import argparse
import json
from dataclasses import replace

import torch

from trace_vl.config import ModelConfig, load_config
from trace_vl.models.model import TraceVL
from trace_vl.runtime import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/reference_2b.yaml")
parser.add_argument("--output", default="docs/implementation_budget.json")
args = parser.parse_args()
config = load_config(args.config)
with torch.device("meta"):
    model = TraceVL(replace(ModelConfig.from_dict(config["model"]), backend="tiny"))
components = {
    k: v for k, v in model.parameter_report()["components"].items() if k not in {"vision", "language"}
}
components.update({name: p.numel() for name, p in model.named_parameters(recurse=False)})
report = {
    "status": "exact_new_module_shapes_on_meta_device_no_pretrained_weights_loaded",
    "new_components": components,
    "new_parameters": sum(components.values()),
    "note": "New modules and direct parameters (including token_types) are counted exactly. Vision/language placeholders are excluded. Full parameter count must be audited after loading real HF backbones.",
}
write_json(args.output, report)
print(json.dumps(report, indent=2))
