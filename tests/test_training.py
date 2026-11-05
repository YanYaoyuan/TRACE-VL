from trace_vl.config import ModelConfig
from trace_vl.inference import evaluate
from trace_vl.models.model import TraceVL
from trace_vl.runtime import load_checkpoint
from trace_vl.training import train


def test_warm_start_context_change_resets_steps_and_learning_rate(fixture_manifest, tmp_path):
    import json

    from trace_vl.runtime import initialize_weights

    config = {
        "model": ModelConfig().to_dict(),
        "data": {"manifest": str(fixture_manifest), "max_frames": 2},
        "training": {"device": "cpu", "max_steps": 1, "batch_size": 2, "lr": 1e-4},
    }
    source = train(config, tmp_path / "source")
    model = TraceVL(ModelConfig(spatial_tokens=12))
    info = initialize_weights(model, source)
    assert info["changed_context_settings"] == ["spatial_tokens"]
    assert not info["optimizer_restored"]
    config["model"] = model.cfg.to_dict()
    config["training"].update(lr=3e-5, workers=1)
    result = train(config, tmp_path / "warm", init_from=source)
    assert load_checkpoint(result)["step"] == 1
    run = json.loads((tmp_path / "warm/run.json").read_text())
    assert run["optimizer_learning_rates"] == [3e-5]
    assert run["initialization"]["source_step"] == 1


def test_retained_checkpoints_survive_last_replacement(fixture_manifest, tmp_path):
    import json
    import os

    config = {
        "seed": 7,
        "model": ModelConfig().to_dict(),
        "data": {"manifest": str(fixture_manifest), "max_frames": 2},
        "training": {
            "device": "cpu",
            "max_steps": 3,
            "batch_size": 2,
            "save_every": 3,
            "keep_steps": [1, 3],
        },
    }
    root = tmp_path / "retained"
    train(config, root)
    assert load_checkpoint(root / "step-00001.pt")["step"] == 1
    assert load_checkpoint(root / "step-00003.pt")["step"] == 3
    assert load_checkpoint(root / "last.pt")["step"] == 3
    assert os.path.samefile(root / "last.pt", root / "step-00003.pt")
    assert not os.path.samefile(root / "last.pt", root / "step-00001.pt")
    resources = json.loads((root / "resources.json").read_text())
    assert len(resources["ranks"]) == 1
    assert resources["ranks"][0]["rank"] == 0


def test_training_resume_and_checkpoint_inference(fixture_manifest, tmp_path):
    config = {
        "seed": 7,
        "model": ModelConfig().to_dict(),
        "data": {"manifest": str(fixture_manifest), "max_frames": 4},
        "training": {"device": "cpu", "max_steps": 2, "batch_size": 2},
        "evaluation": {"device": "cpu", "max_new_tokens": 2},
    }
    checkpoint = train(config, tmp_path / "run")
    saved = load_checkpoint(checkpoint)
    assert saved["step"] == 2
    initial = TraceVL(ModelConfig())
    initial.load_state_dict(saved["model"])
    report = evaluate(config, checkpoint, tmp_path / "eval")
    assert report["samples"] == 2
    assert not report["provenance"]["random_weights"]
    assert (tmp_path / "eval" / "predictions.jsonl").is_file()
    config["training"]["max_steps"] = 3
    train(config, tmp_path / "resumed", resume=checkpoint)
    assert load_checkpoint(tmp_path / "resumed" / "last.pt")["step"] == 3
