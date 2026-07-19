from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_attach_validation_metadata_uses_env_identity_not_return_order():
    from vagen.trainer.validation_identity import attach_validation_input_metadata

    records = [
        {"env_id": "val2", "metrics": {"success": True}},
        {"env_id": "val1", "metrics": {"success": False}},
    ]
    env_configs = [
        {"seed": 11, "env_config": {"eval_set": "base_train"}},
        {"seed": 22, "env_config": {"eval_set": "common_sense_train"}},
    ]
    uids = ["base:11", "common:22"]
    sources = ["base", "common"]

    def metadata(env_config, uid, source):
        return {
            "env_seed": env_config["seed"],
            "eval_set": env_config["env_config"]["eval_set"],
            "uid": uid,
            "data_source": source,
        }

    attached = attach_validation_input_metadata(
        records,
        env_configs=env_configs,
        uids=uids,
        sources=sources,
        input_index_by_env_id={"val1": 0, "val2": 1},
        metadata_fn=metadata,
    )

    assert attached is records
    assert attached[0]["env_seed"] == 22
    assert attached[0]["data_source"] == "common"
    assert attached[1]["env_seed"] == 11
    assert attached[1]["data_source"] == "base"


def test_attach_validation_metadata_rejects_incomplete_identity_map():
    from vagen.trainer.validation_identity import attach_validation_input_metadata

    with pytest.raises(ValueError, match="missing stable input identity"):
        attach_validation_input_metadata(
            [{"env_id": "val2"}],
            env_configs=[{"seed": 1}],
            uids=["uid1"],
            sources=["source1"],
            input_index_by_env_id={"val1": 0},
            metadata_fn=lambda *_: {},
        )


def test_service_reset_tracks_reused_env_identity(monkeypatch):
    from vagen.rollout.qwen_rollout import rollout_manager_service as module

    class FakeConfig:
        def __init__(self, kind):
            self.kind = kind

        def config_id(self):
            return self.kind

        def get(self, key, default=None):
            return default

    class FakeClient:
        def close_batch(self, env_ids):
            assert env_ids == []

        def create_environments_batch(self, configs):
            assert configs == {}

        def reset_batch(self, seeds):
            return {
                env_id: ({"obs_str": f"seed={seed}"}, {})
                for env_id, seed in reversed(list(seeds.items()))
            }

        def get_system_prompts_batch(self, env_ids):
            return {env_id: "system" for env_id in env_ids}

    monkeypatch.setattr(
        module,
        "REGISTERED_ENV",
        {"fake": {"config_cls": lambda kind: FakeConfig(kind)}},
    )
    manager = module.QwenVLRolloutManagerService.__new__(
        module.QwenVLRolloutManagerService
    )
    manager.envs = {"val7": FakeConfig("A"), "val9": FakeConfig("B")}
    manager.env_client = FakeClient()
    manager.split = "val"
    manager.recorder = None

    manager.reset(
        [
            {"env_name": "fake", "env_config": {"kind": "B"}, "seed": 22},
            {"env_name": "fake", "env_config": {"kind": "A"}, "seed": 11},
        ]
    )

    assert manager.input_index_by_env_id == {"val9": 0, "val7": 1}
    assert manager.recorder["val9"][0]["obs_str"] == "seed=22"
    assert manager.recorder["val7"][0]["obs_str"] == "seed=11"


def test_both_rollout_managers_record_reset_input_identity():
    for relative_path in (
        "vagen/rollout/qwen_rollout/rollout_manager.py",
        "vagen/rollout/qwen_rollout/rollout_manager_service.py",
    ):
        source = (ROOT / relative_path).read_text()
        assert "input_index_by_env_id" in source
        assert "input_index_by_env_id[" in source


def test_trainer_merges_validation_metadata_by_stable_identity():
    source = (ROOT / "vagen/trainer/ppo/ray_trainer.py").read_text()
    assert "attach_validation_input_metadata(" in source
    assert "input_index_by_env_id=self.test_rollout_manager.input_index_by_env_id" in source
    assert "zip(micro_validation_rst, env_configs" not in source
