"""RED contract for the evidence-backed VAGEN step60 reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vagen/env/navigation/step60_reconstruction.py"
ENV_CONFIG_PATH = ROOT / "vagen/env/navigation/env_config.py"
ENV_PATH = ROOT / "vagen/env/navigation/env.py"
SERVER_PATH = ROOT / "vagen/server/server.py"
SERVICE_PATH = ROOT / "vagen/env/navigation/service.py"

SYSTEM_SHA256 = "d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a"
INITIAL_SHA256 = "95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2"
STEP_SHA256 = "c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7"
EVIDENCE_FILE_SHA256 = (
    "e9e1ebc4f61b07e5b3b77b165cf72fdfa525d7d840f54296ce5873c5e68463c8"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def reconstruction_module():
    assert MODULE_PATH.is_file(), "step60 reconstruction module is not implemented"
    spec = importlib.util.spec_from_file_location(
        "step60_reconstruction_under_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_navigation_env_class(monkeypatch, reconstruction_module):
    class BaseEnv:
        pass

    def identity_wrapper(function):
        return function

    modules = {
        "vagen": types.ModuleType("vagen"),
        "vagen.env": types.ModuleType("vagen.env"),
        "vagen.env.base": types.ModuleType("vagen.env.base"),
        "vagen.env.base.base_env": types.ModuleType("vagen.env.base.base_env"),
        "vagen.env.utils": types.ModuleType("vagen.env.utils"),
        "vagen.env.utils.context_utils": types.ModuleType(
            "vagen.env.utils.context_utils"
        ),
        "vagen.env.utils.parse_utils": types.ModuleType("vagen.env.utils.parse_utils"),
        "vagen.env.utils.state_reward_text_utils": types.ModuleType(
            "vagen.env.utils.state_reward_text_utils"
        ),
        "vagen.env.navigation": types.ModuleType("vagen.env.navigation"),
        "vagen.env.navigation.env_config": types.ModuleType(
            "vagen.env.navigation.env_config"
        ),
        "vagen.env.navigation.prompt": types.ModuleType("vagen.env.navigation.prompt"),
        "vagen.env.navigation.step60_reconstruction": reconstruction_module,
        "ai2thor": types.ModuleType("ai2thor"),
        "ai2thor.controller": types.ModuleType("ai2thor.controller"),
        "ai2thor.platform": types.ModuleType("ai2thor.platform"),
    }
    modules["vagen.env.base.base_env"].BaseEnv = BaseEnv
    modules["vagen.env.utils.context_utils"].convert_numpy_to_PIL = lambda value: value
    modules["vagen.env.utils.parse_utils"].PARSE_FUNC_MAP = {}
    modules["vagen.env.utils.state_reward_text_utils"].env_state_reward_wrapper = (
        identity_wrapper
    )
    modules["vagen.env.navigation.env_config"].NavigationEnvConfig = object
    prompt = modules["vagen.env.navigation.prompt"]
    prompt.HLIGB_SINGLE_ACTION_MODE = "hligb"
    prompt.SOURCE_EVAL_MODE = "source"
    prompt.action_template = lambda **kwargs: kwargs
    prompt.format_prompt = {}
    prompt.init_observation_template = lambda **kwargs: kwargs
    prompt.system_prompt = lambda **kwargs: kwargs
    modules["ai2thor.platform"].CloudRendering = object
    modules["ai2thor"].controller = modules["ai2thor.controller"]
    modules["ai2thor.controller"].Controller = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "vagen.env.navigation.env_test",
        ENV_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NavigationEnv


def _response(answer: str, *, suffix: str = "") -> str:
    return (
        "<think><observation>OBSERVATION_PAYLOAD</observation>"
        "<reasoning>REASONING_PAYLOAD</reasoning>"
        "<prediction>PREDICTION_PAYLOAD</prediction></think>"
        f"<answer>{answer}</answer>{suffix}"
    )


def test_step60_prompt_fixture_matches_archived_hashes(
    reconstruction_module,
) -> None:
    evidence_path = (
        ROOT
        / "vagen/env/navigation/evidence/vagen_step60_reconstruction_evidence_v1.json"
    )
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == (
        EVIDENCE_FILE_SHA256
    )
    assert _sha256(reconstruction_module.render_system_prompt()) == SYSTEM_SHA256
    initial = reconstruction_module.render_initial_prompt(
        observation="<image>",
        instruction="<INSTRUCTION>",
    )
    assert _sha256(initial) == INITIAL_SHA256
    step = reconstruction_module.render_step_prompt(
        extracted_actions="<ACTIONS>",
        env_feedback="<FEEDBACK>",
        reward="<REWARD>",
        done="<DONE>",
        observation="<image>",
        instruction="<INSTRUCTION>",
    )
    assert _sha256(step.rstrip()) == STEP_SHA256


def test_step60_strict_parser_and_reward_classes(reconstruction_module) -> None:
    classify_response = reconstruction_module.classify_response
    turn_reward = reconstruction_module.turn_reward
    valid = classify_response(_response("moveahead"), max_actions=1)
    assert valid == {
        "actions": ["moveahead"],
        "format_correct": True,
        "error_kind": "ok",
    }
    assert turn_reward(valid, success=False) == pytest.approx(0.02)
    assert turn_reward(valid, success=True) == pytest.approx(10.02)
    adapter = reconstruction_module.parse_response(
        _response("moveahead"),
        special_token_list=["<think>", "</think>", "<answer>", "</answer>"],
        action_sep=",",
        max_actions=1,
    )
    assert adapter["actions"] == ["moveahead"]

    for answer in ("stay", "stop", "rotatelleft"):
        invalid = classify_response(_response(answer), max_actions=1)
        assert invalid["actions"] == []
        assert invalid["format_correct"] is False
        assert invalid["error_kind"] == "invalid_action_name"
        assert turn_reward(invalid, success=False) == pytest.approx(-0.2)

    assert classify_response(
        _response(" moveahead "), max_actions=1
    )["format_correct"] is False
    empty_observation = _response("moveahead").replace(
        "OBSERVATION_PAYLOAD", ""
    )
    assert classify_response(
        empty_observation, max_actions=1
    )["format_correct"] is False

    for malformed_answer in ("moveahead,", ",moveahead", "moveahead,,", "moveahead, "):
        assert classify_response(
            _response(malformed_answer), max_actions=1
        )["format_correct"] is False

    malformed = classify_response(
        _response("moveahead", suffix="EXTRA_OUTPUT"),
        max_actions=1,
    )
    assert malformed["actions"] == []
    assert malformed["format_correct"] is False
    assert malformed["error_kind"] == "missing_or_malformed_tags"
    assert turn_reward(malformed, success=False) == pytest.approx(-0.2)

    too_many = classify_response(
        _response("moveback,moveright"),
        max_actions=1,
    )
    assert too_many == {
        "actions": [],
        "format_correct": False,
        "error_kind": "too_many_actions",
    }
    assert turn_reward(too_many, success=False) == pytest.approx(0.0)


def test_step60_valid_physical_failure_keeps_format_reward(
    reconstruction_module,
) -> None:
    parsed = reconstruction_module.classify_response(
        _response("moveleft"),
        max_actions=1,
    )
    assert parsed["actions"] == ["moveleft"]
    assert reconstruction_module.turn_reward(parsed, success=False) == pytest.approx(
        0.02
    )


@pytest.mark.parametrize(
    ("response", "success", "expected_actions", "expected_reward", "expected_done"),
    [
        (_response("moveahead"), False, [1], 0.02, False),
        (_response("moveahead"), True, [1], 10.02, True),
        (_response("stay"), False, [], -0.2, False),
        (_response("moveback,moveright"), False, [], 0.0, False),
    ],
)
def test_navigation_env_step_executes_only_strict_valid_actions(
    monkeypatch,
    reconstruction_module,
    response,
    success,
    expected_actions,
    expected_reward,
    expected_done,
) -> None:
    NavigationEnv = _load_navigation_env_class(monkeypatch, reconstruction_module)

    class Config:
        success_reward = 10.0
        format_reward = 0.02

        @staticmethod
        def get(key, default=None):
            values = {
                "special_token_list": ["<think>", "</think>"],
                "action_sep": ",",
                "max_actions_per_step": 1,
            }
            return values.get(key, default)

    event = types.SimpleNamespace(
        metadata={
            "agent": {"position": {"x": 0.0, "z": 0.0}},
            "lastActionSuccess": not expected_actions,
        }
    )
    env = NavigationEnv.__new__(NavigationEnv)
    env.config = Config()
    env.parse_func = reconstruction_module.parse_response
    env.action_lookup = {
        name: index
        for index, name in enumerate(reconstruction_module.SOURCE_ACTION_NAMES, start=1)
    }
    env.step60_reconstruction = True
    env.source_eval_compat = False
    env.valid_actions = []
    env.reward = 0.0
    env.total_reward = 0.0
    env._current_step = 0
    env._max_episode_steps = 30
    env._episode_start_time = 0.0
    env.episode_language_instruction = "instruction"
    env.env = types.SimpleNamespace(last_event=event)
    executed = []
    env._execute_action = executed.append
    env.measure_success = lambda: (float(success), 0.5 if success else 2.0)
    env._render = lambda init_obs=False: {"init_obs": init_obs}

    observation, reward, done, info = env.step(response)
    assert observation == {"init_obs": False}
    assert executed == expected_actions
    assert reward == pytest.approx(expected_reward)
    assert done is expected_done
    assert info["last_action_success"] is (not expected_actions)
    assert info["metrics"]["traj_metrics"]["success"] is success


def test_step60_config_action_order_and_env_mode_are_explicit(
    reconstruction_module,
) -> None:
    assert reconstruction_module.STEP60_RECONSTRUCTION_MODE == (
        "step60_source_reconstruction"
    )
    reconstruction_module.validate_environment_config(
        {
            "render_mode": "vision",
            "prompt_format": "step60_source_reconstruction",
            "use_state_reward": False,
            "max_actions_per_step": 1,
            "format_reward": 0.02,
            "invalid_action_penalty": -0.2,
            "success_threshold": 1.5,
            "step_length": 0.5,
            "success_reward": 10.0,
            "eval_set": "base",
            "gpu_device": 0,
        }
    )
    assert list(reconstruction_module.SOURCE_ACTION_NAMES) == [
        "moveahead",
        "moveback",
        "moveright",
        "moveleft",
        "rotateright",
        "rotateleft",
        "lookup",
        "lookdown",
    ]
    config_source = ENV_CONFIG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "invalid_action_penalty: float = -0.2" in config_source
    assert "STEP60_RECONSTRUCTION_MODE" in env_source
    assert "parse_step60_response" in env_source
    assert "turn_reward" in env_source


def test_reconstruction_environment_assets_are_hash_bound() -> None:
    datasets = ROOT / "vagen/env/navigation/datasets"
    expected = {
        "base.json": "6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a",
        "common_sense.json": "3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf",
    }
    loaded = {}
    for filename, digest in expected.items():
        path = datasets / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        loaded[filename.removesuffix(".json")] = json.loads(
            path.read_text(encoding="utf-8")
        )["tasks"]
        assert len(loaded[filename.removesuffix(".json")]) == 60

    samples = [
        ("base", 27670, "navigate to the GarbageCan in the room and be as close as possible to it"),
        ("base", 12623, "navigate to the DeskLamp in the room and be as close as possible to it"),
        ("base", 24836, "navigate to the DeskLamp in the room and be as close as possible to it"),
        ("base", 9877, "navigate to the GarbageCan in the room and be as close as possible to it"),
        ("base", 13671, "navigate to the Laptop in the room and be as close as possible to it"),
        ("base", 23861, "navigate to the AlarmClock in the room and be as close as possible to it"),
        ("base", 17345, "navigate to the Cup in the room and be as close as possible to it"),
        ("common_sense", 27670, "I have some rubbish that needs to be properly disposed of. Can you navigate to that object and stay close?"),
        ("common_sense", 12623, "I need a desk-mounted lighting device to illuminate my work area. Please navigate to that object and stay near it."),
        ("common_sense", 23861, "I need a device to set an alarm for waking up at a specific time. Please navigate to that object and stay near it."),
        ("common_sense", 795, "I'm looking to braise some meat and need a deep cooking vessel. Please navigate to that object and stay near it."),
    ]
    for eval_set, seed, instruction in samples:
        assert loaded[eval_set][seed % 60]["instruction"] == instruction


def test_service_identity_recomputes_git_assets_config_and_routes(
    monkeypatch,
    reconstruction_module,
) -> None:
    base = reconstruction_module.RECONSTRUCTION_BASE_COMMIT
    head = "a" * 40
    tree = "b" * 40

    def fake_check_output(command, text=False):
        joined = " ".join(command)
        if "status --porcelain" in joined:
            return "" if text else b""
        if "rev-parse HEAD^{tree}" in joined:
            return f"{tree}\n" if text else f"{tree}\n".encode()
        if "rev-parse HEAD^" in joined:
            return f"{base}\n" if text else f"{base}\n".encode()
        if "rev-parse HEAD" in joined:
            return f"{head}\n" if text else f"{head}\n".encode()
        if "rev-list --count" in joined:
            return "1\n" if text else b"1\n"
        if "rev-list --parents" in joined:
            value = f"{head} {base}\n"
            return value if text else value.encode()
        if command[:2] == ["git", "--version"]:
            return "git version test\n" if text else b"git version test\n"
        if "diff --binary" in joined:
            return b"canonical-diff"
        raise AssertionError(command)

    monkeypatch.setattr(
        reconstruction_module.subprocess,
        "check_output",
        fake_check_output,
    )
    routes = [
        "/batch/close",
        "/batch/reset",
        "/batch/reward",
        "/batch/step",
        "/batch/system_prompt",
        "/close/<env_id>",
        "/environments",
        "/health",
        "/reconstruction/identity",
        "/reset/<env_id>",
        "/reward/<env_id>",
        "/step/<env_id>",
        "/system_prompt/<env_id>",
    ]
    identity = reconstruction_module.service_runtime_identity(
        service_routes=routes
    )
    assert identity["reconstruction_identity"]["runtime_head"] == head
    assert identity["reconstruction_identity"]["runtime_parent"] == base
    assert identity["reconstruction_identity"]["runtime_tree"] == tree
    assert identity["environment_assets"]["base"]["rows"] == 60
    assert identity["environment_config"]["step_length"] == 0.5
    assert identity["service_routes"] == routes


def test_navigation_service_creation_failure_closes_late_successes(
    monkeypatch,
    reconstruction_module,
) -> None:
    created = []

    class BaseService:
        pass

    class Config:
        def __init__(self, **values):
            self.__dict__.update(values)

    class FakeNavigationEnv:
        def __init__(self, config):
            if config.eval_set == "common_sense":
                raise RuntimeError("intentional constructor failure")
            time.sleep(0.05)
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    modules = {
        "vagen": types.ModuleType("vagen"),
        "vagen.env": types.ModuleType("vagen.env"),
        "vagen.env.base": types.ModuleType("vagen.env.base"),
        "vagen.env.utils": types.ModuleType("vagen.env.utils"),
        "vagen.env.navigation": types.ModuleType("vagen.env.navigation"),
        "vagen.env.base.base_service": types.ModuleType(
            "vagen.env.base.base_service"
        ),
        "vagen.env.navigation.env": types.ModuleType(
            "vagen.env.navigation.env"
        ),
        "vagen.env.navigation.env_config": types.ModuleType(
            "vagen.env.navigation.env_config"
        ),
        "vagen.server": types.ModuleType("vagen.server"),
        "vagen.server.serial": types.ModuleType("vagen.server.serial"),
        "vagen.env.navigation.service_config": types.ModuleType(
            "vagen.env.navigation.service_config"
        ),
        "vagen.env.utils.state_reward_text_utils": types.ModuleType(
            "vagen.env.utils.state_reward_text_utils"
        ),
        "vagen.env.navigation.step60_reconstruction": reconstruction_module,
    }
    modules["vagen.env.base.base_service"].BaseService = BaseService
    modules["vagen.env.navigation.env"].NavigationEnv = FakeNavigationEnv
    modules["vagen.env.navigation.env_config"].NavigationEnvConfig = Config
    modules["vagen.server.serial"].serialize_observation = lambda value: value
    modules["vagen.env.navigation.service_config"].NavigationServiceConfig = object
    modules[
        "vagen.env.utils.state_reward_text_utils"
    ].service_state_reward_wrapper = lambda function: function
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "vagen.env.navigation.service_test",
        SERVICE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service = module.NavigationService(
        types.SimpleNamespace(max_workers=2, devices=[0, 1])
    )
    base_config = {
        "render_mode": "vision",
        "prompt_format": "step60_source_reconstruction",
        "use_state_reward": False,
        "max_actions_per_step": 1,
        "format_reward": 0.02,
        "invalid_action_penalty": -0.2,
        "success_threshold": 1.5,
        "step_length": 0.5,
        "success_reward": 10.0,
    }
    with pytest.raises(RuntimeError, match="batch creation failed"):
        service.create_environments_batch(
            {
                "ok": {
                    "env_name": "navigation",
                    "env_config": {**base_config, "eval_set": "base"},
                },
                "fail": {
                    "env_name": "navigation",
                    "env_config": {**base_config, "eval_set": "common_sense"},
                },
            }
        )
    assert len(created) == 1
    assert created[0].closed is True
    assert service.environments == {}
    assert service.env_configs == {}
    assert all(not assigned for assigned in service.device_status.values())


def test_navigation_service_releases_environment_gpu_assignment() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert "assigned_env_ids.discard(env_id)" in source
    assert "self.device_status.pop(env_id" not in source


def test_legacy_batch_api_routes_remain_available() -> None:
    source = SERVER_PATH.read_text(encoding="utf-8")
    for route in (
        "/health",
        "/reconstruction/identity",
        "/environments",
        "/batch/reset",
        "/batch/step",
        "/batch/reward",
        "/batch/system_prompt",
        "/batch/close",
    ):
        assert repr(route) in source
