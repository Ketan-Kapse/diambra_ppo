from __future__ import annotations

from pathlib import Path

import pytest

import agent

VALID_CONFIG = """
folders:
  parent_dir: ./artifacts
  model_name: model_tuned2
settings:
  game_id: doapp
  action_space: multi_discrete
wrappers_settings:
  normalize_reward: false
"""


def write_config(tmp_path: Path, contents: str = VALID_CONFIG) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def test_load_config_accepts_expected_shape(tmp_path: Path) -> None:
    config = agent.load_config(write_config(tmp_path))

    assert config["settings"]["game_id"] == "doapp"
    assert config["settings"]["action_space"] == "multi_discrete"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("[]", "must be a YAML mapping"),
        ("settings: {}", "section 'folders'"),
        (
            VALID_CONFIG.replace("multi_discrete", "continuous"),
            "settings.action_space must be one of",
        ),
    ],
)
def test_load_config_rejects_invalid_configuration(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    with pytest.raises(agent.ConfigurationError, match=message):
        agent.load_config(write_config(tmp_path, contents))


def test_load_config_uses_safe_yaml_loader(tmp_path: Path) -> None:
    malicious = VALID_CONFIG + "payload: !!python/object/apply:os.system ['echo unsafe']\n"

    with pytest.raises(agent.ConfigurationError, match="Unable to load configuration"):
        agent.load_config(write_config(tmp_path, malicious))


def test_resolve_model_path_uses_explicit_checkpoint(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    model_path = tmp_path / "custom.zip"
    model_path.touch()
    config = agent.load_config(config_path)

    assert agent.resolve_model_path("custom.zip", config_path, config) == model_path


def test_resolve_model_path_infers_checkpoint_from_config(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    model_path = tmp_path / "model_tuned2.zip"
    model_path.touch()
    config = agent.load_config(config_path)

    assert agent.resolve_model_path(None, config_path, config) == model_path


def test_resolve_model_path_reports_every_checked_location(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = agent.load_config(config_path)

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found") as error:
        agent.resolve_model_path("missing", config_path, config)

    assert "missing.zip" in str(error.value)


def test_validate_only_checks_real_assets_without_runtime_dependencies(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    (tmp_path / "model_tuned2.zip").touch()

    result = agent.main(config_path, validate_only=True)

    assert result == 0


@pytest.mark.parametrize("option", ["--trained-model", "--trainedModel", "--trainedmodel"])
def test_parser_accepts_checkpoint_option_aliases(option: str) -> None:
    options = agent.build_parser().parse_args([option, "policy.zip", "--test", "1"])

    assert options.trained_model == "policy.zip"
    assert options.test is True


def test_parser_accepts_validate_only() -> None:
    options = agent.build_parser().parse_args(["--validate-only"])

    assert options.validate_only is True


class FakeAction:
    def tolist(self) -> list[int]:
        return [1, 2]


class FakePolicy:
    def __init__(self) -> None:
        self.deterministic_values: list[bool] = []

    def predict(self, observation: str, *, deterministic: bool) -> tuple[FakeAction, None]:
        assert observation == "observation"
        self.deterministic_values.append(deterministic)
        return FakeAction(), None


class FakeEnvironment:
    def __init__(self, reset_info: list[dict[str, bool]]) -> None:
        self.reset_info = iter(reset_info)
        self.actions: list[list[int]] = []

    def reset(self) -> tuple[str, dict[str, bool]]:
        return "observation", next(self.reset_info)

    def step(self, action: list[int]) -> tuple[str, float, bool, bool, dict[str, bool]]:
        self.actions.append(action)
        return "observation", 1.0, True, False, {}


def test_play_agent_stops_when_environment_reports_match_complete() -> None:
    policy = FakePolicy()
    environment = FakeEnvironment([{}, {}, {"env_done": True}])

    completed = agent.play_agent(policy, environment, deterministic=True)

    assert completed == 2
    assert environment.actions == [[1, 2], [1, 2]]
    assert policy.deterministic_values == [True, True]


def test_play_agent_honors_episode_limit() -> None:
    policy = FakePolicy()
    environment = FakeEnvironment([{}, {}])

    completed = agent.play_agent(policy, environment, max_episodes=1)

    assert completed == 1
