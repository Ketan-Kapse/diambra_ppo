"""Run a trained Stable-Baselines3 PPO policy in DIAMBRA Arena."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ACTION_SPACES = {"discrete", "multi_discrete"}


class ConfigurationError(ValueError):
    """Raised when the agent configuration is missing or invalid."""


def load_config(cfg_file: str | Path) -> dict[str, Any]:
    """Load and validate the YAML configuration used to build the arena."""
    config_path = Path(cfg_file).expanduser()
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to load configuration '{config_path}': {exc}") from exc

    if not isinstance(config, Mapping):
        raise ConfigurationError("Configuration must be a YAML mapping.")

    for section in ("folders", "settings", "wrappers_settings"):
        if not isinstance(config.get(section), Mapping):
            raise ConfigurationError(f"Configuration section '{section}' must be a mapping.")

    settings = config["settings"]
    game_id = settings.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise ConfigurationError("settings.game_id must be a non-empty string.")

    action_space = settings.get("action_space")
    if action_space not in SUPPORTED_ACTION_SPACES:
        choices = ", ".join(sorted(SUPPORTED_ACTION_SPACES))
        raise ConfigurationError(f"settings.action_space must be one of: {choices}.")

    return dict(config)


def resolve_model_path(
    trained_model: str | Path | None,
    cfg_file: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Resolve an explicit checkpoint or infer one from folders.model_name."""
    config_path = Path(cfg_file).expanduser().resolve()
    folders = config["folders"]

    if trained_model is None:
        model_name = folders.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ConfigurationError(
                "folders.model_name must be set when --trained-model is omitted."
            )
        requested = Path(model_name)
    else:
        requested = Path(trained_model).expanduser()

    requested_variants = [requested]
    if requested.suffix == "":
        requested_variants.append(requested.with_suffix(".zip"))

    candidates: list[Path] = []
    for variant in requested_variants:
        if variant.is_absolute():
            candidates.append(variant)
        else:
            candidates.extend((config_path.parent / variant, Path.cwd() / variant))

            parent_dir = Path(str(folders.get("parent_dir", ".")))
            game_id = str(config["settings"]["game_id"])
            model_name = str(folders.get("model_name", ""))
            candidates.append(
                config_path.parent / parent_dir / game_id / model_name / "model" / variant
            )

    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    checked = "\n  - ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Model checkpoint not found. Checked:\n  - {checked}")


def play_agent(
    agent: Any,
    env: Any,
    *,
    deterministic: bool = False,
    max_episodes: int | None = None,
) -> int:
    """Play until DIAMBRA reports match completion or an episode limit is reached."""
    observation, _ = env.reset()
    completed_episodes = 0

    while True:
        action, _ = agent.predict(observation, deterministic=deterministic)
        env_action = action.tolist() if hasattr(action, "tolist") else action
        observation, _, terminated, truncated, _ = env.step(env_action)

        if not (terminated or truncated):
            continue

        completed_episodes += 1
        observation, reset_info = env.reset()
        match_complete = isinstance(reset_info, Mapping) and bool(reset_info.get("env_done"))
        limit_reached = max_episodes is not None and completed_episodes >= max_episodes
        if match_complete or limit_reached:
            return completed_episodes


def main(
    cfg_file: str | Path,
    trained_model: str | Path | None = None,
    test: bool = False,
    deterministic: bool = False,
    max_episodes: int | None = None,
    validate_only: bool = False,
) -> int:
    """Build the DIAMBRA environment, load the PPO policy, and run evaluation."""
    config = load_config(cfg_file)
    model_path = resolve_model_path(trained_model, cfg_file, config)
    print("Config parameters =", json.dumps(config, sort_keys=True, indent=4))
    print(f"Loading PPO checkpoint: {model_path}")

    if validate_only:
        print("Configuration and checkpoint validation passed.")
        return 0

    from diambra.arena import Roles, SpaceTypes, load_settings_flat_dict
    from diambra.arena.stable_baselines3.make_sb3_env import (
        EnvironmentSettings,
        WrappersSettings,
        make_sb3_env,
    )
    from stable_baselines3 import PPO

    settings_params = dict(config["settings"])
    if isinstance(settings_params.get("frame_shape"), list):
        settings_params["frame_shape"] = tuple(settings_params["frame_shape"])
    settings_params["action_space"] = {
        "discrete": SpaceTypes.DISCRETE,
        "multi_discrete": SpaceTypes.MULTI_DISCRETE,
    }[settings_params["action_space"]]

    settings = load_settings_flat_dict(EnvironmentSettings, settings_params)
    settings.role = Roles.P1
    wrappers_settings = load_settings_flat_dict(
        WrappersSettings, dict(config["wrappers_settings"])
    )

    env = None
    try:
        env, num_envs = make_sb3_env(
            settings.game_id,
            settings,
            wrappers_settings,
            no_vec=True,
        )
        print(f"Activated {num_envs} environment(s)")

        agent = PPO.load(str(model_path), env=env)
        print("Policy architecture:")
        print(agent.policy)

        episode_limit = 1 if test and max_episodes is None else max_episodes
        completed = play_agent(
            agent,
            env,
            deterministic=deterministic,
            max_episodes=episode_limit,
        )
        print(f"Completed {completed} episode(s)")
    finally:
        if env is not None:
            env.close()

    return 0


def parse_bool(value: str | bool) -> bool:
    """Parse common command-line boolean representations."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value}")


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser, including legacy option aliases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg-file",
        "--cfgFile",
        dest="cfg_file",
        default="config.yaml",
        help="Path to the YAML environment configuration.",
    )
    parser.add_argument(
        "--trained-model",
        "--trainedModel",
        "--trainedmodel",
        dest="trained_model",
        help="Path to a Stable-Baselines3 PPO checkpoint. Inferred when omitted.",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Stop after one episode. An optional true/false value is accepted.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Choose the highest-probability policy action during evaluation.",
    )
    parser.add_argument(
        "--max-episodes",
        type=positive_int,
        help="Stop after this many completed episodes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the configuration and checkpoint without launching DIAMBRA.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and execute the agent."""
    parser = build_parser()
    options = parser.parse_args(argv)
    try:
        return main(
            cfg_file=options.cfg_file,
            trained_model=options.trained_model,
            test=options.test,
            deterministic=options.deterministic,
            max_episodes=options.max_episodes,
            validate_only=options.validate_only,
        )
    except (ConfigurationError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(cli())
