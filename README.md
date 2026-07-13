# DIAMBRA PPO Agent

[![CI](https://github.com/Ketan-Kapse/diambra_ppo/actions/workflows/ci.yml/badge.svg)](https://github.com/Ketan-Kapse/diambra_ppo/actions/workflows/ci.yml)

A Stable-Baselines3 PPO policy for running **Dead or Alive++** (`doapp`) matches in
[DIAMBRA Arena](https://diambra.ai/). The repository includes the environment configuration,
trained policy checkpoints, and the DIAMBRA submission manifest needed to run the agent.

## What is included

- A configurable PPO inference entrypoint with explicit checkpoint selection.
- Support for stochastic or deterministic policy evaluation.
- A bounded test mode for quick smoke runs.
- Runtime-free validation of the checked-in configuration and checkpoint wiring.
- Configuration and checkpoint validation with actionable error messages.
- Unit tests and CI across Python 3.10 and 3.11.

## Repository layout

| Path | Purpose |
| --- | --- |
| `agent.py` | CLI, DIAMBRA environment setup, model loading, and evaluation loop |
| `config.yaml` | Arena, wrapper, and PPO training configuration |
| `submission-manifest.yaml` | AI-vs-COM DIAMBRA submission definition |
| `model_tuned2.zip` | Checkpoint used by the submission manifest |
| `model_tuned.zip`, `models.zip` | Earlier PPO checkpoints retained for comparison |
| `tests/` | Fast unit tests that do not require the game runtime |

## Runtime

The submission manifest uses DIAMBRA's Python 3.10 Stable-Baselines3 image, which supplies the
game runtime, Stable-Baselines3, and PyTorch. The included checkpoints were created with
Stable-Baselines3 2.1.0.

For a local arena run, install DIAMBRA Arena and its Stable-Baselines3 integration according to
the official DIAMBRA documentation, then install this repository's direct Python dependency:

```bash
python -m pip install -r requirements.txt
```

## Run the agent

The checkpoint can be supplied explicitly:

```bash
python agent.py \
  --cfg-file config.yaml \
  --trained-model model_tuned2.zip
```

When `--trained-model` is omitted, the agent resolves `folders.model_name` from `config.yaml` and
looks for the matching `.zip` file beside the configuration.

Use deterministic actions and stop after one completed episode for an evaluation smoke test:

```bash
python agent.py --deterministic --max-episodes 1
```

Validate the checked-in assets without installing or launching DIAMBRA:

```bash
python agent.py --validate-only
```

`--test` is a backwards-compatible shorthand for a one-episode run. The historical options
`--cfgFile`, `--trainedModel`, and `--trainedmodel` remain supported, while new commands should
prefer the kebab-case names shown above.

## Configuration

The runtime consumes three required YAML sections:

- `folders` identifies the default model checkpoint.
- `settings` configures the game, character, frame shape, difficulty, and action space.
- `wrappers_settings` controls observation and action preprocessing.

Only `discrete` and `multi_discrete` action spaces are accepted. Invalid configuration and missing
checkpoint errors list the exact field or filesystem locations involved.

## DIAMBRA submission

`submission-manifest.yaml` launches the policy in AI-vs-COM mode and explicitly passes
`/sources/model_tuned2.zip`. This keeps the submitted artifact and the checkpoint selected by the
Python entrypoint in sync.

## Development

Install the lightweight development dependencies and run the same checks used by CI:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python agent.py --validate-only
pytest
```

The unit suite covers safe configuration loading, validation failures, model-path inference, CLI
compatibility aliases, deterministic prediction, DIAMBRA match completion, and episode limits.
