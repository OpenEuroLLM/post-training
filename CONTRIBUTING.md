# Contributing

Everyone is welcome to contribute, both in code and discussion. Make sure you read our [design philosophy](https://github.com/OpenEuroLLM/post-training?tab=readme-ov-file#-design-philosophy) first.

## Setup

```bash
uv sync --extra dev --extra trl
uv run pre-commit install
```

## Making changes

- Keep PRs small and focused on a single change.
- Follow existing code style — pre-commit hooks will enforce formatting with ruff and black.

## Submitting a PR

1. Open an issue first for non-trivial changes.
2. Branch off `main`.
3. Use the PR template.
