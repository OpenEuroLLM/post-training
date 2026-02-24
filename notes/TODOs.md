
## TODOs

- [ ] high: Multi-cluster support
  - [ ] ~~Add support for configuring environments in the job script~~
  - [ ] Remove flash attention from pyproject.toml and add instructions for installing from pre-compiled whl files
- [ ] medium: Add configs
  - [ ] SFT/DPO for Olmo 3
- [ ] medium: Add results
  - [ ] SFT/DPO for Olmo 3
- [ ] low: Update README.md on evaluating the checkpoints with `oellm-cli`.

## In Progress
- [ ] medium: Add configs
  - [ ] SFT/DPO for Tulu-3
- [ ] medium: Add results
  - [ ] SFT/DPO for Tulu-3
- [ ] low: Add guardrails to prevent incorrect submissions (and a `--force` flag to override it)

## Done
- [ ] high: Multi-cluster support
  - [X] Add support for downloading the datasets and models before submitting the job for air-gapped HPCs
  - [X] Add support for `module load` and `module purge` in the job script
- [X] high: Add support for epoch-based training
- [X] medium: Add chat template for Tulu-3
