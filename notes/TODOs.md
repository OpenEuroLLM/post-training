# TODOs

- [ ] high: Add support for epoch-based training
- [ ] high: Multi-cluster support
  - [ ] Add support for downloading the datasets and models before submitting the job for air-gapped HPCs
  - [ ] Add support for `module load` and `module purge` in the job script
  - [ ] Add support for configuring  environments in the job script
  - [ ] Remove flash attention from pyproject.toml and add instructions for installing from pre-compiled whl files
- [ ] medium: Add chat template for Tulu-3
- [ ] medium: Add configs
  - [ ] SFT/DPO for Tulu-3
  - [ ] SFT/DPO for Olmo 3
- [ ] medium: Add results
  - [ ] SFT/DPO for Tulu-3
  - [ ] SFT/DPO for Olmo 3
- [ ] low: Update README.md on evaluating the checkpoints with `oellm-cli`.
- [ ] low: Add guardrails to prevent incorrect submissions (and a `--force` flag to override it)
