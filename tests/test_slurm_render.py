"""Automated tests for SLURM template rendering."""

import pytest

from post_training.config import PostTrainingConfig
from post_training.slurm.launcher import (
    render_llamafactory_slurm_script,
    render_trl_container_slurm_script,
)


@pytest.fixture
def config():
    cfg = PostTrainingConfig()
    cfg.container.image = "/shared/containers/llamafactory.sif"
    cfg.container.bind_mounts = ["/scratch:/scratch"]
    cfg.container.env_file = "/shared/env/cluster.env"
    return cfg


def test_llamafactory_qos_mem_rendered(tmp_path, config):
    """qos and mem appear as #SBATCH directives when set."""
    config.slurm.qos = "boost_qos_dbg"
    config.slurm.mem = "64G"
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_llamafactory_slurm_script(config, run_dir).read_text()

    assert "#SBATCH --qos=boost_qos_dbg" in content
    assert "#SBATCH --mem=64G" in content


def test_llamafactory_qos_mem_absent_when_none(tmp_path, config):
    """qos and mem directives are suppressed when not set."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_llamafactory_slurm_script(config, run_dir).read_text()

    assert "--qos" not in content
    assert "--mem" not in content


# ---------------------------------------------------------------------------
# TRL container template
# ---------------------------------------------------------------------------


def test_trl_container_qos_mem_rendered(tmp_path, config):
    """qos and mem appear as #SBATCH directives when set."""
    config.slurm.qos = "boost_qos_dbg"
    config.slurm.mem = "64G"
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "#SBATCH --qos=boost_qos_dbg" in content
    assert "#SBATCH --mem=64G" in content


def test_trl_container_qos_mem_absent_when_none(tmp_path, config):
    """qos and mem directives are suppressed when not set."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "--qos" not in content
    assert "--mem" not in content
