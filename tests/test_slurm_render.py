"""Automated tests for SLURM template rendering."""

import pytest

from post_training.config import PostTrainingConfig
from post_training.slurm.launcher import (
    render_llamafactory_slurm_script,
    render_trl_container_slurm_script,
    render_trl_slurm_script,
)


@pytest.fixture
def config():
    cfg = PostTrainingConfig()
    cfg.container.image = "/shared/containers/trl.sif"
    cfg.container.bind_mounts = ["/scratch:/scratch"]
    cfg.container.env_file = "/shared/env/cluster.env"
    return cfg


# ---------------------------------------------------------------------------
# account field — all three templates
# ---------------------------------------------------------------------------


def test_trl_account_rendered(tmp_path, config):
    """account appears as #SBATCH directive when set."""
    config.slurm.account = "my_project"
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "#SBATCH --account=my_project" in content


def test_trl_account_absent_when_none(tmp_path, config):
    """account directive is suppressed when not set."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "--account" not in content


def test_trl_container_account_rendered(tmp_path, config):
    config.slurm.account = "my_project"
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "#SBATCH --account=my_project" in content


def test_trl_container_account_absent_when_none(tmp_path, config):
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "--account" not in content


def test_llamafactory_account_rendered(tmp_path, config):
    config.slurm.account = "my_project"
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_llamafactory_slurm_script(config, run_dir).read_text()

    assert "#SBATCH --account=my_project" in content


def test_llamafactory_account_absent_when_none(tmp_path, config):
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_llamafactory_slurm_script(config, run_dir).read_text()

    assert "--account" not in content


# ---------------------------------------------------------------------------
# qos / mem fields — LlamaFactory and TRL container templates
# ---------------------------------------------------------------------------


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
