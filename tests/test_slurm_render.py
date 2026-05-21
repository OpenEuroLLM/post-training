"""Automated tests for SLURM template rendering."""

import pytest

from post_training.backend import LlamaFactoryBackend, TRLBackend
from post_training.config import ContainerConfig, PostTrainingConfig
from post_training.slurm.launcher import (
    render_llamafactory_slurm_script,
    render_trl_container_slurm_script,
    render_trl_slurm_script,
)


@pytest.fixture
def config():
    cfg = PostTrainingConfig()
    cfg.container = ContainerConfig(
        image="/shared/containers/trl.sif",
        bind_mounts=["/scratch:/scratch"],
        env_file="/shared/env/cluster.env",
    )
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


# ---------------------------------------------------------------------------
# --tokenize-only forwarding — TRL templates and backend dispatch
# ---------------------------------------------------------------------------


def _train_invocation(content: str) -> str:
    """Return the rendered line that invokes scripts/train.py."""
    for line in content.splitlines():
        if "scripts/train.py" in line:
            return line
    raise AssertionError("no scripts/train.py invocation found in rendered script")


def test_trl_tokenize_only_appended(tmp_path, config):
    """--tokenize-only is appended to the train.py invocation when requested."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_slurm_script(
        config, run_dir, "configs/trl/sft.yaml", tokenize_only=True
    ).read_text()

    assert "--tokenize-only" in _train_invocation(content)


def test_trl_tokenize_only_absent_by_default(tmp_path, config):
    """No --tokenize-only flag is emitted when the kwarg is omitted."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "--tokenize-only" not in content


def test_trl_container_tokenize_only_appended(tmp_path, config):
    """--tokenize-only is appended in the containerized TRL template."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(
        config, run_dir, "configs/trl/sft.yaml", tokenize_only=True
    ).read_text()

    assert "--tokenize-only" in _train_invocation(content)


def test_trl_container_tokenize_only_absent_by_default(tmp_path, config):
    """No --tokenize-only flag in the containerized TRL template by default."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    content = render_trl_container_slurm_script(config, run_dir, "configs/trl/sft.yaml").read_text()

    assert "--tokenize-only" not in content


def test_trl_backend_forwards_tokenize_only_non_container(tmp_path, config):
    """TRLBackend.render_slurm_script forwards the flag on the bare-metal path."""
    config.container = None  # force non-container branch
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    script = TRLBackend().render_slurm_script(
        config, run_dir, "configs/trl/sft.yaml", tokenize_only=True
    )

    assert "--tokenize-only" in _train_invocation(script.read_text())


def test_trl_backend_treats_container_null_as_bare_metal(tmp_path):
    """container: null uses the non-container TRL template."""
    cfg = PostTrainingConfig()
    cfg.container = None
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    script = TRLBackend().render_slurm_script(cfg, run_dir, "configs/trl/sft.yaml")
    content = script.read_text()

    assert "singularity exec" not in content
    assert "accelerate launch" in content


def test_trl_backend_forwards_tokenize_only_container(tmp_path, config):
    """TRLBackend.render_slurm_script forwards the flag on the container path."""
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    script = TRLBackend().render_slurm_script(
        config, run_dir, "configs/trl/sft.yaml", tokenize_only=True
    )

    assert "--tokenize-only" in _train_invocation(script.read_text())


def test_llamafactory_backend_ignores_tokenize_only(tmp_path, config):
    """LlamaFactoryBackend accepts the kwarg silently and never emits --tokenize-only.

    submit.py raises before reaching this point, but the backend itself
    must not crash if a caller passes the kwarg.
    """
    run_dir = tmp_path / "outputs" / "my-run"
    run_dir.mkdir(parents=True)

    script = LlamaFactoryBackend().render_slurm_script(
        config, run_dir, "configs/llamafactory/long-context.yaml", tokenize_only=True
    )

    assert "--tokenize-only" not in script.read_text()
