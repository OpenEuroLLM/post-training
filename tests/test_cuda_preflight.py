"""Test CUDA startup decisions without requiring a GPU."""

import os
import subprocess

import pytest

from post_training.config import ContainerConfig, PostTrainingConfig
from post_training.slurm.launcher import render_trl_container_slurm_script


@pytest.mark.parametrize("mode,probes,status", [("default", 1, 0), ("fallback", 2, 0), ("fail", 2, 1)])
def test_cuda_preflight(tmp_path, mode, probes, status):
    config = PostTrainingConfig()
    config.container = ContainerConfig(image="/shared/training.sif")
    content = render_trl_container_slurm_script(config, tmp_path, "config.yaml").read_text()
    subprocess.run(["bash", "-n"], input=content, text=True, check=True)
    start = content.index("    # Keep working defaults;")
    end = content.index("    export NODE_RANK", start)
    block = content[start:end].replace('\\"', '"').replace("\\$", "$")
    assert content.index("scripts/check_cuda.py") < content.index("accelerate launch")
    mock = '''
python() {
    printf 'PROBE:%s\\n' "$LD_LIBRARY_PATH"
    case "$MODE" in
        default) return 0 ;;
        fallback) [[ "$LD_LIBRARY_PATH" != *'/usr/local/cuda/compat'* ]] ;;
        fail) return 1 ;;
    esac
}
'''
    original = "/host/lib:/usr/local/cuda/compat:/usr/local/cuda/compat/lib:/other/lib"
    result = subprocess.run(
        ["bash", "-c", mock + block + '\nprintf "FINAL:%s\\n" "$LD_LIBRARY_PATH"'],
        env={**os.environ, "MODE": mode, "LD_LIBRARY_PATH": original},
        text=True, capture_output=True,
    )
    assert result.returncode == status
    assert result.stdout.count("PROBE:") == probes
    if mode == "default":
        assert f"FINAL:{original}" in result.stdout
    elif mode == "fallback":
        assert "FINAL:/host/lib:/other/lib" in result.stdout
    else:
        assert "FINAL:" not in result.stdout
