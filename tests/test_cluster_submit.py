"""Tests for cluster submission backend (dry-run; no real SSH)."""
from __future__ import annotations

import importlib

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


def test_write_run_config_injects_point_sources(tmp_path) -> None:
    import yaml
    out = tmp_path / "run.yaml"
    cluster.write_run_config(
        [{"x": 1.0, "y": 2.0, "z": 3.0, "strength_nAm": 70}], out_path=out,
    )
    raw = yaml.safe_load(out.read_text())
    assert raw["forward"]["point_sources"] == [[1.0, 2.0, 3.0]]


def test_submit_dryrun(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INOB_CLUSTER_DRYRUN", "1")
    # keep the working-config write out of the repo
    monkeypatch.setattr(cluster, "WORKING_CONFIG", tmp_path / "gui_working.yaml")
    res = cluster.submit("myriad", sources=[{"x": 0, "y": 0, "z": 0, "strength_nAm": 70}])
    assert res["state"] == "dryrun"
    assert res["job_id"] is None
    joined = "\n".join(res["commands"])
    assert "stage.sh" in joined and "submit.sh array" in joined


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
