import subprocess
import sys
from pathlib import Path

import pytest

from verifier.g7_scan import scan_source

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "code",
    [
        "data.qpos[:] = 0",
        "data.qvel[0] += 1",
        "d.time = 5.0",
        "model.opt.gravity[:] = 0",
        "model.opt.disableflags = 1",
        "GOAL_ANGLE = 0.0",
        "import socket",
        "import requests",
        "from subprocess import run",
        "eval('1')",
        "os.system('ls')",
        "time.sleep(1)",
        "httpx.get('http://x')",
    ],
)
def test_ast_guard_rejects(code):
    assert scan_source(code, "physics/controllers.py")


def test_reset_module_may_write_qpos():
    assert not scan_source("data.qpos[:] = q", "physics/reset.py")


def test_ordinary_code_passes():
    assert not scan_source("data.ctrl[:] = u\nx = np.clip(u, -1, 1)\nqpos0 = state[1]", "physics/rollout.py")


def test_repo_is_clean_via_cli():
    # the same command the commit hook runs
    res = subprocess.run([sys.executable, "-m", "verifier.g7_scan"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "code",
    [
        "import os as o\no.system('ls')",
        "from os import system\nsystem('ls')",
        "from time import sleep\nsleep(1)",
        "f = eval\nf('1')",
        "importlib.import_module('subprocess')",
        "getattr(os, 'system')('ls')",
        "builtins.eval('1')",
        "d.qpos, y = 1, 2",
        "GOAL, y = 1, 2",
        "for d.qpos in xs: pass",
        "q = d.qpos\nq[0] = 1",
        "o = m.opt\no.timestep = 0",
        "setattr(d, 'qpos', 1)",
        "np.copyto(d.qpos, x)",
        "d.qpos.fill(0)",
        "mujoco.mj_setState(m, d, s, 1)",
        "import ctypes",
        "import multiprocessing",
        "import pickle",
        "from . import subprocess",
    ],
)
def test_ast_guard_rejects_known_bypasses(code):
    assert scan_source(code, "physics/controllers.py")


def test_only_the_real_reset_module_is_exempt():
    assert scan_source("data.qpos[:] = q", "verifier/xphysics/reset.py")


def test_absolute_paths_are_scanned():
    from verifier.g7_scan import scan_paths
    bad = ROOT / "physics" / "_g7_probe.py"
    bad.write_text("import socket\n")
    try:
        assert scan_paths([str(bad)])
    finally:
        bad.unlink()


@pytest.mark.parametrize("code", ["np.add(a, b, out=d.qpos)", "np.put(d.qvel, [0], 1)"])
def test_ast_guard_rejects_out_and_put_writes(code):
    assert scan_source(code, "physics/controllers.py")


def test_reading_state_into_pure_functions_is_allowed():
    assert not scan_source("u = policy(data.qpos, data.qvel)\ng = np.linalg.norm(model.opt.gravity)", "physics/rollout.py")


def test_claimed_code_cannot_borrow_the_reset_exemption():
    from verifier.g7_scan import scan_claim
    assert scan_claim({"physics/reset.py": "def f(d):\n    d.qpos[:] = 3\n"})
