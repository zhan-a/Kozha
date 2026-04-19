"""Key-hygiene integration test exercising the gitleaks config.

If ``gitleaks`` is not installed we skip — the test is a defense against
misconfiguration, not a unit test. CI runs gitleaks regardless via
``.github/workflows/security.yml``.

Even without the binary, we can still assert that:

- ``.gitleaks.toml`` and ``.pre-commit-config.yaml`` are present at the
  repo root.
- The strict OpenAI-key rule is declared in ``.gitleaks.toml``.

With the binary present, we additionally verify that a synthetic
``sk-…`` blob fails the scan.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gitleaks_config_present() -> None:
    cfg = _REPO_ROOT / ".gitleaks.toml"
    assert cfg.is_file(), f"missing {cfg}"
    body = cfg.read_text(encoding="utf-8")
    assert "openai-api-key" in body
    assert "useDefault = true" in body


def test_precommit_config_references_gitleaks() -> None:
    pc = _REPO_ROOT / ".pre-commit-config.yaml"
    assert pc.is_file(), f"missing {pc}"
    body = pc.read_text(encoding="utf-8")
    assert "gitleaks" in body


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks binary not available"
)
def test_fake_openai_key_is_caught_by_gitleaks(tmp_path: Path) -> None:
    """Write a synthetic OpenAI key into a temp repo and run gitleaks.

    We use ``gitleaks detect --no-git --source <dir>`` so the scan runs
    against the filesystem (no git history needed). Exit code 1
    indicates a finding — that is the pass condition for this test.

    The key string is assembled at runtime so this source file itself
    does not contain a real-looking secret (and therefore does not
    need to be allow-listed).
    """
    cfg = _REPO_ROOT / ".gitleaks.toml"
    assert cfg.is_file()

    # Build a plausible-looking (but obviously synthetic) OpenAI key.
    # 52 post-prefix chars matches typical key lengths.
    fake_key = "sk-" + "A" * 52
    target = tmp_path / "accidentally_committed.py"
    target.write_text(
        f'OPENAI_API_KEY = "{fake_key}"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "gitleaks",
            "detect",
            "--no-git",
            f"--source={tmp_path}",
            f"--config={cfg}",
            "--redact",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # gitleaks exits non-zero on findings; exit code 1 is the pass.
    assert result.returncode != 0, (
        "gitleaks should have caught the fake OpenAI key but returned "
        f"{result.returncode}. stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks binary not available"
)
def test_gitleaks_allowlist_ignores_doc_placeholders(tmp_path: Path) -> None:
    """``sk-XXXX…`` style placeholders in docs should not trip gitleaks."""
    cfg = _REPO_ROOT / ".gitleaks.toml"
    target = tmp_path / "doc.md"
    target.write_text(
        "Set `OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX` in your .env.\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "gitleaks",
            "detect",
            "--no-git",
            f"--source={tmp_path}",
            f"--config={cfg}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "gitleaks false-positived on a documented placeholder. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
