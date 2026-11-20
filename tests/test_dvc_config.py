"""Guard tests for the committed DVC config, ``.dvc/config`` (ADR-0004).

The shared config must point at the team's DagsHub remote and must never hold a credential.
Each member's DagsHub token belongs only in ``.dvc/config.local``, which DVC keeps out of git
(``dvc remote modify --local``; see #13 for the exact commands).

Run ``uv run pytest tests/test_dvc_config.py`` before committing any change under ``.dvc/``:
that is what catches a forgotten ``--local`` in time. CI runs only after a push, so a failure
there means the token is already public: revoke it on DagsHub right away, create a new one, and
store it with ``--local``.

The file is INI-style, so the standard library's ``configparser`` reads it. DVC quotes the
remote's section name, so it appears here as ``'remote "origin"'``, quotes included.
"""

import configparser
from pathlib import Path

DVC_CONFIG = Path(__file__).resolve().parents[1] / ".dvc" / "config"
REMOTE_SECTION = "'remote \"origin\"'"
DAGSHUB_ENDPOINT = "https://dagshub.com/RyanChenJung/claim-frequency-mlops.s3"
# Any setting whose name contains one of these words is treated as a credential, for example
# the S3 options access_key_id, secret_access_key, and session_token.
CREDENTIAL_WORDS = ("key", "secret", "token", "password", "credential")


def read_config(path: Path) -> configparser.ConfigParser:
    """Parse a DVC config file."""
    config = configparser.ConfigParser()
    with open(path, encoding="utf-8") as f:
        config.read_file(f)
    return config


def find_credentials(path: Path) -> list[str]:
    """Return ``section.option`` for every setting in ``path`` that looks like a credential."""
    config = read_config(path)
    return [
        f"{section}.{option}"
        for section in config.sections()
        for option in config[section]
        if any(word in option.lower() for word in CREDENTIAL_WORDS)
    ]


def test_default_remote_is_the_dagshub_s3_endpoint():
    config = read_config(DVC_CONFIG)
    assert config["core"]["remote"] == "origin"
    assert config["core"]["analytics"] == "false"
    assert config[REMOTE_SECTION]["url"] == "s3://dvc"
    assert config[REMOTE_SECTION]["endpointurl"] == DAGSHUB_ENDPOINT


def test_committed_dvc_config_holds_no_credentials():
    assert find_credentials(DVC_CONFIG) == [], (
        "a credential is in .dvc/config; move it with `dvc remote modify --local`"
    )


def test_find_credentials_flags_a_token_written_without_local(tmp_path):
    # What `dvc remote modify origin secret_access_key <token>` (no --local) would write.
    leaked = tmp_path / "config"
    leaked.write_text(
        DVC_CONFIG.read_text(encoding="utf-8").rstrip("\n")
        + "\n    secret_access_key = not-a-real-token\n",
        encoding="utf-8",
    )
    assert find_credentials(leaked) == [f"{REMOTE_SECTION}.secret_access_key"]
