import os
from pathlib import Path
import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def test_no_key_in_files():
    env_path = os.environ.get("YANDEX_ENV_FILE") or (ROOT / ".env")
    if not Path(env_path).exists():
        pytest.skip("нет файла с ключом")
    env = dotenv_values(env_path)
    key, folder = env["YANDEX_API_KEY"].encode(), env["YANDEX_FOLDER_ID"].encode()
    assert key and folder
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            b = p.read_bytes()
            if key in b:
                hits.append(("API key", str(p)))
    assert not hits, hits


def test_gitignore():
    assert "*.env" in (ROOT / ".gitignore").read_text()
