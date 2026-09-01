"""Credentials, loaded from .env beside this file.

`biko.py --build`, `evaluate.py` and the Streamlit app are each launched a
different way, so relying on what happens to be exported in the calling shell
means one of them is always missing a key. Reading .env instead gives all
three the same view. Import this module before anything reads os.environ.
"""
import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# override=False so an explicitly exported key still wins -- handy when
# testing a second account without editing .env.
load_dotenv(Path(__file__).with_name(".env"), override=False)

# Answer generation (generator.py). Retrieval works without it; generation
# does not.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# LlamaCloud / LlamaParse -- document parsing only. Deliberately separate from
# the key above: it drives no model and cannot answer a question.
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
