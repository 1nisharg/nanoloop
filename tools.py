"""
tools.py — everything agents can use.

  call_llm    : single call to Gemini (gemini-2.5-flash), via REST
  arxiv_search: fetch papers by query, return title + abstract
  web_search  : DuckDuckGo instant answers (no API key needed)
  run_code    : execute Python in a subprocess sandbox, return stdout/stderr

Design rule: every tool is a plain function. No classes, no state.
The ledger (state) lives in ledger.py. Tools are stateless.
"""

import os
import re
import sys
import json
import subprocess
import textwrap
import tempfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

# ---------------------------------------------------------------------------
# LLM — Google Gemini
# ---------------------------------------------------------------------------

GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def call_llm(
    system: str,
    user: str,
    model: str = GEMINI_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Call Gemini with a system + user message. Returns the assistant's reply.

    Requires GEMINI_API_KEY in environment.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Windows PowerShell : $env:GEMINI_API_KEY='AIza...'\n"
            "Mac/Linux          : export GEMINI_API_KEY=AIza..."
        )

    # Gemini uses a different endpoint per model — rebuild if caller passes custom model
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user}]}
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
        },
    }).encode()

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    if "error" in data:
        raise RuntimeError(f"Gemini API error: {data['error']}")

    # extract text from candidates[0].content.parts[0].text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from e


# ---------------------------------------------------------------------------
# Arxiv search
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arxiv. Returns list of dicts with keys:
        id, title, authors, abstract, url, published
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    })

    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    results = []

    for entry in root.findall("atom:entry", ns):
        results.append({
            "id":        entry.find("atom:id", ns).text.strip(),
            "title":     entry.find("atom:title", ns).text.strip().replace("\n", " "),
            "abstract":  entry.find("atom:summary", ns).text.strip().replace("\n", " "),
            "url":       entry.find("atom:id", ns).text.strip(),
            "published": entry.find("atom:published", ns).text[:10],
            "authors":   [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
            ][:3],
        })

    return results


# ---------------------------------------------------------------------------
# Web search — DuckDuckGo (no API key)
# ---------------------------------------------------------------------------

DDG_URL = "https://api.duckduckgo.com/"


def web_search(query: str, max_results: int = 5) -> list[str]:
    """
    DuckDuckGo instant-answer search. No API key required.
    Returns a list of short text snippets.
    """
    params = urllib.parse.urlencode({
        "q":      query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })

    url = f"{DDG_URL}?{params}"
    headers = {"User-Agent": "autoresearch/1.0 (research tool)"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    snippets = []

    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])

    for r in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(r, dict) and r.get("Text"):
            snippets.append(r["Text"])

    return snippets[:max_results]


# ---------------------------------------------------------------------------
# Code runner — subprocess sandbox
# ---------------------------------------------------------------------------

CODE_TIMEOUT = 60  # seconds — matches autoresearch's fixed-budget philosophy


def run_code(code: str, timeout: int = CODE_TIMEOUT) -> dict:
    """
    Run Python code in a subprocess. Returns:
        exit_code : int
        stdout    : str
        stderr    : str
        metric    : dict  — parsed from lines matching "METRIC: name=value"
        timed_out : bool
    """
    # write to temp file so tracebacks show real line numbers
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="autoresearch_"
    ) as f:
        f.write(textwrap.dedent(code))
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # pass full env so venv packages are available
            env=os.environ.copy(),
        )
        timed_out = False
        stdout    = proc.stdout
        stderr    = proc.stderr
        exit_code = proc.returncode

    except subprocess.TimeoutExpired:
        timed_out = True
        stdout    = ""
        stderr    = f"TimeoutExpired: code exceeded {timeout}s"
        exit_code = -1

    finally:
        os.unlink(path)

    # parse METRIC: name=value lines
    metrics = {}
    for line in stdout.splitlines():
        m = re.match(r"METRIC:\s*(\w+)\s*=\s*(.+)", line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val

    return {
        "exit_code": exit_code,
        "stdout":    stdout,
        "stderr":    stderr,
        "metrics":   metrics,
        "timed_out": timed_out,
    }