"""
ledger.py — append-only experiment memory.

Every run appends one JSON line to ledger.jsonl.
You can query it, summarize it, or just read it with `cat ledger.jsonl | jq`.

Design: dead simple. No database. No ORM. Just newline-delimited JSON.
The file is the truth. If you delete it, memory resets. That's fine.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional

LEDGER_FILE = Path("ledger.jsonl")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def log_experiment(
    topic:          str,
    mode:           str,
    iteration:      int,
    hypothesis:     str,
    executor_result: dict,
    critic_result:  dict,
    literature:     list  = None,
    tags:           list  = None,
) -> str:
    """
    Append one experiment to ledger.jsonl. Returns the experiment id.
    """
    exp_id = _make_id(topic, iteration)

    entry = {
        "id":          exp_id,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic":       topic,
        "mode":        mode,
        "iteration":   iteration,
        "hypothesis":  hypothesis,
        "plan":        executor_result.get("plan", ""),
        "code_result": _serialize_code_result(executor_result.get("code_result")),
        "verdict":     critic_result.get("verdict", "NEUTRAL"),
        "reason":      critic_result.get("reason", ""),
        "suggest":     critic_result.get("suggest", ""),
        "literature":  [p.get("title") for p in (literature or [])],
        "tags":        tags or [],
    }

    with LEDGER_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    return exp_id


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_all() -> list[dict]:
    """Return every entry in the ledger, oldest first."""
    if not LEDGER_FILE.exists():
        return []
    entries = []
    with LEDGER_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_by_topic(topic: str) -> list[dict]:
    """All entries for a given topic string (exact match)."""
    return [e for e in load_all() if e["topic"] == topic]


def load_by_verdict(verdict: str) -> list[dict]:
    """e.g. load_by_verdict('PASS')"""
    return [e for e in load_all() if e["verdict"] == verdict.upper()]


def best_experiments(n: int = 5) -> list[dict]:
    """Return last n PASS verdicts, most recent first."""
    passed = [e for e in load_all() if e["verdict"] == "PASS"]
    return list(reversed(passed))[:n]


def topic_history(topic: str) -> str:
    """
    Human-readable summary of all experiments on a topic.
    Used by the Proposer to avoid repeating failed directions.
    """
    entries = load_by_topic(topic)
    if not entries:
        return "No prior experiments for this topic."

    lines = [f"Prior experiments on: {topic}", ""]
    for e in entries:
        lines.append(
            f"  [{e['iteration']}] {e['verdict']:7s} | {e['hypothesis'][:120]}..."
            if len(e['hypothesis']) > 120
            else f"  [{e['iteration']}] {e['verdict']:7s} | {e['hypothesis']}"
        )
        if e["verdict"] != "PASS" and e.get("suggest"):
            lines.append(f"           critic: {e['suggest']}")
    return "\n".join(lines)


def stats() -> dict:
    """Quick summary stats over the full ledger."""
    all_e = load_all()
    if not all_e:
        return {"total": 0}

    verdicts = [e["verdict"] for e in all_e]
    return {
        "total":   len(all_e),
        "pass":    verdicts.count("PASS"),
        "revise":  verdicts.count("REVISE"),
        "reject":  verdicts.count("REJECT"),
        "neutral": verdicts.count("NEUTRAL"),
        "topics":  len({e["topic"] for e in all_e}),
        "latest":  all_e[-1]["timestamp"],
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _make_id(topic: str, iteration: int) -> str:
    slug = hashlib.md5(f"{topic}{iteration}{time.time()}".encode()).hexdigest()[:8]
    return f"exp_{slug}"


def _serialize_code_result(cr: Optional[dict]) -> Optional[dict]:
    if cr is None:
        return None
    return {
        "exit_code": cr.get("exit_code"),
        "metrics":   cr.get("metrics", {}),
        "timed_out": cr.get("timed_out", False),
        # truncate stdout/stderr so the ledger stays readable
        "stdout":    (cr.get("stdout") or "")[:500],
        "stderr":    (cr.get("stderr") or "")[:300],
    }