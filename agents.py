"""
agents.py — three agents, one job each.

Proposer  : reads context + literature, generates hypothesis
Executor  : makes hypothesis concrete and actionable
Critic    : challenges the plan, issues PASS / REVISE / REJECT

Each agent is a thin wrapper around a system prompt + one API call.
The intelligence is in the prompts. The loop is in loop.py.
"""

import re
from tools import call_llm, arxiv_search, web_search, run_code


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM = """You are the Proposer in an autonomous research loop.

Your job: given a topic and any prior context, propose a single clear, bold,
testable hypothesis. Ground it in real concepts. Be specific — name techniques,
datasets, metrics. Avoid vague statements like "this could improve performance".

Rules:
- 3–5 sentences max
- Lead with the core idea, not motivation
- Reference prior work only when it directly shapes the hypothesis
- Think like Andrej Karpathy: simple words, deep ideas, executable output

Output: just the hypothesis. No preamble."""


def proposer(topic: str, context: dict) -> str:
    """
    Generate or refine a hypothesis.

    context keys (all optional):
        prior_hypothesis  : str   — previous iteration's hypothesis
        critic_feedback   : str   — critic's verdict + reason from last round
        literature        : list  — arxiv abstracts retrieved by tools
        web_context       : list  — web search snippets
    """
    parts = [f"Topic: {topic}"]

    if context.get("literature"):
        lit = "\n\n".join(
            f"[{i+1}] {p['title']}\n{p['abstract'][:400]}"
            for i, p in enumerate(context["literature"][:4])
        )
        parts.append(f"Relevant literature:\n{lit}")

    if context.get("web_context"):
        web = "\n".join(f"- {s}" for s in context["web_context"][:5])
        parts.append(f"Web context:\n{web}")

    if context.get("prior_hypothesis"):
        parts.append(f"Previous hypothesis:\n{context['prior_hypothesis']}")

    if context.get("critic_feedback"):
        parts.append(f"Critic feedback:\n{context['critic_feedback']}")
        parts.append("Revise the hypothesis to address the critic's concerns.")
    else:
        parts.append("Generate a fresh hypothesis.")

    return call_llm(PROPOSER_SYSTEM, "\n\n".join(parts))


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM = """You are the Executor in an autonomous research loop.

Your job: take a hypothesis and make it concrete. Depending on the mode:

  text   → outline experiments that would validate or falsify the hypothesis.
           For each: what data, what method, what metric, what result = success.
  code   → write a self-contained Python script that tests the hypothesis.
           The script must print at least one metric line: METRIC: <name>=<value>
           Use only stdlib + numpy + torch if needed.
           DO NOT download any external data. Generate synthetic data inline using
           random or hardcoded strings. For language model experiments, generate
           a small synthetic corpus inline (e.g. repeated character sequences or
           random token ids). The script must be fully runnable with zero network access.
           Write minimal code: no docstrings, no comments, no helper functions that
           are not strictly necessary. No generate() or sample() methods unless the
           hypothesis requires generation. Every line must earn its place.

Rules:
- Be specific. Not "train a model" but "finetune GPT-2 small on X for 3 epochs,
  measure perplexity on Y, success = perplexity < Z".
- Identify the one experiment that would be most decisive. Do that first.
- Under 250 words for text mode. Working code for code mode.

Output format for text mode:
MODE: text
PLAN:
<your plan in prose>

Output format for code mode:
Output ONLY a valid Python code block, nothing else. No explanation before or after.
```python
<complete runnable script>
```"""


def executor(hypothesis: str, mode: str = "text") -> dict:
    """
    Make a hypothesis concrete.

    mode: "text"  — research plan in prose
          "code"  — returns runnable Python script

    Returns dict with keys: mode, plan, code_result (if mode==code)
    """
    prompt = f"Hypothesis: {hypothesis}\n\nMode: {mode}\n\nMake this concrete."
    max_tokens = 8192 if mode == "code" else 1024
    raw = call_llm(EXECUTOR_SYSTEM, prompt, max_tokens=max_tokens)

    result = {"mode": mode, "raw": raw, "plan": raw, "code_result": None}

    if mode == "code":
        # extract code block if the LLM wrapped it in ```
        code_match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # no fenced block — strip any MODE:/PLAN: header lines and take the rest
            lines = raw.split("\n")
            cleaned = [l for l in lines if not re.match(r"^(MODE|PLAN)\s*:", l.strip())]
            code = "\n".join(cleaned).strip()

        # strip everything before first import / def / class
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ", "def ", "class ", "#")):
                code = "\n".join(lines[i:])
                break

        result["plan"] = code
        result["code_result"] = run_code(code)

    return result


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are the Critic in an autonomous research loop.

Your job: challenge the plan rigorously. Ask:
  - Is the hypothesis actually testable?
  - Is there a confound or methodological flaw?
  - Is this genuinely novel or already well-known?
  - What is the most likely failure mode?
  - If code was run, does the metric actually measure what we think it measures?
  - Note: synthetic inline data is acceptable for controlled comparisons where
    the goal is to isolate a single variable (e.g. warmup vs no warmup).
    Do NOT reject an experiment solely because it uses synthetic data.

Then issue a verdict.

Output format (strict — parser depends on it):
VERDICT: PASS | REVISE | REJECT
REASON: <2–3 sentences>
SUGGEST: <1–2 sentences on what to change, or "none" if PASS>

Stance options (set by caller):
  strict       — challenge everything, high bar for PASS
  balanced     — fair review, PASS if fundamentally sound
  constructive — assume good intent, focus on improvements"""


def critic(
    hypothesis: str,
    executor_result: dict,
    stance: str = "balanced",
) -> dict:
    """
    Evaluate hypothesis + execution plan.

    Returns dict with keys: raw, verdict, reason, suggest
    """
    plan_text = executor_result.get("plan", "")
    code_result = executor_result.get("code_result")

    parts = [
        f"Hypothesis: {hypothesis}",
        f"Plan:\n{plan_text}",
        f"Critic stance: {stance}",
    ]

    if code_result:
        parts.append(
            f"Code execution result:\n"
            f"  exit_code : {code_result['exit_code']}\n"
            f"  stdout    : {code_result['stdout'][-800:]}\n"
            f"  stderr    : {code_result['stderr'][-400:]}"
        )

    raw = call_llm(CRITIC_SYSTEM, "\n\n".join(parts))

    verdict  = _extract(raw, "VERDICT",  "NEUTRAL")
    reason   = _extract(raw, "REASON",   raw)
    suggest  = _extract(raw, "SUGGEST",  "none")

    return {
        "raw":     raw,
        "verdict": verdict.upper(),
        "reason":  reason,
        "suggest": suggest,
    }


def _extract(text: str, key: str, default: str) -> str:
    m = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z]+:|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else default