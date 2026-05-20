"""
loop.py — the research loop. This is the whole system.

Usage:
    python loop.py --topic "sparse autoencoders for LLM interpretability"
    python loop.py --topic "..." --mode code --iters 5 --critic strict
    python loop.py --stats
    python loop.py --history "your topic"

What happens each iteration:
    1. tools    : fetch literature (arxiv) + web context
    2. proposer : generate / refine hypothesis using literature + critic feedback
    3. executor : make it concrete (text plan or runnable code)
    4. critic   : PASS / REVISE / REJECT with reason
    5. ledger   : log everything to ledger.jsonl
    6. repeat   : if REJECT or REVISE, feed critic feedback back to proposer

After all iterations: synthesize a final research report.
"""

import argparse
import sys
import textwrap
from typing import Optional

import agents
import ledger
import tools


# ---------------------------------------------------------------------------
# Synthesis — final report after all iterations
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = """You are a senior researcher summarizing a completed research loop.

Write a lab notebook entry (150 words max):
1. Strongest surviving hypothesis
2. Most actionable next step  
3. Biggest open question remaining

Style: Karpathy-terse. No fluff. First person. Present tense."""


def synthesize(topic: str, history: list[dict]) -> str:
    summary = "\n\n---\n\n".join(
        f"Iteration {h['iter']}\n"
        f"Hypothesis: {h['hypothesis']}\n"
        f"Plan: {h['plan'][:400]}\n"
        f"Verdict: {h['verdict']} — {h['reason']}"
        for h in history
    )
    return tools.call_llm(SYNTH_SYSTEM, f"Topic: {topic}\n\n{summary}")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run(
    topic:       str,
    mode:        str  = "text",   # "text" | "code"
    iters:       int  = 3,
    critic_mode: str  = "balanced",
    use_arxiv:   bool = True,
    use_web:     bool = True,
    verbose:     bool = True,
) -> list[dict]:
    """
    Run the Proposer → Executor → Critic loop for `iters` iterations.
    Returns the full history list.
    """
    def log(msg): 
        if verbose:
            print(msg, flush=True)

    log(f"\n{'='*60}")
    log(f"  autoresearch")
    log(f"  topic : {topic}")
    log(f"  mode  : {mode}  |  iters: {iters}  |  critic: {critic_mode}")
    log(f"{'='*60}\n")

    # show prior work so we don't repeat ourselves
    prior = ledger.topic_history(topic)
    if "No prior" not in prior:
        log(f"[ledger] prior experiments found:\n{prior}\n")

    context     = {"prior_hypothesis": None, "critic_feedback": None}
    history     = []

    for i in range(1, iters + 1):
        log(f"── iteration {i}/{iters} {'─'*40}")

        # 1. fetch literature
        if use_arxiv:
            log(f"[arxiv]   searching: {topic[:60]}...")
            try:
                context["literature"] = tools.arxiv_search(topic, max_results=4)
                log(f"[arxiv]   {len(context['literature'])} papers found")
            except Exception as e:
                log(f"[arxiv]   failed: {e}")
                context["literature"] = []

        if use_web:
            log(f"[web]     searching: {topic[:60]}...")
            try:
                context["web_context"] = tools.web_search(topic)
                log(f"[web]     {len(context['web_context'])} snippets")
            except Exception as e:
                log(f"[web]     failed: {e}")
                context["web_context"] = []

        # 2. proposer
        log(f"\n[proposer] generating hypothesis...")
        hypothesis = agents.proposer(topic, context)
        log(f"\n  hypothesis:\n{textwrap.indent(hypothesis, '  ')}\n")

        # 3. executor
        log(f"[executor] mode={mode} ...")
        exec_result = agents.executor(hypothesis, mode=mode)

        if mode == "code":
            log(f"\n  code:\n{textwrap.indent(exec_result['plan'], '  ')}\n")
            cr = exec_result.get("code_result", {})
            if cr:
                status = "✓ ok" if cr["exit_code"] == 0 else f"✗ exit {cr['exit_code']}"
                log(f"[executor] ran code — {status}")
                if cr.get("metrics"):
                    log(f"[executor] metrics: {cr['metrics']}")
                if cr.get("timed_out"):
                    log(f"[executor] timed out after {tools.CODE_TIMEOUT}s")
                if cr.get("stderr"):
                    log(f"[executor] stderr:\n{textwrap.indent(cr['stderr'][-400:], '  ')}")
        else:
            log(f"\n  plan:\n{textwrap.indent(exec_result['plan'][:500], '  ')}\n")

        # 4. critic
        log(f"[critic]   stance={critic_mode} ...")
        crit_result = agents.critic(hypothesis, exec_result, stance=critic_mode)
        verdict = crit_result["verdict"]
        log(f"\n  verdict : {verdict}")
        log(f"  reason  : {crit_result['reason']}")
        if crit_result["suggest"] and crit_result["suggest"].lower() != "none":
            log(f"  suggest : {crit_result['suggest']}")

        # 5. log to ledger
        exp_id = ledger.log_experiment(
            topic=topic,
            mode=mode,
            iteration=i,
            hypothesis=hypothesis,
            executor_result=exec_result,
            critic_result=crit_result,
            literature=context.get("literature", []),
        )
        log(f"\n[ledger]  logged → {exp_id}")

        # 6. store for synthesis
        history.append({
            "iter":       i,
            "hypothesis": hypothesis,
            "plan":       exec_result.get("plan", ""),
            "verdict":    verdict,
            "reason":     crit_result["reason"],
            "suggest":    crit_result["suggest"],
        })

        # feed critic back to proposer for next round
        context["prior_hypothesis"] = hypothesis
        context["critic_feedback"]  = (
            f"Verdict: {verdict}\n"
            f"Reason: {crit_result['reason']}\n"
            f"Suggestion: {crit_result['suggest']}"
        )

        log("")

    # 7. synthesize
    log(f"{'='*60}")
    log(f"  final synthesis")
    log(f"{'='*60}\n")
    report = synthesize(topic, history)
    log(report)
    log("")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="autoresearch — autonomous research loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        examples:
          python loop.py --topic "sparse autoencoders for LLM interpretability"
          python loop.py --topic "muon optimizer for transformer training" --mode code --iters 5
          python loop.py --topic "..." --critic strict --no-web
          python loop.py --stats
          python loop.py --history "sparse autoencoders for LLM interpretability"
        """),
    )

    parser.add_argument("--topic",   type=str, help="research topic or question")
    parser.add_argument("--mode",    type=str, default="text", choices=["text", "code"],
                        help="text = research plan, code = runnable experiment (default: text)")
    parser.add_argument("--iters",   type=int, default=3,
                        help="number of proposer→executor→critic iterations (default: 3)")
    parser.add_argument("--critic",  type=str, default="balanced",
                        choices=["strict", "balanced", "constructive"],
                        help="critic stance (default: balanced)")
    parser.add_argument("--no-arxiv", action="store_true", help="skip arxiv search")
    parser.add_argument("--no-web",   action="store_true", help="skip web search")
    parser.add_argument("--stats",    action="store_true", help="print ledger stats and exit")
    parser.add_argument("--history",  type=str, metavar="TOPIC",
                        help="print experiment history for a topic and exit")

    args = parser.parse_args()

    if args.stats:
        s = ledger.stats()
        print(f"\nledger stats")
        print(f"  total      : {s.get('total', 0)}")
        print(f"  pass       : {s.get('pass', 0)}")
        print(f"  revise     : {s.get('revise', 0)}")
        print(f"  reject     : {s.get('reject', 0)}")
        print(f"  topics     : {s.get('topics', 0)}")
        print(f"  latest     : {s.get('latest', 'n/a')}")
        return

    if args.history:
        print(ledger.topic_history(args.history))
        return

    if not args.topic:
        parser.error("--topic is required (or use --stats / --history)")

    run(
        topic       = args.topic,
        mode        = args.mode,
        iters       = args.iters,
        critic_mode = args.critic,
        use_arxiv   = not args.no_arxiv,
        use_web     = not args.no_web,
    )


if __name__ == "__main__":
    main()