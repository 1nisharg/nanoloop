# nanoloop

> an autonomous research loop in ~500 lines of Python.  
> Proposer → Executor → Critic. Overnight. No framework.

---

## what it is

nanoloop runs a tight three-agent loop on any research topic you give it:

- **Proposer** reads arxiv + web, generates a bold, testable hypothesis
- **Executor** makes it concrete — a research plan or a runnable Python experiment
- **Critic** challenges it hard: `PASS` / `REVISE` / `REJECT`
- **Ledger** logs everything to `ledger.jsonl` — permanent, append-only memory
- On `REVISE` or `REJECT`, critic feedback feeds back to the Proposer. It learns. It doesn't repeat failed directions.

Run it before you sleep. Read the report when you wake up.

---

## why

Most research tooling is either too heavy (full frameworks, vector DBs, orchestration layers) or too shallow (one-shot LLM calls with no feedback loop). nanoloop is neither. It's a minimal, inspectable loop you can read in an afternoon and trust overnight.

The whole system is five files:

```
loop.py      — the loop. the whole system. ~130 lines.
agents.py    — proposer, executor, critic. one function each.
tools.py     — call_llm, arxiv_search, web_search, run_code
ledger.py    — append-only experiment log
ledger.jsonl — created on first run. your experiment memory.
```

No config files. No abstraction layers. If you want to change how an agent thinks, edit its system prompt in `agents.py` directly.

---

## install

```bash
git clone https://github.com/yourname/nanoloop
cd nanoloop
pip install -r requirements.txt   # just standard lib + numpy (optional)
```

Set your API key (nanoloop uses Gemini 2.5 Flash by default):

```bash
# Mac / Linux
export GEMINI_API_KEY=AIza...

# Windows PowerShell
$env:GEMINI_API_KEY="AIza..."
```

---

## quick start

```bash
python loop.py --topic "sparse autoencoders for LLM interpretability"
```

That's it. Three iterations, balanced critic, text mode. Results print to stdout and log to `ledger.jsonl`.

---

## how it works

Each iteration does five things in order:

```
1. tools     — fetch arxiv papers + web snippets on the topic
2. proposer  — generate (or revise) a hypothesis using literature + critic feedback
3. executor  — make it concrete: a research plan (text) or runnable code (code)
4. critic    — PASS / REVISE / REJECT with reason + suggestion
5. ledger    — log everything. next iteration reads this.
```

After all iterations, a final synthesis agent writes a lab notebook entry:
the strongest surviving hypothesis, the most actionable next step, and the biggest open question.

---

## modes

### text mode (default)

The Executor writes a structured research plan — what data, what method, what metric, what result counts as success.

Best for: mapping a field, validating ideas before coding, generating experiment designs.

```bash
python loop.py --topic "muon optimizer for transformer training"
# or explicitly:
python loop.py --topic "muon optimizer for transformer training" --mode text
```

**example output (executor, text mode):**
```
MODE: text
PLAN:
Finetune a 125M GPT-2 on OpenWebText for 10k steps with Muon vs AdamW.
Track val loss every 500 steps. Success = Muon reaches AdamW's 10k-step
loss at ≤ 7k steps (≥ 30% sample efficiency gain). Use identical LR
schedule (cosine, 1e-3 peak), batch size 512, no other changes.
```

---

### code mode

The Executor writes a self-contained Python script and **runs it**. The Critic sees the actual metric output — not just a plan.

Best for: testing things you can measure, benchmarking, sanity-checking ideas with real numbers.

```bash
python loop.py --topic "learning rate warmup for small LLMs" --mode code
```

The script must print at least one metric line:
```
METRIC: name=value
```

nanoloop parses this and passes it to the Critic. The Critic evaluates whether the metric actually measures what we think it measures.

**example output (executor, code mode):**
```python
import numpy as np

steps = 1000
warmup = 100
lr_warmup   = np.array([min(1, i/warmup) * 1e-3 for i in range(steps)])
lr_constant = np.full(steps, 1e-3)

loss_warmup   = np.exp(-lr_warmup.cumsum() / steps) + np.random.normal(0, 0.01, steps)
loss_constant = np.exp(-lr_constant.cumsum() / steps) + np.random.normal(0, 0.012, steps)

print(f"METRIC: final_loss_warmup={loss_warmup[-1]:.4f}")
print(f"METRIC: final_loss_constant={loss_constant[-1]:.4f}")
print(f"METRIC: warmup_advantage={loss_constant[-1] - loss_warmup[-1]:.4f}")
```

**note:** in code mode, scripts must be fully self-contained. No network calls, no file downloads. Generate synthetic data inline. The sandbox has stdlib + numpy available.

---

## all options

```bash
python loop.py \
  --topic   "your research question"   # required
  --mode    text                        # text | code  (default: text)
  --iters   3                           # iterations   (default: 3)
  --critic  balanced                    # strict | balanced | constructive (default: balanced)
  --no-arxiv                            # skip arxiv search
  --no-web                              # skip web search
```

### critic stances

| stance | behavior | use when |
|---|---|---|
| `balanced` | fair review, PASS if fundamentally sound | default, most runs |
| `strict` | high bar, challenges everything, rare PASS | pressure-testing a hypothesis |
| `constructive` | assumes good intent, focuses on improvements | building momentum on a new idea |

---

## overnight runs

```bash
# 10 iterations, strict critic, code mode
python loop.py \
  --topic  "learning rate warmup strategies for small LLMs" \
  --mode   code \
  --iters  10 \
  --critic strict \
  > run_$(date +%Y%m%d).log 2>&1 &

echo "running. tail -f run_$(date +%Y%m%d).log"
```

Wake up to a synthesis report and a populated ledger.

---

## the ledger

Every experiment is logged to `ledger.jsonl` — one JSON line per iteration, forever.

```json
{
  "id":         "exp_a3f2c1d8",
  "timestamp":  "2026-03-15T02:14:33Z",
  "topic":      "sparse autoencoders for LLM interpretability",
  "mode":       "text",
  "iteration":  2,
  "hypothesis": "Sparse autoencoders trained on residual stream activations...",
  "plan":       "...",
  "verdict":    "REVISE",
  "reason":     "Hypothesis conflates feature detection with causal attribution.",
  "suggest":    "Add an intervention experiment: ablate top-k features, measure task accuracy drop.",
  "literature": ["Towards Monosemanticity", "Scaling Monosemanticity"]
}
```

### querying the ledger

```bash
# stats across all runs
python loop.py --stats

# full history for a topic
python loop.py --history "sparse autoencoders for LLM interpretability"

# raw inspection
cat ledger.jsonl | python -m json.tool | less

# all PASS verdicts
cat ledger.jsonl | python -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    if e['verdict'] == 'PASS':
        print(e['timestamp'], '|', e['hypothesis'][:100])
"
```

### from Python

```python
import ledger

# summary stats
ledger.stats()
# {'total': 12, 'pass': 3, 'revise': 6, 'reject': 3, 'topics': 2, 'latest': '...'}

# all experiments for a topic
ledger.load_by_topic("sparse autoencoders for LLM interpretability")

# best experiments (most recent PASSes)
ledger.best_experiments(n=3)

# human-readable history (what the Proposer reads)
print(ledger.topic_history("your topic"))
```

---

## customising agents

All agent intelligence lives in system prompts inside `agents.py`. To change how an agent thinks, edit its `_SYSTEM` string directly. No config, no indirection.

```python
# agents.py

PROPOSER_SYSTEM = """You are the Proposer in an autonomous research loop.
...
"""  # ← edit this

EXECUTOR_SYSTEM = """..."""  # ← or this

CRITIC_SYSTEM   = """..."""  # ← or this
```

### example: add a constraint to the Proposer

```python
PROPOSER_SYSTEM = """You are the Proposer in an autonomous research loop.

Constraint: only propose hypotheses testable on a single consumer GPU (≤24GB VRAM).
Constraint: prefer simple baselines before complex ones.

Your job: ...
"""
```

---

## example runs

### map a field (text mode, 5 iters)
```bash
python loop.py \
  --topic "state space models vs transformers for long context" \
  --mode text \
  --iters 5
```

### pressure-test an idea (strict critic, code mode)
```bash
python loop.py \
  --topic "flash attention reduces memory without hurting convergence" \
  --mode code \
  --critic strict \
  --iters 3
```

### build momentum on a new idea (constructive, text mode)
```bash
python loop.py \
  --topic "mixture of depths for adaptive compute in LLMs" \
  --mode text \
  --critic constructive \
  --iters 4
```

### skip external search (pure reasoning, fast)
```bash
python loop.py \
  --topic "why does batch norm fail with batch size 1" \
  --mode text \
  --no-arxiv \
  --no-web \
  --iters 2
```

---

## files

| file | what it does | lines |
|---|---|---|
| `loop.py` | main loop, CLI, synthesis | ~130 |
| `agents.py` | proposer, executor, critic | ~150 |
| `tools.py` | call_llm, arxiv, web, run_code | ~150 |
| `ledger.py` | append-only experiment log | ~100 |
| `ledger.jsonl` | your experiment memory (auto-created) | — |

---

## requirements

- Python 3.10+
- `GEMINI_API_KEY` (Gemini 2.5 Flash)
- numpy (optional, available in code mode sandbox)
- no other dependencies

---

*"One day, frontier AI research used to be done by meat computers..."*  
*— Karpathy, March 2026*
