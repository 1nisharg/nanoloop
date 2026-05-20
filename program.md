# program.md

This is the one file you edit. Everything else runs itself.

---

## what this is

autoresearch runs a loop: **Proposer → Executor → Critic**, overnight, autonomously.

- **Proposer** reads literature (arxiv + web), generates a bold hypothesis
- **Executor** makes it concrete — a research plan or runnable code
- **Critic** challenges it: PASS / REVISE / REJECT
- **Ledger** logs everything to `ledger.jsonl` — permanent memory
- On REVISE or REJECT, the critic's feedback goes back to the Proposer. It learns.

---

## quick start

```bash
# 1. set your key
export GROQ_API_KEY=gsk_...

# 2. run
python loop.py --topic "your research question here"

# 3. with options
python loop.py \
  --topic  "sparse autoencoders for LLM interpretability" \
  --mode   text \       # or: code
  --iters  5 \          # iterations overnight
  --critic strict       # strict | balanced | constructive

# 4. check what ran
python loop.py --stats
python loop.py --history "sparse autoencoders for LLM interpretability"
cat ledger.jsonl | python -m json.tool | less
```

---

## modes

| mode   | what executor does                          | use when                          |
|--------|---------------------------------------------|-----------------------------------|
| `text` | writes a research plan: data, method, metric | validating ideas, mapping fields  |
| `code` | writes + runs a Python experiment            | testing things you can measure    |

In `code` mode, the script must print: `METRIC: name=value`
The critic sees the metric output and evaluates whether it actually measures what we think.

---

## critic stances

| stance          | behavior                                              |
|-----------------|-------------------------------------------------------|
| `strict`        | high bar, challenges everything, rare PASS            |
| `balanced`      | fair, PASS if fundamentally sound (default)           |
| `constructive`  | assumes good intent, focuses on how to improve        |

Start with `balanced`. Use `strict` when you want pressure-testing.
Use `constructive` when you want to build momentum on a new idea.

---

## the ledger

Every experiment is logged to `ledger.jsonl`. Each line:

```json
{
  "id":         "exp_a3f2c1d8",
  "timestamp":  "2026-03-15T02:14:33Z",
  "topic":      "sparse autoencoders for LLM interpretability",
  "mode":       "text",
  "iteration":  2,
  "hypothesis": "...",
  "plan":       "...",
  "verdict":    "REVISE",
  "reason":     "...",
  "suggest":    "...",
  "literature": ["Paper title 1", "Paper title 2"]
}
```

The Proposer reads the ledger before each run. It won't repeat failed directions.
This is the memory that compounds over time.

---

## running overnight

```bash
# 10 iterations, strict critic, code mode — wake up to results
python loop.py \
  --topic  "learning rate warmup strategies for small LLMs" \
  --mode   code \
  --iters  10 \
  --critic strict \
  > run_$(date +%Y%m%d).log 2>&1 &

echo "running in background. check: tail -f run_$(date +%Y%m%d).log"
```

---

## editing this file

This file is your research org spec. Change it to change how the system thinks.

Things you can add here that agents will respect (reference it in your --topic prompt):
- constraints: "only use architectures that run on a single GPU"
- priors: "we already know X doesn't work, skip it"
- goals: "optimize for sample efficiency, not final accuracy"
- style: "prefer simple baselines before complex ones"

The agents don't read this file automatically — you reference it by including
relevant context in your `--topic` string. Think of it as your standing instructions.

---

## files

```
loop.py      — main loop. the whole system. < 130 lines.
agents.py    — proposer, executor, critic. one function each.
tools.py     — call_llm, arxiv_search, web_search, run_code
ledger.py    — append-only experiment log. query with load_all(), stats(), etc.
program.md   — this file. you edit this.
ledger.jsonl — created on first run. your experiment memory.
```

That's it. No framework. No config files. No abstraction layers.
If you want to change how an agent thinks, edit its system prompt in agents.py directly.

---

*"One day, frontier AI research used to be done by meat computers..."*
*— Karpathy, March 2026*