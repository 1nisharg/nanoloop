# workflow guide

two modes. two audiences. one loop.

---

## table of contents

- [setup — everyone does this first](#setup)
- [text mode — for users](#text-mode--for-users)
- [code mode — for developers](#code-mode--for-developers)
- [the ledger — reading your results](#the-ledger)
- [running overnight](#running-overnight)
- [troubleshooting](#troubleshooting)

---

## setup

everyone does this once.

**step 1 — get a Gemini API key**

go to google console,  sign up (free), create an API key.

**step 2 — clone and enter the repo**

```bash
git clone https://github.com/you/autoresearch
cd autoresearch
```

**step 3 — set your key**

```bash
# mac / linux
export GEMINI_API_KEY=gsk_your_key_here

# windows (cmd)
set GEMINI_API_KEY=gsk_your_key_here

# windows (powershell)
$env:GEMINI_API_KEY="gsk_your_key_here"

# to make it permanent, add the export line to your ~/.bashrc or ~/.zshrc
```

**step 4 — verify it works**

```bash
python loop.py --stats
```

expected output:
```
ledger stats
  total      : 0
  pass       : 0
  revise     : 0
  reject     : 0
  topics     : 0
  latest     : n/a
```

if you see this, you're ready. no pip installs needed for text mode.

---

## text mode — for users

text mode is for thinking. you give it a research question. it reads papers,
generates a hypothesis, makes a plan, and a critic challenges the plan.
no code gets written or run. results land in `ledger.jsonl`.

use text mode when you want to:
- validate whether a research idea is worth pursuing
- understand what's already been tried in a field
- get a research plan before writing any code
- pressure-test a hypothesis before investing compute

---

### workflow: validate a research idea

**the question you're answering:** is this idea worth pursuing?

```bash
python loop.py \
  --topic  "whether rotary position embeddings outperform sinusoidal in decoder-only LMs under 1B params" \
  --mode   text \
  --iters  3 \
  --critic strict
```

**what happens, step by step:**

```
iteration 1
───────────
[arxiv]    searches "rotary position embeddings decoder-only LMs"
           returns 4 papers with titles + abstracts

[web]      DuckDuckGo search for same query
           returns 3–5 text snippets

[proposer] reads: your topic + paper abstracts + web snippets
           outputs: one specific, testable hypothesis
           example output:
             "RoPE outperforms sinusoidal on long-context tasks (>512 tokens)
              in decoder-only LMs under 1B params, but the gap closes below
              256 tokens. The decisive test is perplexity on sequences of
              length 128, 512, 2048 with matched parameter counts."

[executor] reads: the hypothesis
           outputs: a concrete research plan
           example output:
             "1. Use GPT-2 small (117M) as the base architecture.
              2. Train two versions: sinusoidal vs RoPE, identical otherwise.
              3. Evaluate perplexity on PG-19 at sequence lengths 128/512/2048.
              4. Success = RoPE perplexity < sinusoidal at length 2048,
                 gap < 0.5 bits/byte at length 128."

[critic]   reads: hypothesis + plan
           stance: strict (challenges everything)
           example output:
             VERDICT: REVISE
             REASON: The plan doesn't control for training sequence length.
                     Models trained on short sequences may not generalize to
                     long ones regardless of position encoding.
             SUGGEST: Fix training sequence length at 2048 for both models.
                      Add a third baseline: ALiBi.

[ledger]   logs everything → exp_a3f2c1d8

iteration 2
───────────
[proposer] reads: original topic + prior hypothesis + critic feedback
           now knows: need to fix training seq length, add ALiBi baseline
           outputs: revised, stronger hypothesis

[executor] outputs: updated plan addressing the critic's concerns

[critic]   reads: new hypothesis + new plan
           example output:
             VERDICT: PASS
             REASON: Well-controlled experiment with three baselines and
                     a clear falsification criterion.
             SUGGEST: none

[ledger]   logs → exp_b7c4e2f1

iteration 3
───────────
[proposer] builds on the passing hypothesis
           looks for the sharpest version of the idea

... same flow ...

────────────────────
final synthesis
────────────────────
strongest hypothesis: RoPE vs sinusoidal gap is sequence-length dependent,
                      controlled experiment needs matched training length
next step: run the three-way comparison at lengths 128/512/2048
open question: does ALiBi's linear decay hurt performance on short sequences?
```

**reading your results:**

```bash
# quick summary
python loop.py --stats

# full history for this topic
python loop.py --history "whether rotary position embeddings outperform sinusoidal..."

# read the raw log
cat ledger.jsonl
```

---

### workflow: map a research field

use this when you're new to an area and want to understand the landscape before
diving in.

```bash
python loop.py \
  --topic  "what are the open problems in mechanistic interpretability of LLMs as of 2025" \
  --mode   text \
  --iters  2 \
  --critic constructive
```

use `--critic constructive` here — you want the system to build a map, not
tear down every direction.

**what you get:**
- iteration 1: broad hypothesis about the field's frontier
- executor: 3–5 open problems with specific why-not-solved-yet reasoning
- iteration 2: refined, with the most important problems ranked
- synthesis: a lab notebook entry you can actually use as a reading list

---

### workflow: critique a paper before submission

```bash
python loop.py \
  --topic  "paper: we claim sparse autoencoders find monosemantic features in GPT-2.
            method: train SAE on layer 6 MLP activations, evaluate feature
            interpretability by human raters. result: 73% of features rated
            interpretable vs 31% for PCA baseline." \
  --mode   text \
  --iters  2 \
  --critic strict
```

paste your abstract or key claims as the topic. the critic will find:
- weak baselines
- confounds
- missing ablations
- claims the data doesn't support
- what a reviewer will say

---

### text mode — decision guide

| situation | iters | critic | why |
|-----------|-------|--------|-----|
| quick sanity check on an idea | 1 | balanced | fast signal |
| validating before writing code | 3 | strict | pressure-test it |
| exploring a new field | 2 | constructive | build a map, not tear one down |
| pre-submission paper review | 2 | strict | find what reviewers will find |
| brainstorming directions | 2 | constructive | generate options, not kill them |

---

## code mode — for developers

code mode is for measuring. the executor writes a self-contained Python
script that tests the hypothesis, runs it in a sandbox (60s timeout),
and the critic sees the actual output including metrics.

use code mode when you want to:
- run real experiments overnight
- measure something, not just plan to measure it
- iterate on training code automatically
- replicate a paper's result

---

### one rule for code mode

every script the executor writes must print at least one metric line:

```
METRIC: name=value
```

the system parses this. the critic evaluates whether the metric actually
measures what the hypothesis claims. if there's no `METRIC:` line, the
critic will notice.

---

### setup for code mode

code mode experiments may use numpy and torch. install them once:

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

for GPU experiments:
```bash
pip install numpy torch  # installs CUDA version automatically
```

---

### workflow: run an overnight experiment

this is the core use case. you define a research question involving
something measurable. the system runs real Python experiments, measures
a metric, and iterates.

```bash
python loop.py \
  --topic  "does learning rate warmup improve final loss for a 6-layer GPT trained on TinyShakespeare" \
  --mode   code \
  --iters  5 \
  --critic balanced
```

**what happens inside each iteration:**

```
[proposer]  hypothesis:
              "100-step linear warmup from lr=0 to lr=3e-4 reduces final
               training loss compared to constant lr=3e-4 on a 6-layer GPT
               trained for 1000 steps on TinyShakespeare. Expected delta: >0.05
               bits/char."

[executor]  writes a Python script like this:

              import torch
              import torch.nn as nn
              import urllib.request

              # download TinyShakespeare
              url = "https://raw.githubusercontent.com/.../input.txt"
              ...

              # build minimal GPT (6 layers, 6 heads, d_model=384)
              class GPT(nn.Module):
                  ...

              # train with warmup
              def train(use_warmup: bool):
                  model = GPT()
                  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                  for step in range(1000):
                      if use_warmup and step < 100:
                          lr = 3e-4 * (step / 100)
                          for g in optimizer.param_groups:
                              g['lr'] = lr
                      loss = ...
                  return loss.item()

              loss_warmup   = train(use_warmup=True)
              loss_constant = train(use_warmup=False)
              delta = loss_constant - loss_warmup

              print(f"METRIC: loss_warmup={loss_warmup:.4f}")
              print(f"METRIC: loss_constant={loss_constant:.4f}")
              print(f"METRIC: delta={delta:.4f}")

            runs the script in a subprocess (60s timeout)
            
            output:
              METRIC: loss_warmup=2.4821
              METRIC: loss_constant=2.6103
              METRIC: delta=0.1282

[critic]    reads: hypothesis + code + metrics
              VERDICT: PASS
              REASON: delta=0.1282 exceeds the predicted >0.05 threshold.
                      The experiment is clean — single variable changed,
                      matched random seeds, same architecture.
              SUGGEST: none

[ledger]    logs: hypothesis, full code, metrics, verdict → exp_c9d1f3a2
```

**checking results mid-run:**

open a second terminal while the loop is running:

```bash
# see what's been logged so far
python loop.py --stats

# see the actual metrics from code runs
python -c "
import ledger, json
for e in ledger.load_all():
    cr = e.get('code_result')
    if cr and cr.get('metrics'):
        print(e['iteration'], e['verdict'], cr['metrics'])
"
```

---

### workflow: replicate a paper result

```bash
python loop.py \
  --topic  "replicate: Muon optimizer (Kosson et al. 2024) achieves lower
            training loss than AdamW on a 2-layer transformer after 500 steps.
            use identical architecture, batch size 64, sequence length 128." \
  --mode   code \
  --iters  3 \
  --critic strict
```

the executor will:
1. implement both optimizers from scratch or from the paper's pseudocode
2. train both on the same data with identical hyperparameters
3. print `METRIC: muon_loss` and `METRIC: adamw_loss`

the critic will check:
- are the hyperparameters actually matched?
- is the metric the same one the paper reports?
- is the delta within the paper's claimed range?

---

### workflow: modify your own training code

if you have existing training code and want the system to experiment on it,
include it in the topic:

```bash
python loop.py \
  --topic  "improve val_loss on this training loop. current val_loss: 2.84.
            code: $(cat train.py)" \
  --mode   code \
  --iters  5 \
  --critic balanced
```

the executor will modify the code, run it, and measure whether val_loss improved.
the critic will check whether the improvement is real and the change is clean.

for large files, write the key parts to a separate file and reference it:

```bash
# write a short description of your setup to context.txt
echo "6-layer GPT, d_model=384, training on TinyShakespeare, current val_loss=2.84, 
optimizer=AdamW lr=3e-4, batch_size=64, seq_len=256" > context.txt

python loop.py \
  --topic  "$(cat context.txt)" \
  --mode   code \
  --iters  8 \
  --critic strict \
  > overnight.log 2>&1 &
```

---

### what the executor can and cannot do

**can use:**
- Python stdlib (math, random, itertools, collections, etc.)
- numpy
- torch
- anything already installed in your Python environment

**cannot do:**
- external HTTP requests (blocked by the sandbox env)
- write files outside /tmp
- run for more than 60 seconds (hard timeout)
- import packages not installed locally

**the 60-second rule:**

this is intentional. it forces the executor to write minimal, fast experiments.
if an experiment needs more time, the critic will say so and the proposer will
scope it down next iteration. this is the fixed-budget philosophy from autoresearch.

if you need longer runs, change `CODE_TIMEOUT` in `tools.py`:

```python
CODE_TIMEOUT = 300  # 5 minutes
```

---

### code mode — decision guide

| situation | iters | critic | why |
|-----------|-------|--------|-----|
| quick measurement, sanity check | 1–2 | balanced | fast result |
| overnight experiment loop | 8–15 | strict | maximize quality |
| replicating a paper | 3–5 | strict | high bar for match |
| exploring a new technique | 3–5 | constructive | give it room to work |
| debugging a training issue | 2–3 | balanced | find the problem |

---

## the ledger

`ledger.jsonl` is the memory of the system. every experiment is one JSON line.
it grows forever. the proposer reads it before each run.

### reading it

```bash
# stats
python loop.py --stats

# history for a topic
python loop.py --history "your topic string"

# all experiments as readable JSON
cat ledger.jsonl | python -m json.tool

# just the verdicts and hypotheses
python -c "
import ledger
for e in ledger.load_all():
    print(f\"[{e['iteration']}] {e['verdict']:7s} | {e['hypothesis'][:80]}\")
"

# only passing experiments
python -c "
import ledger
for e in ledger.load_by_verdict('PASS'):
    print(e['hypothesis'])
    print()
"

# metrics from code runs
python -c "
import ledger
for e in ledger.load_all():
    cr = e.get('code_result')
    if cr and cr.get('metrics'):
        print(e['id'], e['verdict'], cr['metrics'])
"
```

### the ledger as proposer memory

the most important thing about the ledger: the proposer reads it.

if you run the same topic twice, the second run starts with full knowledge of
what the first run tried and what the critic said. it won't repeat failed
directions. it will try to address the critic's suggestions.

this is what makes the system compound over time rather than restart from
zero each night.

### resetting the ledger

```bash
# nuclear reset — clears all memory
rm ledger.jsonl

# reset just one topic's history (keep everything else)
python -c "
import ledger, json
entries = [e for e in ledger.load_all() if e['topic'] != 'topic to remove']
with open('ledger.jsonl', 'w') as f:
    for e in entries:
        f.write(json.dumps(e) + '\n')
print(f'kept {len(entries)} entries')
"
```

---

## running overnight

the standard overnight pattern:

```bash
# text mode — idea exploration
python loop.py \
  --topic  "your research question" \
  --mode   text \
  --iters  10 \
  --critic strict \
  > run_$(date +%Y%m%d_%H%M).log 2>&1 &

# code mode — real experiments
python loop.py \
  --topic  "your measurable question" \
  --mode   code \
  --iters  8 \
  --critic balanced \
  > run_$(date +%Y%m%d_%H%M).log 2>&1 &

# watch it live
tail -f run_*.log

# check in the morning
python loop.py --stats
python loop.py --history "your topic"
```

**time estimates at ~500 tps:**

| mode | what determines speed | time per iteration |
|------|-----------------------|-------------------|
| text | 3 LLM calls, ~300 tokens each | 15–40 seconds |
| code (fast script) | 3 LLM calls + script runtime | 30–90 seconds |
| code (slow script) | 3 LLM calls + up to 60s timeout | up to 2 minutes |

10 text iterations overnight = ~5 minutes. easily 50+ iterations in a night.
10 code iterations = 5–20 minutes depending on what experiments are written.

---

## troubleshooting

**`GEMINI_API_KEY not set`**
```bash
export GEMINI_API_KEY=gsk_...
# confirm it's set
echo $GEMINI_API_KEY
```

**`GEMINI API error: model not found`**

the model string changed. check the current model list at GOOGLE CONSOLE
and update `GOOGLE_MODEL` in `tools.py`:
```python
GOOGLE_MODEL = "YOUR MODEL"  
```

**arxiv search fails**

arxiv rate-limits aggressive queries. add `--no-arxiv` to skip it:
```bash
python loop.py --topic "..." --no-arxiv
```

**code times out every iteration**

the executor is writing scripts that are too slow. two options:

1. increase the timeout in `tools.py`: `CODE_TIMEOUT = 120`
2. add to your topic: "experiments must complete in under 30 seconds,
   use tiny models and minimal data"

**code runs but no METRIC: line**

the critic will flag this. the next iteration's proposer will see the
critic feedback and instruct the executor to print a metric. if it keeps
happening, add to your topic: "the script MUST print METRIC: name=value
as the last line, this is required"

**ledger.jsonl is growing large**

it's plain text. it won't cause problems until it's millions of lines.
if you want to archive old runs:
```bash
# archive runs older than today
mv ledger.jsonl ledger_$(date +%Y%m%d).jsonl.bak
```

**hypothesis quality is poor**

the proposer's quality depends on the topic string. vague topics get vague
hypotheses. compare:

```
bad:  "improve transformer training"
good: "does replacing layer norm with RMS norm reduce training time without
       degrading perplexity on a 6-layer GPT trained on TinyShakespeare"
```

the more specific your topic, the more specific the hypothesis.

---

## summary

```
text mode
  who   : researchers, students, anyone thinking through ideas
  input : a question or claim
  output: refined hypothesis + research plan + critic verdict
  time  : 15–40s per iteration
  deps  : none beyond Python 3.10+

code mode
  who   : developers, ML practitioners with training code
  input : a measurable question
  output: runnable experiments + real metrics + critic verdict
  time  : 30s–2min per iteration
  deps  : numpy, torch (optional, only if executor uses them)

both modes
  memory   : ledger.jsonl — proposer reads this, won't repeat failures
  overnight: run with --iters 10+ in background, check in the morning
  editing  : change agent behavior by editing system prompts in agents.py
             change your research direction by editing program.md
```
