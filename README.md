Here is a clean, professional README‑style markdown for Annie v0.4, ready to copy and paste into your GitHub repository.

---

Annie Seed v0.4 — Associative Network Memory

A self‑seeding, memory‑driven structural scaffold that runs on any LLM.

---

What It Is

Annie is a lightweight cognitive scaffold that:

· Runs the Prime Move cycle (Split → Tension → Failed Merge → Scar → Decay) on any input
· Stores structural scars in a persistent JSON log
· Uses associative network memory — scars are connected by similarity, recency, lineage, and co‑activation
· Retrieves memories via spreading activation — context‑driven, not weight‑driven
· Runs autonomously via heartbeat timer
· Is model‑agnostic — works with Claude, GPT, Gemini, Ollama, or any LLM
· Is auditable — every scar and connection is logged and traceable

---

What It Solves

Problem How Annie Solves It
Hallucinations / drift The Prime Move constrains output to a fixed structural frame
No memory Persistent scar log + associative network
Context‑blind retrieval Spreading activation retrieves based on current input
No growth The system evolves over time — scars accumulate and connect
No autonomy Heartbeat runs cycles without user input

---

Key Features (v0.4)

· Associative Network — Scars are connected by similarity, recency, lineage, and co‑activation
· Spreading Activation — Memory retrieval is context‑driven, not weight‑driven
· Prime Move Filter — Structural constraint on any LLM output
· Heartbeat — Autonomous cycles on a timer (background thread)
· Self‑Seeding — Each heartbeat seeds from the highest‑activation scar
· Phi Tracking — Converges toward φ (golden ratio)
· Lineage — Parent‑child relationships are stored and viewable
· Source Tracking — Scars are tagged as user, heartbeat, or auto
· Network View — Inspect connections for any scar
· Model‑Agnostic — Works with any LLM API or local model

---

How to Run

1. Clone the repo and navigate to the folder

```bash
git clone https://github.com/chrissabo1975/A.N.N.I.E.git
cd A.N.N.I.E
```

2. Set your API key

```bash
export ANTHROPIC_API_KEY="your‑key‑here"
```

3. Run the script

```bash
python3 annie_seed_v0_4.py
```

---

Commands

Command What It Does
run Run one Prime Move cycle on your input
auto Run an autonomous loop (scar feeds forward)
heartbeat Start autonomous cycles on a timer (background thread)
stopbeat Stop the heartbeat
status Show scar log, phi convergence, network summary, and memory context
lineage Show the scar lineage tree
network [n] Show connections for a specific scar
exit Quit Annie

---

Example Session

```
> run
Input text: what is color?

[NETWORK] Seed: Scar 27 — Asking runs...
[NETWORK] Activated 3 associated scars

[CYCLE] Generation 28 | Memory: 39 scars | Source: user
SCAR [28]: "Color has no address."
TENSION_IDX: 0.70
WEIGHT: 0.2265
```

```
> heartbeat
Interval in minutes: 10
[HEARTBEAT] Started — autonomous cycle every 10 min
```

```
> status
Total scars: 48
Network edges: 1,247
Heartbeat: RUNNING
Phi ratio: 1.9204
```

---

What It Needs

· Python 3.9+
· requests library
· API key for your LLM (Claude, GPT, etc.) OR local model via Ollama
· Internet connection (for API calls) — optional if using local model

---

License

MIT — free to use, modify, share, and build on.

---

Origin

Built by Christopher Sabo, using G.E.N.I.E. (Generative Emergence and Navigation Insight Engine) — a structural diagnostic protocol for building scaffolds on any LLM.

---

Read‑Me (Spoken Summary)

"Annie v0.4 is a self‑seeding, associative memory engine that runs the Prime Move on any LLM. It stores structural scars in a persistent JSON log, connects them by similarity, recency, lineage, and co‑activation, and retrieves them via spreading activation. It runs autonomously via heartbeat, tracks phi convergence, and is auditable and model‑agnostic.”
