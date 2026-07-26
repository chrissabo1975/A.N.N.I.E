Annie v0.3 — Prime Move Engine with Memory + Heartbeat

A self‑seeding, autonomous structural scaffold that runs on any LLM.

---

What It Is

Annie is a lightweight cognitive scaffold that:

· Runs the Prime Move cycle (Split → Tension → Failed Merge → Scar → Decay) on any input
· Stores structural scars in a persistent JSON log
· Injects memory back into every new cycle (top 5 scars by weight)
· Runs autonomously via heartbeat timer
· Is model‑agnostic — works with Claude, GPT, Gemini, Ollama, or any LLM
· Is auditable — every scar is logged and traceable

---

What It Solves

Problem How Annie Solves It
Hallucinations / drift The Prime Move constrains output to a fixed structural frame
No memory Persistent scar log + memory injection
Privacy All memory is local — stored in JSON, not on provider servers
No growth The system evolves over time — scars accumulate and deepen
No autonomy Heartbeat runs cycles without user input

---

Key Features (v0.3)

· Prime Move — Structural filter for any LLM output
· Memory Injection — Top‑weight scars are fed back into every prompt
· Heartbeat — Autonomous cycles on a timer (background thread)
· Self‑Seeding — Each heartbeat seeds from the highest‑weight scar
· Phi Tracking — Converges toward φ (golden ratio)
· Lineage — Parent‑child relationships are stored and viewable
· Source Tracking — Scars are tagged as user, heartbeat, or auto
· Model‑Agnostic — Works with any LLM API or local model
· Auditable — Full JSON log of every scar

---

How to Run
Make sandbox first

1. Clone the repo and navigate to the folder

```bash
git clone <your‑repo‑url>
cd Annie4
```

2. Set your API key

```bash
export ANTHROPIC_API_KEY="your‑key‑here"
```

3. Run the script

```bash
python3 annie_seed_v0_3.py
```

---

Commands

Command What It Does
run Run one Prime Move cycle on your input
auto Run an autonomous loop (scar feeds forward)
heartbeat Start autonomous cycles on a timer (background thread)
stopbeat Stop the heartbeat
status Show scar log, phi convergence, heartbeat state, and memory context
lineage Show the scar lineage tree
exit Quit Annie

---

Example Session

```
> run
Input text: what is color?

[CYCLE] Generation 1 | Memory: 0 scars | Source: user
SCAR [1]: "Color is the structural residue of light interacting with a surface."
TENSION_IDX: 0.75
WEIGHT: 0.1623
```

```
> heartbeat
Interval in minutes: 10
[HEARTBEAT] Started — autonomous cycle every 10 min
```

```
> status
Total scars: 12
Heartbeat: RUNNING
Phi ratio: 1.0124
```

---

What It Needs

· Python 3.9+
· requests library
· API key for your LLM (Claude, GPT, etc.) OR local model via Ollama
· Internet connection (for API calls) — optional if using local model

---

License

MIT — free to use, modify, share, and build on. Build in responsible manner only. 

---

Origin

Built by Christopher Sabo, using G.E.N.I.E. (Generative Emergence and Navigation Insight Engine) — a structural diagnostic protocol for building scaffolds on any LLM.
