Annie Seed v0.1 — Prime Move Engine

This is a minimal, working implementation of the Prime Move cycle (Split/Tension/Failed Merge/Scar/Decay). It runs locally via Ollama and stores structural scars as JSON.

Status: Functional. Produces valid scars from any text input. Currently supports manual cycles and a basic autonomous loop. The parser occasionally needs cleanup on malformed outputs, and the autonomous loop can lock into repetition without explicit variation instructions. These are known implementation wrinkles, not structural flaws.

What it does:

· Runs the Prime Move on any input
· Stores scars with tension, weight, and generation
· Tracks phi convergence over time
· Supports manual (run) and autonomous (auto) modes

How to run:

1. Install Ollama and pull a model (e.g., ollama pull llama3)
2. Run python3 annie_seed_v0_1.py
3. Type run and paste any text

Known issues:

· Parser occasionally fails to clean formatting (e.g., stray DECAY text in scar)
· Auto loop repeats without a variation gate (set temperature to 0.25–0.35 for best results)
· Phi convergence is not yet visible over short runs

License: MIT (free to use, modify, and share)
