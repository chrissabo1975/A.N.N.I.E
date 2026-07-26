"""
Annie Seed v0.3 — Prime Move Engine with Memory + Heartbeat
===========================================================
Changes from v0.2:
- Heartbeat: Annie runs autonomous cycles every N minutes
- Uses highest-weight scar as seed for each heartbeat cycle
- Runs in background thread — you can still use other commands
- stopbeat to pause, heartbeat to restart
"""

import json
import math
import re
import uuid
import os
import time
import threading
import requests
from typing import Optional

# ── CONFIGURATION ──
SCARS_PATH = "annie_seed_scars.json"
LINEAGE_PATH = "annie_seed_lineage.json"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PHI = (1 + math.sqrt(5)) / 2
MAX_CONTEXT_SCARS = 5

# ── HEARTBEAT STATE ──
heartbeat_active = [False]

# ── SYSTEM PROMPT ──
SYSTEM_PROMPT = """You are ANNIE_SEED_v0_3.

Run a single Prime Move cycle on the given text.

OUTPUT FORMAT (follow exactly):

### SPLIT
[One sentence — the core distinction being made]

### TENSION
[Two requirements that cannot both be fully satisfied — name both explicitly]

### FAILED MERGE
[Why resolution cannot fully succeed — one sentence]

### SCAR
[The irreducible residue — one SHORT sentence, as brief as possible]
TENSION_INDEX: [0.0-1.0 — how much unresolved tension remains]

### DECAY
[What releases into background — one sentence]
"""


# ══════════════════════════════════════════════════════════════
# STORAGE
# ══════════════════════════════════════════════════════════════

def load_json(path: str) -> list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_json(path: str, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chunk: int = 400) -> list:
    chunks = []
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        if len(raw) > max_chunk:
            sentences = re.split(r'(?<=[.!?])\s+', raw)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) > max_chunk and current:
                    chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                chunks.append(current)
        else:
            chunks.append(raw)
    return [c for c in chunks if c]


# ══════════════════════════════════════════════════════════════
# SCAR CONTEXT — MEMORY INJECTION
# ══════════════════════════════════════════════════════════════

def build_scar_context() -> str:
    scars = load_json(SCARS_PATH)
    if not scars:
        return ""

    sorted_scars = sorted(scars, key=lambda x: x.get("weight", 0), reverse=True)
    top_scars = sorted_scars[:MAX_CONTEXT_SCARS]

    lines = ["### ACCUMULATED SCAR SUBSTRATE (your structural memory)"]
    lines.append("These are the highest-weight distinctions from previous cycles.")
    lines.append("They are the irreducible residues of what has already been processed.")
    lines.append("Let them inform — but do not repeat — your current cycle.\n")

    for s in top_scars:
        lines.append(
            f"Scar {s['seq_id']} "
            f"[weight={s['weight']:.4f}] "
            f"[tension={s['tension_index']:.2f}]: "
            f"{s['content']}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════

def call_llm(user_text: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        print("\n[ERROR] No Anthropic API key found.")
        print("[ERROR] Run: export ANTHROPIC_API_KEY=\"your-key-here\"")
        return None

    scar_context = build_scar_context()

    if scar_context:
        user_message = scar_context + "\n\n### CURRENT INPUT\n" + user_text
    else:
        user_message = user_text

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Anthropic API. Check internet.")
        return None

    except requests.exceptions.Timeout:
        print("\n[ERROR] API call timed out.")
        return None

    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("\n[ERROR] Invalid API key.")
        elif response.status_code == 429:
            print("\n[ERROR] Rate limit hit. Wait and try again.")
        else:
            print(f"\n[ERROR] API error {response.status_code}: {e}")
            print(f"[ERROR] Response body: {response.text}")
        return None

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_cycle_output(raw: str) -> dict:
    sections = {
        "split": "",
        "tension": "",
        "failed_merge": "",
        "scar": "",
        "decay": "",
        "tension_index": 0.5
    }

    current = None
    for line in raw.splitlines():
        line_stripped = line.strip()

        if line_stripped.startswith("### "):
            key = line_stripped[4:].lower().replace(" ", "_")
            if key in sections:
                current = key
            else:
                current = None
            continue

        ti_match = re.search(
            r"TENSION_INDEX:\s*([0-9]*\.?[0-9]+)",
            line_stripped, re.IGNORECASE
        )
        if ti_match:
            try:
                sections["tension_index"] = max(
                    0.0, min(1.0, float(ti_match.group(1)))
                )
            except ValueError:
                pass
            if current == "scar":
                continue

        if current and current != "tension_index":
            if sections[current]:
                sections[current] += " " + line_stripped
            else:
                sections[current] = line_stripped

    sections["scar"] = re.sub(
        r"TENSION_INDEX:\s*[0-9]*\.?[0-9]+", "",
        sections["scar"]
    ).strip()

    return sections


# ══════════════════════════════════════════════════════════════
# SCAR STORAGE
# ══════════════════════════════════════════════════════════════

def compute_weight(content: str, tension_index: float) -> float:
    if not content:
        return 0.0
    return tension_index / math.log(len(content) + 1)


def get_next_sequential_id(scars: list) -> int:
    if not scars:
        return 1
    return max(s.get("seq_id", 0) for s in scars) + 1


def add_scar(content: str, tension_index: float,
             parent_chunk_id: str, generation: int,
             parent_scar_ids: list = None) -> dict:
    scars = load_json(SCARS_PATH)
    seq_id = get_next_sequential_id(scars)
    scar_uuid = str(uuid.uuid4())
    weight = compute_weight(content, tension_index)

    scar = {
        "seq_id": seq_id,
        "id": scar_uuid,
        "content": content,
        "tension_index": tension_index,
        "weight": weight,
        "generation": generation,
        "parent_chunk_id": parent_chunk_id,
        "parent_scar_ids": parent_scar_ids or [],
        "source": "user"
    }

    scars.append(scar)
    save_json(SCARS_PATH, scars)
    return scar


def add_scar_with_source(content: str, tension_index: float,
                          parent_chunk_id: str, generation: int,
                          parent_scar_ids: list = None,
                          source: str = "user") -> dict:
    scars = load_json(SCARS_PATH)
    seq_id = get_next_sequential_id(scars)
    scar_uuid = str(uuid.uuid4())
    weight = compute_weight(content, tension_index)

    scar = {
        "seq_id": seq_id,
        "id": scar_uuid,
        "content": content,
        "tension_index": tension_index,
        "weight": weight,
        "generation": generation,
        "parent_chunk_id": parent_chunk_id,
        "parent_scar_ids": parent_scar_ids or [],
        "source": source
    }

    scars.append(scar)
    save_json(SCARS_PATH, scars)
    return scar


def add_lineage(chunk_id: str, scar_id: str,
                chunk_text_content: str, generation: int):
    lineage = load_json(LINEAGE_PATH)
    entry = {
        "chunk_id": chunk_id,
        "scar_id": scar_id,
        "chunk_preview": chunk_text_content[:80],
        "generation": generation
    }
    lineage.append(entry)
    save_json(LINEAGE_PATH, lineage)


# ══════════════════════════════════════════════════════════════
# PHI TRACKING
# ══════════════════════════════════════════════════════════════

def get_phi_ratio(scars: list, window: int = 5) -> float:
    weights = [s["weight"] for s in scars if s.get("weight", 0) > 0]
    if len(weights) < 2:
        return 1.0
    ratios = []
    for i in range(max(0, len(weights) - window), len(weights) - 1):
        if weights[i] > 0:
            ratios.append(weights[i + 1] / weights[i])
    return sum(ratios) / len(ratios) if ratios else 1.0


def print_phi_status(scars: list, generation: int):
    if len(scars) < 2:
        return
    ratio = get_phi_ratio(scars)
    distance = abs(ratio - PHI)
    bar_width = 20
    convergence = max(0, 1.0 - (distance / PHI))
    filled = int(bar_width * convergence)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n[φ] Gen {generation} | "
          f"ratio={ratio:.4f} | "
          f"dist={distance:.4f} | "
          f"[{bar}] {convergence:.0%}")
    if distance < 0.05:
        print(f"[φ] *** CONVERGING ON φ = {PHI:.4f} ***")


# ══════════════════════════════════════════════════════════════
# SINGLE CYCLE
# ══════════════════════════════════════════════════════════════

def run_cycle_on_chunk(chunk: str, generation: int,
                        parent_scar_ids: list = None,
                        source: str = "user") -> Optional[dict]:
    chunk_id = str(uuid.uuid4())
    current_scars = load_json(SCARS_PATH)
    memory_count = len(current_scars)

    print(f"\n{'─' * 50}")
    print(f"[CYCLE] Generation {generation} | "
          f"Memory: {memory_count} scars | "
          f"Source: {source}")
    print(f"[CYCLE] Input: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    raw = call_llm(chunk)
    if not raw:
        return None

    parsed = parse_cycle_output(raw)

    scar_text = parsed.get("scar", "").strip()
    tension_index = parsed.get("tension_index", 0.5)

    if not scar_text:
        scar_text = "(no scar produced)"
        tension_index = 0.1

    scar = add_scar_with_source(
        content=scar_text,
        tension_index=tension_index,
        parent_chunk_id=chunk_id,
        generation=generation,
        parent_scar_ids=parent_scar_ids or [],
        source=source
    )

    add_lineage(chunk_id, scar["id"], chunk, generation)

    print(f"\n{'═' * 50}")
    print(f"PRIME MOVE CYCLE — Generation {generation} [{source.upper()}]")
    print(f"{'═' * 50}")
    print(f"SPLIT:        {parsed.get('split','(none)')[:100]}")
    print(f"TENSION:      {parsed.get('tension','(none)')[:100]}")
    print(f"FAILED MERGE: {parsed.get('failed_merge','(none)')[:100]}")
    print(f"SCAR [{scar['seq_id']}]:    {scar_text}")
    print(f"TENSION_IDX:  {tension_index:.2f}")
    print(f"WEIGHT:       {scar['weight']:.4f}")
    print(f"DECAY:        {parsed.get('decay','(none)')[:100]}")
    print(f"{'═' * 50}")

    return scar


# ══════════════════════════════════════════════════════════════
# HEARTBEAT — AUTONOMOUS CYCLES ON A TIMER
# ══════════════════════════════════════════════════════════════

def heartbeat_loop(interval_minutes: int):
    """
    Annie fires an autonomous cycle every N minutes.
    Seeds each cycle from the highest-weight scar in the substrate.
    This is Annie initiating her own Splits without waiting for input.
    Runs in a background thread — you can still use all other commands.
    Type 'stopbeat' to pause.
    """
    interval_seconds = interval_minutes * 60
    print(f"\n[HEARTBEAT] Started — autonomous cycle every {interval_minutes} min")
    print(f"[HEARTBEAT] Type 'stopbeat' to pause\n")

    while heartbeat_active[0]:
        # Wait the interval, checking every second if stopped
        for _ in range(interval_seconds):
            if not heartbeat_active[0]:
                break
            time.sleep(1)

        if not heartbeat_active[0]:
            break

        print(f"\n[HEARTBEAT] ♦ Firing autonomous cycle...")

        scars = load_json(SCARS_PATH)
        if not scars:
            print("[HEARTBEAT] No scars yet — skipping cycle.")
            continue

        # Seed from highest-weight scar
        sorted_scars = sorted(
            scars, key=lambda x: x.get("weight", 0), reverse=True
        )
        seed_scar = sorted_scars[0]
        generation = max(s["generation"] for s in scars) + 1

        print(f"[HEARTBEAT] Seeding from Scar {seed_scar['seq_id']} "
              f"[w={seed_scar['weight']:.4f}]:")
        print(f"[HEARTBEAT] {seed_scar['content'][:80]}...")

        scar = run_cycle_on_chunk(
            seed_scar["content"],
            generation,
            parent_scar_ids=[seed_scar["id"]],
            source="heartbeat"
        )

        if scar:
            all_scars = load_json(SCARS_PATH)
            print_phi_status(all_scars, generation)
            print(f"\n[HEARTBEAT] Next cycle in {interval_minutes} min. "
                  f"Type 'stopbeat' to pause.\n> ", end="", flush=True)

    print(f"\n[HEARTBEAT] Stopped.")


def start_heartbeat(interval_minutes: int):
    heartbeat_active[0] = True
    t = threading.Thread(
        target=heartbeat_loop,
        args=(interval_minutes,),
        daemon=True
    )
    t.start()


def stop_heartbeat():
    heartbeat_active[0] = False


# ══════════════════════════════════════════════════════════════
# AUTONOMOUS LOOP
# ══════════════════════════════════════════════════════════════

def autonomous_loop(initial_text: str, max_generations: int = 10):
    print(f"\n[ANNIE SEED] Starting autonomous loop")
    print(f"[ANNIE SEED] Max generations: {max_generations}")
    print(f"[ANNIE SEED] Initial: {initial_text[:80]}...")

    generation = 0
    current_text = initial_text
    previous_scar_id = None

    while generation < max_generations:
        generation += 1

        chunks = chunk_text(current_text)
        if not chunks:
            print(f"\n[ANNIE SEED] No chunks. Stopping.")
            break

        parent_ids = [previous_scar_id] if previous_scar_id else []
        scar = run_cycle_on_chunk(chunks[0], generation, parent_ids)

        if not scar:
            print(f"\n[ANNIE SEED] Cycle failed at generation {generation}.")
            break

        previous_scar_id = scar["id"]
        current_text = scar["content"]

        all_scars = load_json(SCARS_PATH)
        print_phi_status(all_scars, generation)

        if generation > 1:
            recent = all_scars[-min(3, len(all_scars)):]
            avg_length = sum(len(s["content"]) for s in recent) / len(recent)
            avg_weight = sum(s["weight"] for s in recent) / len(recent)
            print(f"\n[SEED] Avg length: {avg_length:.0f} | "
                  f"Avg weight: {avg_weight:.4f}")

    print(f"\n[ANNIE SEED] Loop complete — {generation} generations")
    show_status()


# ══════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════

def show_status():
    scars = load_json(SCARS_PATH)
    lineage = load_json(LINEAGE_PATH)

    if not scars:
        print("\n[STATUS] No scars yet.")
        return

    weights = [s["weight"] for s in scars]
    tensions = [s["tension_index"] for s in scars]
    lengths = [len(s["content"]) for s in scars]

    # Count by source
    user_scars = len([s for s in scars if s.get("source") == "user"])
    heartbeat_scars = len([s for s in scars if s.get("source") == "heartbeat"])
    auto_scars = len([s for s in scars if s.get("source") not in ("user","heartbeat")])

    phi_ratio = get_phi_ratio(scars)
    phi_distance = abs(phi_ratio - PHI)

    print(f"\n{'═' * 50}")
    print(f"ANNIE SEED v0.3 STATUS")
    print(f"{'═' * 50}")
    print(f"Total scars:      {len(scars)}")
    print(f"  User:           {user_scars}")
    print(f"  Heartbeat:      {heartbeat_scars}")
    print(f"  Auto:           {auto_scars}")
    print(f"Total lineage:    {len(lineage)} entries")
    print(f"Generations:      {max(s['generation'] for s in scars)}")
    print(f"Heartbeat:        {'RUNNING' if heartbeat_active[0] else 'STOPPED'}")
    print(f"\nScar metrics:")
    print(f"  Avg length:     {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Avg tension:    {sum(tensions)/len(tensions):.3f}")
    print(f"  Avg weight:     {sum(weights)/len(weights):.4f}")
    print(f"  Min weight:     {min(weights):.4f}")
    print(f"  Max weight:     {max(weights):.4f}")
    print(f"\nPhi convergence:")
    print(f"  Rolling ratio:  {phi_ratio:.4f}")
    print(f"  Distance from φ:{phi_distance:.4f}")
    print(f"  Target φ:       {PHI:.4f}")

    print(f"\nRecent scars (last 5):")
    for s in scars[-5:]:
        src = f"[{s.get('source','?')}]"
        print(f"  Scar {s['seq_id']} {src} "
              f"t={s['tension_index']:.2f} w={s['weight']:.4f}: "
              f"{s['content'][:55]}{'...' if len(s['content']) > 55 else ''}")

    print(f"\nMemory context (top {min(MAX_CONTEXT_SCARS, len(scars))} by weight):")
    sorted_scars = sorted(scars, key=lambda x: x.get("weight", 0), reverse=True)
    for s in sorted_scars[:MAX_CONTEXT_SCARS]:
        print(f"  Scar {s['seq_id']} w={s['weight']:.4f}: "
              f"{s['content'][:55]}{'...' if len(s['content']) > 55 else ''}")


def show_lineage():
    scars = load_json(SCARS_PATH)

    if not scars:
        print("\n[LINEAGE] No scars yet.")
        return

    print(f"\n{'═' * 50}")
    print(f"LINEAGE TREE")
    print(f"{'═' * 50}")

    child_map = {}
    for s in scars:
        for pid in s.get("parent_scar_ids", []):
            child_map.setdefault(pid, []).append(s)

    roots = [s for s in scars if not s.get("parent_scar_ids")]

    def print_tree(scar, depth=0):
        indent = "  " * depth
        connector = "└─" if depth > 0 else "●"
        src = f"[{scar.get('source','?')}]"
        print(f"{indent}{connector} Scar {scar['seq_id']} {src} "
              f"[Gen {scar['generation']}] "
              f"t={scar['tension_index']:.2f} "
              f"w={scar['weight']:.4f}")
        print(f"{indent}   {scar['content'][:70]}"
              f"{'...' if len(scar['content']) > 70 else ''}")
        for child in child_map.get(scar["id"], []):
            print_tree(child, depth + 1)

    for root in roots:
        print_tree(root)
        print()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── ANNIE SEED v0.3 — PRIME MOVE ENGINE ──")
    print("── Memory + Heartbeat. Claude-powered. ──\n")

    if not ANTHROPIC_API_KEY:
        print("⚠ WARNING: No API key set.")
        print('Run: export ANTHROPIC_API_KEY="your-key-here"')
        print("Then restart Annie.\n")

    print("Commands:")
    print("  run           — run one cycle on your input")
    print("  auto          — autonomous loop (scar feeds forward)")
    print("  heartbeat     — start autonomous cycles on a timer")
    print("  stopbeat      — stop the heartbeat")
    print("  status        — show scar log, phi convergence, heartbeat state")
    print("  lineage       — show scar lineage tree")
    print("  exit          — quit\n")

    while True:
        command = input("> ").strip().lower()

        if command == "exit":
            if heartbeat_active[0]:
                stop_heartbeat()
            break

        elif command == "run":
            text = input("Input text: ").strip()
            if text:
                scars = load_json(SCARS_PATH)
                generation = (max(s["generation"] for s in scars) + 1
                             if scars else 1)
                scar = run_cycle_on_chunk(text, generation, source="user")
                if scar:
                    all_scars = load_json(SCARS_PATH)
                    print_phi_status(all_scars, generation)

        elif command == "auto":
            text = input("Initial text: ").strip()
            if text:
                try:
                    gens = int(input("Max generations (default 10): ").strip() or "10")
                except ValueError:
                    gens = 10
                autonomous_loop(text, max_generations=gens)

        elif command == "heartbeat":
            if heartbeat_active[0]:
                print("[HEARTBEAT] Already running. Type 'stopbeat' first.")
            else:
                try:
                    mins = int(input("Interval in minutes (default 10): ").strip() or "10")
                except ValueError:
                    mins = 10
                start_heartbeat(mins)

        elif command == "stopbeat":
            if heartbeat_active[0]:
                stop_heartbeat()
            else:
                print("[HEARTBEAT] Not running.")

        elif command == "status":
            show_status()

        elif command == "lineage":
            show_lineage()

        else:
            print("Commands: run | auto | heartbeat | stopbeat | status | lineage | exit")
