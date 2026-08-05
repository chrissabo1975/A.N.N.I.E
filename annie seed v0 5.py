"""
Annie Seed v0.5 — Prime Move Generative Engine
===============================================
Changes from v0.4:
- API key via environment variable (no hardcoded keys)
- Reverse pruning: compresses scar clusters into meta-scars
  via backward cycle (Decay → Scar → Failed Merge)
  Activates only when cluster size >= PRUNE_MIN_CLUSTER
  and domain scar count >= PRUNE_MIN_DOMAIN_SCARS

SETUP:
1. pip3 install requests
2. export ANTHROPIC_API_KEY="your-key-here"
3. python3 annie_seed_v0_5.py

To make the key permanent:
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc

COMMANDS:
  run          - run one cycle
  auto         - autonomous loop
  heartbeat    - timed autonomous cycles
  stopbeat     - stop heartbeat
  prune        - run reverse pruning on eligible clusters
  prunestatus  - show pruning eligibility
  status       - scar log and network summary
  lineage      - lineage tree
  network [n]  - connections for Scar n
  exit         - quit
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

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

# API key from environment variable — never hardcode this
# Set with: export ANTHROPIC_API_KEY="your-key-here"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# File paths — change to full paths for dedicated sandbox
# Example: "/Users/yourname/Annie7/logs/annie_seed_scars.json"
SCARS_PATH = "annie_seed_scars.json"
LINEAGE_PATH = "annie_seed_lineage.json"
ARCHIVE_PATH = "annie_seed_archive.json"

# Model
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

# Memory
MAX_CONTEXT_SCARS = 5
DEFAULT_HEARTBEAT_MINUTES = 10

# Pruning thresholds
PRUNE_MIN_CLUSTER = 5        # Min scars in a cluster to prune
PRUNE_MIN_DOMAIN_SCARS = 10  # Min domain scars before pruning activates
PRUNE_LOW_WEIGHT = 0.15      # Scars below this are pruning candidates
PRUNE_LOW_TENSION = 0.40     # Scars below this are pruning candidates
META_SCAR_MAX_LENGTH = 150   # Meta-scar must be shorter than this

# Network
ACTIVATION_DECAY = 0.5
ACTIVATION_THRESHOLD = 0.25
MAX_SPREAD_DEPTH = 3
CO_ACTIVATION_INCREMENT = 0.1

STOP_WORDS = {
    'the','a','an','is','are','was','were','to','of','in','for',
    'on','with','at','by','and','or','but','not','it','its',
    'this','that','be','been','have','has','i','you','we','they',
    'my','your','our','their','what','which','who','how','when',
    'where','why','do','does','did','will','would','could','should',
    'may','might','must','can','if','as','so','no','all','one',
    'from','into','than','then','only','also','just'
}

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PHI = (1 + math.sqrt(5)) / 2
heartbeat_active = [False]

# ══════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ANNIE_SEED_v0_5 — a Prime Move generative engine.

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

REVERSE_CYCLE_PROMPT = """You are running the Prime Move cycle IN REVERSE on a cluster of scars.

These scars are related structural residues that can be compressed
into a single more fundamental distinction.

Run the reverse cycle in this order:

### DECAY
What is fading or releasing across this cluster?
What do these scars have in common that is dissolving?

### FAILED MERGE
What could the cluster NOT resolve?
What tension persisted across all of these scars?

### META_SCAR
The single irreducible distinction the cluster collectively expresses.
ONE sentence. Shorter than any scar in the cluster.
Must capture what ALL scars are pointing at.
TENSION_INDEX: [0.0-1.0 — should be HIGHER than cluster average]

CRITICAL: The meta-scar must be:
- Shorter than {max_length} characters
- Higher tension than cluster average ({avg_tension:.2f})
- The essence, not a summary
"""


# ══════════════════════════════════════════════════════════════
# STORAGE
# ══════════════════════════════════════════════════════════════

def load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_scar_by_id(scar_id: str, scars: list) -> Optional[dict]:
    for s in scars:
        if s["id"] == scar_id:
            return s
    return None


# ══════════════════════════════════════════════════════════════
# WEIGHT AND SIMILARITY
# ══════════════════════════════════════════════════════════════

def compute_weight(content: str, tension_index: float) -> float:
    if not content:
        return 0.0
    return tension_index / math.log(len(content) + 1)


def extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def compute_similarity(text_a: str, text_b: str) -> float:
    wa = extract_keywords(text_a)
    wb = extract_keywords(text_b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def compute_recency(gen_a: int, gen_b: int, max_gen: int) -> float:
    if max_gen == 0:
        return 1.0
    return max(0.0, 1.0 - (abs(gen_a - gen_b) / max(max_gen, 1)))


def compute_lineage(parents_a: list, id_b: str,
                    parents_b: list, id_a: str) -> float:
    if id_b in parents_a or id_a in parents_b:
        return 1.0
    if set(parents_a) & set(parents_b):
        return 0.5
    return 0.0


# ══════════════════════════════════════════════════════════════
# NETWORK
# ══════════════════════════════════════════════════════════════

def rebuild_connections(scars: list) -> list:
    if len(scars) < 2:
        return scars
    max_gen = max(s["generation"] for s in scars) if scars else 1
    for i, sa in enumerate(scars):
        if "connections" not in sa:
            sa["connections"] = {}
        for j, sb in enumerate(scars):
            if i == j:
                continue
            existing = sa["connections"].get(sb["id"], {})
            co_act = existing.get("co_activation", 0.0) if isinstance(existing, dict) else 0.0
            sim = compute_similarity(sa["content"], sb["content"])
            rec = compute_recency(sa["generation"], sb["generation"], max_gen)
            lin = compute_lineage(sa.get("parent_scar_ids", []), sb["id"],
                                  sb.get("parent_scar_ids", []), sa["id"])
            total = sim * 0.4 + rec * 0.2 + lin * 0.2 + co_act * 0.2
            sa["connections"][sb["id"]] = {
                "total": round(total, 4),
                "similarity": round(sim, 4),
                "recency": round(rec, 4),
                "lineage": round(lin, 4),
                "co_activation": round(co_act, 4)
            }
    return scars


def update_co_activation(activated_ids: list, scars: list) -> list:
    for i in range(len(activated_ids)):
        for j in range(i + 1, len(activated_ids)):
            id_a, id_b = activated_ids[i], activated_ids[j]
            for scar in scars:
                if scar["id"] in (id_a, id_b):
                    other = id_b if scar["id"] == id_a else id_a
                    if "connections" not in scar:
                        scar["connections"] = {}
                    conn = scar["connections"].get(other, {})
                    if isinstance(conn, dict):
                        conn["co_activation"] = min(
                            1.0, conn.get("co_activation", 0.0) + CO_ACTIVATION_INCREMENT
                        )
                        scar["connections"][other] = conn
    return scars


def find_seed_scar(input_text: str, scars: list) -> Optional[dict]:
    if not scars:
        return None
    best, best_score = None, -1.0
    for scar in scars:
        sim = compute_similarity(input_text, scar["content"])
        score = sim * 0.7 + scar.get("weight", 0) * 0.3
        if score > best_score:
            best_score = score
            best = scar
    return best


def spread_activation(seed_id: str, scars: list) -> list:
    activation = {seed_id: 1.0}
    frontier = [seed_id]
    visited = set()

    for _ in range(MAX_SPREAD_DEPTH):
        next_frontier = []
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            scar = get_scar_by_id(node_id, scars)
            if not scar or "connections" not in scar:
                continue
            for target_id, conn_data in scar["connections"].items():
                if target_id in visited:
                    continue
                weight = conn_data.get("total", 0.0) if isinstance(conn_data, dict) else 0.0
                spread = activation[node_id] * weight * ACTIVATION_DECAY
                activation[target_id] = max(activation.get(target_id, 0.0), spread)
                if activation[target_id] > ACTIVATION_THRESHOLD:
                    next_frontier.append(target_id)
        frontier = next_frontier
        if not frontier:
            break

    activated = [(sid, lvl) for sid, lvl in activation.items()
                 if lvl >= ACTIVATION_THRESHOLD]
    activated.sort(key=lambda x: x[1], reverse=True)

    result = []
    for scar_id, level in activated:
        scar = get_scar_by_id(scar_id, scars)
        if scar:
            result.append((scar, level))
    return result


# ══════════════════════════════════════════════════════════════
# MEMORY CONTEXT
# ══════════════════════════════════════════════════════════════

def build_scar_context(input_text: str = "") -> tuple:
    scars = load_json(SCARS_PATH)
    if not scars:
        return "", []

    seed = (find_seed_scar(input_text, scars) if input_text
            else max(scars, key=lambda x: x.get("weight", 0)))
    if not seed:
        return "", []

    activated = spread_activation(seed["id"], scars)
    top = activated[:MAX_CONTEXT_SCARS]
    activated_ids = [s["id"] for s, _ in top]

    if len(activated_ids) > 1:
        updated = update_co_activation(activated_ids, scars)
        save_json(SCARS_PATH, updated)

    lines = ["### ASSOCIATIVE MEMORY (activated by current input)"]
    lines.append(f"Seed: Scar {seed['seq_id']} — {seed['content']}")
    lines.append("Activated associations:\n")

    for scar, level in top:
        if scar["id"] == seed["id"]:
            continue
        lines.append(
            f"Scar {scar['seq_id']} "
            f"[activation={level:.3f}] "
            f"[tension={scar['tension_index']:.2f}]: "
            f"{scar['content']}"
        )

    lines.append("\nLet these inform — but do not repeat — your current cycle.")
    return "\n".join(lines), activated_ids


# ══════════════════════════════════════════════════════════════
# REVERSE PRUNING
# ══════════════════════════════════════════════════════════════

def identify_prune_clusters(scars: list) -> list:
    """
    Find clusters of low-weight, low-tension scars in the same domain.
    Only activates when domain has enough scars to justify compression.
    Returns list of clusters (each a list of scars).
    """
    candidates = [
        s for s in scars
        if (s.get("weight", 0) < PRUNE_LOW_WEIGHT and
            s.get("tension_index", 0) < PRUNE_LOW_TENSION and
            s.get("source") not in ("pruning_meta",) and
            not s.get("is_meta_scar", False))
    ]

    # Group by domain
    domain_groups = {}
    for scar in candidates:
        d = scar.get("domain", "general")
        domain_groups.setdefault(d, []).append(scar)

    # Count total scars per domain in full log
    domain_totals = {}
    for scar in scars:
        d = scar.get("domain", "general")
        domain_totals[d] = domain_totals.get(d, 0) + 1

    clusters = []
    for domain, cluster_scars in domain_groups.items():
        if (len(cluster_scars) >= PRUNE_MIN_CLUSTER and
                domain_totals.get(domain, 0) >= PRUNE_MIN_DOMAIN_SCARS):
            cluster_scars.sort(key=lambda x: x.get("weight", 0))
            clusters.append(cluster_scars)

    return clusters


def run_reverse_cycle(cluster: list) -> Optional[dict]:
    """
    Run the reverse Prime Move cycle on a cluster of scars.
    Stops at Failed Merge — does not complete the full cycle.
    Produces a meta-scar: the irreducible essence of the cluster.
    """
    avg_tension = sum(s["tension_index"] for s in cluster) / len(cluster)
    avg_length = sum(len(s["content"]) for s in cluster) / len(cluster)

    cluster_text = "\n".join(
        f"  Scar {s['seq_id']} [t={s['tension_index']:.2f}] "
        f"[w={s['weight']:.4f}]: {s['content']}"
        for s in cluster
    )

    prompt = REVERSE_CYCLE_PROMPT.format(
        max_length=META_SCAR_MAX_LENGTH,
        avg_tension=avg_tension
    ) + f"\n\nCLUSTER ({len(cluster)} scars):\n{cluster_text}"

    if not ANTHROPIC_API_KEY:
        print("[PRUNE] No API key — cannot run reverse cycle.")
        return None

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
                "max_tokens": 512,
                "system": "You compress scar clusters into meta-scars via reverse Prime Move cycle.",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        raw = None
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                raw = block.get("text", "").strip()
                break

        if not raw:
            return None

        # Extract meta-scar
        meta_content = None
        meta_match = re.search(
            r"###\s*META_SCAR\s*\n(.*?)(?=TENSION_INDEX|###|\Z)",
            raw, re.DOTALL | re.IGNORECASE
        )
        if meta_match:
            meta_content = re.sub(r'\n+', ' ', meta_match.group(1).strip()).strip()

        tension_index = avg_tension * 1.2  # Default: higher than cluster
        ti_match = re.search(
            r"TENSION_INDEX:\s*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE
        )
        if ti_match:
            tension_index = max(0.0, min(1.0, float(ti_match.group(1))))

        if not meta_content:
            return None

        return {
            "content": meta_content,
            "tension_index": tension_index,
            "raw": raw
        }

    except Exception as e:
        print(f"[PRUNE] Reverse cycle error: {e}")
        return None


def validate_meta_scar(meta: dict, cluster: list) -> tuple:
    """
    Quality gate for meta-scars.
    Must be shorter, more precise, and higher tension than cluster.
    Returns (passes: bool, reason: str)
    """
    avg_tension = sum(s["tension_index"] for s in cluster) / len(cluster)
    avg_length = sum(len(s["content"]) for s in cluster) / len(cluster)
    avg_weight = sum(s.get("weight", 0) for s in cluster) / len(cluster)

    content = meta["content"]
    tension = meta["tension_index"]

    if len(content) > META_SCAR_MAX_LENGTH:
        return False, f"Too long ({len(content)} chars, max {META_SCAR_MAX_LENGTH})"

    if len(content) >= avg_length:
        return False, f"Not shorter than cluster average ({len(content)} >= {avg_length:.0f})"

    if tension <= avg_tension:
        return False, f"Tension not higher than cluster ({tension:.3f} <= {avg_tension:.3f})"

    meta_weight = compute_weight(content, tension)
    if meta_weight <= avg_weight:
        return False, f"Weight not higher than cluster ({meta_weight:.4f} <= {avg_weight:.4f})"

    return True, "Passes all gates"


def archive_cluster(cluster: list, meta_scar: dict):
    """Move compressed scars to archive. They are never deleted."""
    archive = load_json(ARCHIVE_PATH)
    for scar in cluster:
        archived = dict(scar)
        archived["archived"] = True
        archived["compressed_into"] = meta_scar["id"]
        archive.append(archived)
    save_json(ARCHIVE_PATH, archive)


def prune(domain_filter: str = None):
    """
    Run reverse pruning on eligible clusters.
    Compresses low-weight scar clusters into meta-scars.
    Archives compressed scars — nothing is deleted.
    """
    scars = load_json(SCARS_PATH)

    if not scars:
        print("[PRUNE] No scars yet.")
        return

    clusters = identify_prune_clusters(scars)

    if domain_filter:
        clusters = [c for c in clusters if c[0].get("domain") == domain_filter]

    if not clusters:
        print("[PRUNE] No eligible clusters found.")
        print(f"[PRUNE] Need {PRUNE_MIN_CLUSTER}+ low-weight scars in a domain")
        print(f"[PRUNE] with {PRUNE_MIN_DOMAIN_SCARS}+ total scars in that domain.")
        return

    print(f"\n[PRUNE] Found {len(clusters)} eligible cluster(s)")
    compressed = 0

    for cluster in clusters:
        domain = cluster[0].get("domain", "general")
        avg_t = sum(s["tension_index"] for s in cluster) / len(cluster)
        print(f"\n[PRUNE] Cluster: {len(cluster)} scars in '{domain}' "
              f"(avg tension={avg_t:.3f})")

        # Run reverse cycle
        meta_result = run_reverse_cycle(cluster)
        if not meta_result:
            print(f"[PRUNE] Reverse cycle failed — skipping cluster.")
            continue

        print(f"[PRUNE] Meta-scar candidate: {meta_result['content']}")
        print(f"[PRUNE] Meta tension: {meta_result['tension_index']:.3f}")

        # Validate
        passes, reason = validate_meta_scar(meta_result, cluster)
        if not passes:
            print(f"[PRUNE] ✗ Quality gate failed: {reason}")
            continue

        print(f"[PRUNE] ✓ Quality gate passed")

        # Add meta-scar to active memory
        scars = load_json(SCARS_PATH)
        seq_id = get_next_seq_id(scars)
        meta_scar = {
            "seq_id": seq_id,
            "id": str(uuid.uuid4()),
            "content": meta_result["content"],
            "tension_index": meta_result["tension_index"],
            "weight": compute_weight(
                meta_result["content"], meta_result["tension_index"]
            ),
            "generation": max(s["generation"] for s in scars),
            "parent_chunk_id": str(uuid.uuid4()),
            "parent_scar_ids": [s["id"] for s in cluster],
            "source": "pruning_meta",
            "is_meta_scar": True,
            "domain": domain,
            "connections": {}
        }

        # Archive the cluster
        archive_cluster(cluster, meta_scar)

        # Remove cluster from active scars
        cluster_ids = {s["id"] for s in cluster}
        scars = [s for s in scars if s["id"] not in cluster_ids]

        # Add meta-scar and rebuild connections
        scars.append(meta_scar)
        scars = rebuild_connections(scars)
        save_json(SCARS_PATH, scars)

        print(f"[PRUNE] Compressed {len(cluster)} scars → Meta-scar {seq_id}")
        compressed += 1

    print(f"\n[PRUNE] Complete: {compressed}/{len(clusters)} clusters compressed.")
    archive = load_json(ARCHIVE_PATH)
    print(f"[PRUNE] Archive now contains {len(archive)} scars.")


def show_prune_status():
    """Show pruning eligibility without running anything."""
    scars = load_json(SCARS_PATH)
    if not scars:
        print("\n[PRUNE] No scars yet.")
        return

    candidates = [
        s for s in scars
        if (s.get("weight", 0) < PRUNE_LOW_WEIGHT and
            s.get("tension_index", 0) < PRUNE_LOW_TENSION)
    ]

    clusters = identify_prune_clusters(scars)

    domain_totals = {}
    for s in scars:
        d = s.get("domain", "general")
        domain_totals[d] = domain_totals.get(d, 0) + 1

    archive = load_json(ARCHIVE_PATH)

    print(f"\n{'═' * 50}")
    print(f"PRUNING STATUS")
    print(f"{'═' * 50}")
    print(f"Total active scars:    {len(scars)}")
    print(f"Archived scars:        {len(archive)}")
    print(f"Prune candidates:      {len(candidates)}")
    print(f"  (w<{PRUNE_LOW_WEIGHT}, t<{PRUNE_LOW_TENSION})")
    print(f"Eligible clusters:     {len(clusters)}")
    print(f"  (need {PRUNE_MIN_CLUSTER}+ candidates, {PRUNE_MIN_DOMAIN_SCARS}+ domain scars)")

    if clusters:
        print(f"\nReady to compress:")
        for c in clusters:
            domain = c[0].get("domain", "general")
            avg_t = sum(s["tension_index"] for s in c) / len(c)
            print(f"  '{domain}': {len(c)} scars, avg tension={avg_t:.3f}")
    else:
        print(f"\nNo clusters ready yet.")
        print(f"Keep running cycles to build domain density.")

        # Show progress toward threshold
        print(f"\nDomain counts:")
        for domain, count in sorted(domain_totals.items(),
                                     key=lambda x: x[1], reverse=True):
            bar = "█" * min(count, 20)
            print(f"  {domain}: {count} scars {bar} "
                  f"(need {PRUNE_MIN_DOMAIN_SCARS})")


# ══════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════

def call_llm(user_text: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        print("\n[ERROR] No API key found.")
        print("[ERROR] Run: export ANTHROPIC_API_KEY=\"your-key-here\"")
        print("[ERROR] Then restart Annie.")
        return None

    scar_context, _ = build_scar_context(user_text)
    user_message = (
        scar_context + "\n\n### CURRENT INPUT\n" + user_text
        if scar_context else user_text
    )

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
                "messages": [{"role": "user", "content": user_message}]
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "").strip()
        print(f"[ERROR] Unexpected response: {data}")
        return None

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Anthropic API.")
        return None
    except requests.exceptions.Timeout:
        print("\n[ERROR] API call timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("\n[ERROR] Invalid API key.")
        elif response.status_code == 429:
            print("\n[ERROR] Rate limit. Wait and retry.")
        else:
            print(f"\n[ERROR] {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_cycle_output(raw: str) -> dict:
    sections = {
        "split": "", "tension": "", "failed_merge": "",
        "scar": "", "decay": "", "tension_index": 0.5
    }
    current = None
    for line in raw.splitlines():
        ls = line.strip()
        if ls.startswith("### "):
            key = ls[4:].lower().replace(" ", "_")
            current = key if key in sections else None
            continue
        ti = re.search(r"TENSION_INDEX:\s*([0-9]*\.?[0-9]+)", ls, re.I)
        if ti:
            try:
                sections["tension_index"] = max(0.0, min(1.0, float(ti.group(1))))
            except ValueError:
                pass
            if current == "scar":
                continue
        if current and current != "tension_index":
            sections[current] = (
                sections[current] + " " + ls if sections[current] else ls
            )
    sections["scar"] = re.sub(
        r"TENSION_INDEX:\s*[0-9]*\.?[0-9]+", "", sections["scar"]
    ).strip()
    return sections


# ══════════════════════════════════════════════════════════════
# SCAR STORAGE
# ══════════════════════════════════════════════════════════════

def get_next_seq_id(scars: list) -> int:
    if not scars:
        return 1
    return max(s.get("seq_id", 0) for s in scars) + 1


def add_scar(content: str, tension_index: float, parent_chunk_id: str,
             generation: int, parent_scar_ids: list = None,
             source: str = "user", domain: str = "general") -> dict:
    scars = load_json(SCARS_PATH)
    new_scar = {
        "seq_id": get_next_seq_id(scars),
        "id": str(uuid.uuid4()),
        "content": content,
        "tension_index": tension_index,
        "weight": compute_weight(content, tension_index),
        "generation": generation,
        "parent_chunk_id": parent_chunk_id,
        "parent_scar_ids": parent_scar_ids or [],
        "source": source,
        "domain": domain,
        "is_meta_scar": False,
        "connections": {}
    }
    scars.append(new_scar)
    scars = rebuild_connections(scars)
    save_json(SCARS_PATH, scars)
    return get_scar_by_id(new_scar["id"], scars) or new_scar


def add_lineage(chunk_id: str, scar_id: str,
                chunk_preview: str, generation: int):
    lineage = load_json(LINEAGE_PATH)
    lineage.append({
        "chunk_id": chunk_id,
        "scar_id": scar_id,
        "chunk_preview": chunk_preview[:80],
        "generation": generation
    })
    save_json(LINEAGE_PATH, lineage)


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
            for s in sentences:
                if len(current) + len(s) > max_chunk and current:
                    chunks.append(current.strip())
                    current = s
                else:
                    current = (current + " " + s).strip()
            if current:
                chunks.append(current)
        else:
            chunks.append(raw)
    return [c for c in chunks if c]


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
    convergence = max(0, 1.0 - (distance / PHI))
    filled = int(20 * convergence)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"\n[φ] Gen {generation} | ratio={ratio:.4f} | "
          f"dist={distance:.4f} | [{bar}] {convergence:.0%}")
    if distance < 0.05:
        print(f"[φ] *** CONVERGING ON φ ***")


# ══════════════════════════════════════════════════════════════
# SINGLE CYCLE
# ══════════════════════════════════════════════════════════════

def run_cycle_on_chunk(chunk: str, generation: int,
                        parent_scar_ids: list = None,
                        source: str = "user",
                        domain: str = "general") -> Optional[dict]:
    chunk_id = str(uuid.uuid4())
    scars = load_json(SCARS_PATH)

    if scars:
        seed = find_seed_scar(chunk, scars)
        if seed:
            activated = spread_activation(seed["id"], scars)
            print(f"\n[NETWORK] Seed: Scar {seed['seq_id']} — "
                  f"{seed['content'][:50]}...")
            print(f"[NETWORK] Activated {len(activated)} associations")

    print(f"\n{'─' * 50}")
    print(f"[CYCLE] Gen {generation} | Memory: {len(scars)} | Source: {source}")
    print(f"[CYCLE] {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    raw = call_llm(chunk)
    if not raw:
        return None

    parsed = parse_cycle_output(raw)
    scar_text = parsed.get("scar", "").strip() or "(no scar produced)"
    tension_index = parsed.get("tension_index", 0.5)
    if not parsed.get("scar", "").strip():
        tension_index = 0.1

    scar = add_scar(
        content=scar_text,
        tension_index=tension_index,
        parent_chunk_id=chunk_id,
        generation=generation,
        parent_scar_ids=parent_scar_ids or [],
        source=source,
        domain=domain
    )

    add_lineage(chunk_id, scar["id"], chunk, generation)

    print(f"\n{'═' * 50}")
    print(f"PRIME MOVE — Gen {generation} [{source.upper()}]")
    print(f"{'═' * 50}")
    print(f"SPLIT:        {parsed.get('split','')[:100]}")
    print(f"TENSION:      {parsed.get('tension','')[:100]}")
    print(f"FAILED MERGE: {parsed.get('failed_merge','')[:100]}")
    print(f"SCAR [{scar['seq_id']}]:    {scar_text}")
    print(f"TENSION:      {tension_index:.2f} | WEIGHT: {scar['weight']:.4f}")
    print(f"DECAY:        {parsed.get('decay','')[:100]}")
    print(f"{'═' * 50}")

    return scar


# ══════════════════════════════════════════════════════════════
# HEARTBEAT
# ══════════════════════════════════════════════════════════════

def heartbeat_loop(interval_minutes: int):
    interval_seconds = interval_minutes * 60
    print(f"\n[HEARTBEAT] Started — every {interval_minutes} min")
    print(f"[HEARTBEAT] 'stopbeat' to pause\n")

    while heartbeat_active[0]:
        for _ in range(interval_seconds):
            if not heartbeat_active[0]:
                break
            time.sleep(1)

        if not heartbeat_active[0]:
            break

        print(f"\n[HEARTBEAT] ♦ Firing...")
        scars = load_json(SCARS_PATH)

        if not scars:
            print("[HEARTBEAT] No scars — skipping.")
            continue

        seed_scar = max(scars, key=lambda x: x.get("weight", 0))
        generation = max(s["generation"] for s in scars) + 1

        scar = run_cycle_on_chunk(
            seed_scar["content"], generation,
            parent_scar_ids=[seed_scar["id"]],
            source="heartbeat",
            domain=seed_scar.get("domain", "general")
        )

        if scar:
            print_phi_status(load_json(SCARS_PATH), generation)
            print(f"\n[HEARTBEAT] Next in {interval_minutes} min. 'stopbeat' to pause.\n> ",
                  end="", flush=True)

    print(f"\n[HEARTBEAT] Stopped.")


def start_heartbeat(interval_minutes: int):
    heartbeat_active[0] = True
    threading.Thread(
        target=heartbeat_loop, args=(interval_minutes,), daemon=True
    ).start()


# ══════════════════════════════════════════════════════════════
# AUTONOMOUS LOOP
# ══════════════════════════════════════════════════════════════

def autonomous_loop(initial_text: str, max_generations: int = 10,
                    domain: str = "general"):
    print(f"\n[ANNIE] Autonomous loop — {max_generations} generations")
    generation = 0
    current_text = initial_text
    previous_scar_id = None

    while generation < max_generations:
        generation += 1
        chunks = chunk_text(current_text)
        if not chunks:
            break

        parent_ids = [previous_scar_id] if previous_scar_id else []
        scar = run_cycle_on_chunk(
            chunks[0], generation, parent_ids,
            source="auto", domain=domain
        )

        if not scar:
            break

        previous_scar_id = scar["id"]
        current_text = scar["content"]

        all_scars = load_json(SCARS_PATH)
        print_phi_status(all_scars, generation)

    print(f"\n[ANNIE] Loop complete — {generation} generations")
    show_status()


# ══════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════

def show_status():
    scars = load_json(SCARS_PATH)
    archive = load_json(ARCHIVE_PATH)
    lineage = load_json(LINEAGE_PATH)

    if not scars:
        print("\n[STATUS] No scars yet.")
        return

    weights = [s["weight"] for s in scars]
    tensions = [s["tension_index"] for s in scars]
    lengths = [len(s["content"]) for s in scars]

    user_n = len([s for s in scars if s.get("source") == "user"])
    hb_n = len([s for s in scars if s.get("source") == "heartbeat"])
    meta_n = len([s for s in scars if s.get("is_meta_scar")])
    total_conn = sum(len(s.get("connections", {})) for s in scars)

    phi_ratio = get_phi_ratio(scars)
    phi_dist = abs(phi_ratio - PHI)

    print(f"\n{'═' * 50}")
    print(f"ANNIE SEED v0.5 — STATUS")
    print(f"{'═' * 50}")
    print(f"Active scars:     {len(scars)}")
    print(f"  User:           {user_n}")
    print(f"  Heartbeat:      {hb_n}")
    print(f"  Auto:           {len(scars)-user_n-hb_n-meta_n}")
    print(f"  Meta-scars:     {meta_n}")
    print(f"Archived scars:   {len(archive)}")
    print(f"Network edges:    {total_conn}")
    print(f"Lineage entries:  {len(lineage)}")
    print(f"Heartbeat:        {'RUNNING' if heartbeat_active[0] else 'STOPPED'}")
    print(f"\nScar metrics:")
    print(f"  Avg length:     {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Avg tension:    {sum(tensions)/len(tensions):.3f}")
    print(f"  Avg weight:     {sum(weights)/len(weights):.4f}")
    print(f"  Max weight:     {max(weights):.4f}")
    print(f"\nPhi convergence:")
    print(f"  Rolling ratio:  {phi_ratio:.4f}")
    print(f"  Distance from φ:{phi_dist:.4f}")
    if phi_dist < 0.05:
        print(f"  *** CONVERGING ON φ ***")

    print(f"\nRecent scars (last 5):")
    for s in scars[-5:]:
        meta = " [META]" if s.get("is_meta_scar") else ""
        print(f"  Scar {s['seq_id']}{meta} [{s.get('source','?')}] "
              f"t={s['tension_index']:.2f} w={s['weight']:.4f}: "
              f"{s['content'][:50]}{'...' if len(s['content'])>50 else ''}")


def show_lineage():
    scars = load_json(SCARS_PATH)
    if not scars:
        print("\n[LINEAGE] No scars yet.")
        return

    print(f"\n{'═' * 50}\nLINEAGE TREE\n{'═' * 50}")
    child_map = {}
    for s in scars:
        for pid in s.get("parent_scar_ids", []):
            child_map.setdefault(pid, []).append(s)

    roots = [s for s in scars if not s.get("parent_scar_ids")]

    def print_tree(scar, depth=0):
        ind = "  " * depth
        con = "└─" if depth > 0 else "●"
        meta = " [META]" if scar.get("is_meta_scar") else ""
        print(f"{ind}{con} Scar {scar['seq_id']}{meta} "
              f"t={scar['tension_index']:.2f} w={scar['weight']:.4f}")
        print(f"{ind}   {scar['content'][:70]}"
              f"{'...' if len(scar['content'])>70 else ''}")
        for child in child_map.get(scar["id"], []):
            print_tree(child, depth + 1)

    for root in roots:
        print_tree(root)
        print()


def show_network(scar_seq_id: int):
    scars = load_json(SCARS_PATH)
    target = next((s for s in scars if s["seq_id"] == scar_seq_id), None)
    if not target:
        print(f"\n[NETWORK] Scar {scar_seq_id} not found.")
        return

    print(f"\n{'═' * 50}")
    print(f"NETWORK — Scar {scar_seq_id}")
    print(f"{'═' * 50}")
    print(f"Content: {target['content']}")
    print(f"Weight: {target['weight']:.4f} | Tension: {target['tension_index']:.2f}")
    if target.get("is_meta_scar"):
        print(f"[META-SCAR — compressed from {len(target.get('parent_scar_ids',[]))} scars]")

    connections = target.get("connections", {})
    sorted_conns = sorted(
        connections.items(),
        key=lambda x: x[1].get("total", 0) if isinstance(x[1], dict) else 0,
        reverse=True
    )

    print(f"\nTop connections:")
    for target_id, conn_data in sorted_conns[:10]:
        connected = get_scar_by_id(target_id, scars)
        if connected:
            total = conn_data.get("total", 0) if isinstance(conn_data, dict) else 0
            sim = conn_data.get("similarity", 0) if isinstance(conn_data, dict) else 0
            print(f"  → Scar {connected['seq_id']} "
                  f"[total={total:.3f} sim={sim:.3f}]: "
                  f"{connected['content'][:50]}...")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── ANNIE SEED v0.5 — PRIME MOVE GENERATIVE ENGINE ──")
    print("── Associative Memory + Reverse Pruning ──")
    print("── github.com/chrissabo1975/prime-move-theory ──\n")

    if not ANTHROPIC_API_KEY:
        print("⚠  No API key detected.")
        print('   Run: export ANTHROPIC_API_KEY="your-key-here"')
        print("   Then restart Annie.\n")

    print("Commands:")
    print("  run            — run one cycle")
    print("  auto           — autonomous loop")
    print("  heartbeat      — timed autonomous cycles")
    print("  stopbeat       — stop heartbeat")
    print("  prune          — compress low-weight clusters into meta-scars")
    print("  prunestatus    — show pruning eligibility")
    print("  status         — scar log and network summary")
    print("  lineage        — lineage tree")
    print("  network [n]    — connections for Scar n")
    print("  exit           — quit\n")

    while True:
        command = input("> ").strip().lower()

        if command == "exit":
            if heartbeat_active[0]:
                heartbeat_active[0] = False
            break

        elif command == "run":
            text = input("Input text: ").strip()
            if text:
                domain = input("Domain (enter for 'general'): ").strip() or "general"
                scars = load_json(SCARS_PATH)
                generation = (
                    max(s["generation"] for s in scars) + 1 if scars else 1
                )
                scar = run_cycle_on_chunk(
                    text, generation, source="user", domain=domain
                )
                if scar:
                    print_phi_status(load_json(SCARS_PATH), generation)

        elif command == "auto":
            text = input("Initial text: ").strip()
            if text:
                domain = input("Domain (enter for 'general'): ").strip() or "general"
                try:
                    gens = int(input("Max generations (default 10): ").strip() or "10")
                except ValueError:
                    gens = 10
                autonomous_loop(text, max_generations=gens, domain=domain)

        elif command == "heartbeat":
            if heartbeat_active[0]:
                print("[HEARTBEAT] Already running. 'stopbeat' first.")
            else:
                try:
                    mins = int(input(
                        f"Interval in minutes (default {DEFAULT_HEARTBEAT_MINUTES}): "
                    ).strip() or str(DEFAULT_HEARTBEAT_MINUTES))
                except ValueError:
                    mins = DEFAULT_HEARTBEAT_MINUTES
                start_heartbeat(mins)

        elif command == "stopbeat":
            if heartbeat_active[0]:
                heartbeat_active[0] = False
            else:
                print("[HEARTBEAT] Not running.")

        elif command == "prune":
            prune()

        elif command == "prunestatus":
            show_prune_status()

        elif command == "status":
            show_status()

        elif command == "lineage":
            show_lineage()

        elif command.startswith("network"):
            parts = command.split()
            if len(parts) > 1:
                try:
                    show_network(int(parts[1]))
                except ValueError:
                    print("Usage: network [scar_number]")
            else:
                print("Usage: network [scar_number]")

        else:
            print("Commands: run | auto | heartbeat | stopbeat | "
                  "prune | prunestatus | status | lineage | network [n] | exit")
