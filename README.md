# Beam Agent

<p align="center">
  <a href="https://github.com/fuegocoding/beam-agent"><img src="https://img.shields.io/badge/GitHub-beam--agent-blue?style=for-the-badge" alt="GitHub"></a>
  <a href="https://github.com/fuegocoding/beam-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License: Proprietary"></a>
</p>

**Digital brains for AI agents.** Beam Agent ships with a personality brain runtime, a library of 9 installable historical-figure brains, and a multi-channel messaging gateway. Build a brain from an adaptive interview, install a Pantheon personality from the marketplace, and deploy the resulting agent across 22+ messaging platforms — fully offline after install.

Built on [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — the agent framework with 22+ messaging channels, a mature plugin/skill system, and a battle-tested conversation loop.

---

## What It Does

<table>
<tr><td><b>Brain Runtime</b></td><td>Local-only personality graph search, context building, and SOUL.md generation. Fully offline after install — no LLM calls on the read path. The brain subsystem flattens legacy and v2.2.0 marketplace schemas into a single canonical node list the retriever indexes.</td></tr>
<tr><td><b>Build-Your-Own Brain</b></td><td>The <code>brain_platform/</code> package ports the cloud <code>beam_mind</code> pipeline to a local single-user runtime: 19-question adaptive interview with LLM-powered depth check + follow-up generation, 3-pass extraction (entities → facts → traits → edges), bi-temporal graph persistence in Neo4j + Graphiti, and brain file export to JSON-LD, Claude Projects, or Obsidian vault. 9 file-type parsers (PDF, DOCX, Obsidian, code, email, journal, reddit, prompt, instructions) feed the import pipeline.</td></tr>
<tr><td><b>Adaptive Interview</b></td><td>Two paths: (1) the offline deterministic scripted interview (no LLM needed), and (2) <code>beam interview --adaptive</code> — the LLM-powered path from the cloud, with depth-checked follow-ups and reactive question selection driven by coverage gaps.</td></tr>
<tr><td><b>Marketplace Personalities</b></td><td>9 installable Pantheon brains (Bill Gates, Elon Musk, Marcus Aurelius, Seneca, Terence Tao, Virginia Woolf, Leonardo da Vinci, Benjamin Franklin, Albert Einstein). Downloaded once, queried forever, fully offline.</td></tr>
<tr><td><b>SOUL.md Identity</b></td><td>Auto-generates a SOUL.md from the active brain's graph. The agent loads it as its identity — it knows who the persona is, how they think, and how they communicate. Regenerated on every brain switch.</td></tr>
<tr><td><b>Brain Search</b></td><td>Two-stage retrieval: keyword-scored node search + result-anchored edge search that surfaces relationships between matched concepts. Returns structured nodes, edges, and a human-readable context string. With <code>brain_platform</code> configured, the agent runtime auto-uses the Neo4j-backed retriever and falls back to offline keyword search when Neo4j is unreachable.</td></tr>
<tr><td><b>Multi-Brain Support</b></td><td>Install as many brains as you want at <code>~/.beam/brains/&lt;name>/</code>. Switch between them with <code>beam brain switch &lt;name></code> or the interactive <code>/brain</code> picker. The marketplace ships new brains for free.</td></tr>
<tr><td><b>24/7 Messaging</b></td><td>Deploy across Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Matrix, Teams, and 15+ more platforms via the Hermes gateway.</td></tr>
</table>

---

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for any OpenAI-compatible provider (OpenRouter, OpenAI, Anthropic, Gemini, DeepSeek, …) — required for both the adaptive interview and the build-your-own brain pipeline
- Neo4j 5.x (required for `brain_platform/`, the LLM-powered build path; not needed for marketplace brains). Free hosted option: [Neo4j Aura](https://neo4j.com/cloud/aura/)

### Install

```bash
# Clone the repo
git clone https://github.com/fuegocoding/beam-agent.git
cd beam-agent

# Install Python dependencies
pip install -e ".[all]"

# Start beam (single command — handles setup on first run)
beam
```

The `beam` command checks for API keys and config on first run, walks you through setup if needed, then starts the interactive CLI.

---

## Install a Brain

Beam Agent ships with **9 Pantheon personalities** ready to install. Each brain is a 200–400 KB personality graph backed by extensive domain research — download takes a few seconds and the brain is fully offline forever after.

```bash
# Install any personality by slug
beam install bill-gates
beam install elon-musk
beam install marcus-aurelius
beam install seneca
beam install terence-tao
beam install virginia-woolf
beam install leonardo-da-vinci
beam install benjamin-franklin
beam install albert-einstein
```

The new brain is downloaded to `~/.beam/brains/<slug>/personality_graph.json`, registered in `~/.beam/config.yaml`, and activated. A matching SOUL.md is materialized to `~/.hermes/SOUL.md`.

### Switch Between Brains

```bash
beam brain list                # Show all installed brains (● = active)
beam brain switch elon-musk    # Switch active brain
beam brain info bill-gates     # Show node count, edges, coverage
beam brain update bill-gates   # Re-download latest from marketplace
beam brain remove elon-musk    # Uninstall
```

Or inside the interactive CLI:

```
/brain                         # Interactive picker (↑/↓ + enter)
/brain list                    # List all installed
/brain switch <name>           # Switch active
/brain info [name]             # Show coverage + stats
/brain remove <name>           # Uninstall
```

Switching regenerates `~/.hermes/SOUL.md` and invalidates the agent's cached system prompt so the new identity shows up on the very next turn.

### Available Personalities

| Slug | Persona | Category | Vibe |
|---|---|---|---|
| `bill-gates` | Bill Gates | other | Co-founder of Microsoft, philanthropist — the most influential technologist of his generation |
| `elon-musk` | Elon Musk | other | Engineer and entrepreneur — Tesla, SpaceX, and the relentless pursuit of the future |
| `marcus-aurelius` | Marcus Aurelius | personal | Roman Emperor and Stoic philosopher — daily wisdom from Meditations |
| `seneca` | Seneca | personal | Roman Stoic philosopher and tutor to Nero — urgent wisdom on time, death, and freedom |
| `terence-tao` | Terence Tao | education | Mathematician — Fields Medalist, polymath, and public voice for mathematics |
| `albert-einstein` | Albert Einstein | education | Theoretical physicist and humanitarian — imagination, curiosity, and conscience |
| `benjamin-franklin` | Benjamin Franklin | personal | American founding father, scientist, and printer — practical wisdom and witty maxims |
| `virginia-woolf` | Virginia Woolf | creative | Modernist novelist and feminist pioneer — consciousness, memory, and a room of one's own |
| `leonardo-da-vinci` | Leonardo da Vinci | creative | Renaissance polymath — painter, engineer, anatomist, and student of nature |

All marketplace brains are free. Source: [`api.openbeam.me`](https://api.openbeam.me). Override with `BEAM_API_URL=<your-mirror>` for self-hosted catalogs.

### Build Your Own Brain

Prefer a custom persona? Two paths — pick the one that matches your setup:

#### Offline path (no Neo4j, no LLM for extraction)

```bash
beam interview          # Start the deterministic scripted interview
beam brain status       # Check brain coverage
beam brain export       # Generate SOUL.md + memory files
```

Or inside the interactive CLI:

```
/interview              # Start interview
/brain status           # Check coverage
/brain export           # Export to SOUL.md
```

The interview walks 6 domains for ~15 questions, then a second pass to fill gaps. The offline builder indexes the transcript as memory chunks and derives a short summary. For richer structured fields (procedural patterns, work loops, etc.), edit `~/.beam/brains/default/personality_graph.json` directly — the file is plain JSON, human-readable, git-friendly.

#### LLM-powered path (the cloud-quality build)

This path ports `beam_mind`'s interview + extraction + graph persistence to a local runtime. Requires Neo4j (set up with `beam brain setup-neo4j`) and an LLM API key.

```bash
# One-time: configure Neo4j (works with Neo4j Aura free tier, Docker, or Desktop)
beam brain setup-neo4j

# Run the LLM-powered adaptive interview (19 questions, depth-checked follow-ups,
# reactive question selection driven by coverage gaps)
beam interview --adaptive

# Import content to enrich the brain (PDF, DOCX, Obsidian, code, email, journal, etc.)
beam brain platform-ingest ~/essays/college-application.pdf
beam brain platform-ingest ~/journals/2026-01-15.md
beam brain platform-ingest ~/code/main.py
# --type flag overrides auto-detection
beam brain platform-ingest something.md --type code

# Search the Neo4j-backed graph for relevant facts
beam brain platform-search "what does the user believe about honesty"

# Find thin dimensions and get probe questions to fill them
beam brain platform-deepen

# Generate the canonical brain file (JSON-LD v2.2.0)
beam brain platform-generate ~/my-brain.json

# Export in any of 3 formats
beam brain platform-export ~/claude-prompt.json --format claude
beam brain platform-export ~/brain.jsonld --format jsonld
beam brain platform-export ~/obsidian-vault.zip --format obsidian
```

The two paths share the same `BrainFileSchema` format. Once a brain is built via `brain_platform`, the agent's `GraphBackedBrainRetriever` auto-uses the Neo4j graph and falls back to the offline retriever when Neo4j is unreachable.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        beam-agent (forked Hermes)                              │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐                 │
│  │  CLI (beam)   │  │  Gateway     │  │  22+ Messaging       │                 │
│  │  beam chat    │  │  Daemon      │  │  Telegram/Discord/   │                 │
│  │  beam install │  │  (Python)    │  │  Slack/WhatsApp/...  │                 │
│  │  beam brain   │  │              │  │                      │                 │
│  │    platform-* │  │              │  │                      │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘                 │
│         └─────────────────┼──────────────────────┘                           │
│                           ▼                                                  │
│                    ┌──────────────┐                                           │
│                    │  Agent Loop  │  + SOUL.md + Skills                       │
│                    └──────┬───────┘                                           │
│         ┌─────────────────┼────────────────────────────────┐                  │
│         ▼                 ▼                                ▼                  │
│  ┌─────────────────┐ ┌─────────────────────┐ ┌──────────────────────────┐    │
│  │ GraphBacked     │ │ Memory              │ │ Interview                 │    │
│  │ BrainRetriever  │ │ (MD files)          │ │  Adaptive (LLM)           │    │
│  │ (auto-detect)   │ │ MDMemory            │ │  Scripted (offline)       │    │
│  │                 │ └──────────┬──────────┘ └────────────┬─────────────┘    │
│  │  ┌────────────┐ │            │                         │                  │
│  │  │brain_plat/ │ │            ▼                         ▼                  │
│  │  │LocalSearch │ │  ┌─────────────────────────┐ ┌────────────────────┐   │
│  │  │(Neo4j)     │ │  │ ~/.beam/brains/<name>/   │ │ BrainExtractor     │   │
│  │  └─────┬──────┘ │  │ personality_graph.json   │ │ (3-pass LLM)       │   │
│  │        │        │  │ + soul.md                │ └─────────┬──────────┘   │
│  │        ▼        │  │ (marketplace brains)     │           │              │
│  │  ┌────────────┐ │  └─────────────────────────┘           ▼              │
│  │  │ brain/     │ │                                     ┌────────────┐     │
│  │  │ BrainSearch│ │                                     │LocalGraph- │     │
│  │  │ (offline)  │ │                                     │Writer     │     │
│  │  └────────────┘ │                                     └─────┬──────┘     │
│  └─────────────────┘                                           │            │
│                                                                ▼            │
│                                              ┌──────────────────────────┐  │
│                                              │ Neo4j + Graphiti          │  │
│                                              │ (required for brain_plat) │  │
│                                              └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘

Two brain stacks, one schema:
  • brain/        — offline-first, keyword search on personality_graph.json
  • brain_platform/ — LLM-powered build path, Neo4j + Graphiti for the graph
Both use the same BrainFileSchema (v2.2.0 JSON-LD) and the same SOUL.md
generator. The agent runtime's GraphBackedBrainRetriever auto-detects
which to use based on NEO4J_URI presence.
```

Reference-only: brain-rust/ contains a Rust workspace (beam-interview,
beam-brain-builder, beam-brain-runtime) that mirrors the Python
schemas. The Python `BrainRetriever` is the runtime path; the Rust
binaries are not invoked by the CLI/gateway today.
```

The brain runtime is fully offline after install for marketplace brains. Search, context building, SOUL.md generation, and stats all run locally against the personality graph file. The `brain_platform/` build path requires Neo4j (for graph persistence) and an LLM API key (for the interview + extraction); the agent's read path falls back to offline search when Neo4j is unreachable.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `beam` | Start interactive chat (default — first-time setup if needed) |
| `beam install <slug>` | Install a Pantheon personality from the marketplace |
| `beam brain list` | Show installed brains (active marked) |
| `beam brain switch <name>` | Switch the active brain |
| `beam brain info [name]` | Show node count, edges, coverage |
| `beam brain update [name]` | Re-download from marketplace |
| `beam brain remove <name>` | Uninstall a brain |
| `beam interview` | Start the offline scripted personality interview |
| `beam interview --adaptive` | Start the LLM-powered adaptive interview (`brain_platform`) |
| `beam brain setup-neo4j` | Configure Neo4j connection for `brain_platform` |
| `beam brain platform-ingest <file>` | Ingest a file (PDF/DOCX/md/code/email/journal/...) into the brain |
| `beam brain platform-search <query>` | Search the Neo4j-backed personality graph |
| `beam brain platform-deepen` | Analyze brain gaps and generate probe questions |
| `beam brain platform-generate <out.json>` | Assemble the canonical brain file from Neo4j |
| `beam brain platform-export <out> --format <fmt>` | Export brain file as `claude`, `jsonld`, or `obsidian` |
| `beam chat` | Interactive chat (explicit) |
| `beam gateway` | Run the messaging gateway daemon |
| `beam setup` | Setup wizard (provider + API key) |
| `/brain` (in CLI) | Interactive brain picker (↑/↓ + enter) |
| `/interview` (in CLI) | Start interview from inside chat |
| `/model` (in CLI) | Switch LLM provider/model |
| `/help` (in CLI) | Show all slash commands |

---

## Storage

| Layer | Format | Location |
|-------|--------|----------|
| Brain config | YAML | `~/.beam/config.yaml` (`active_brain` + `brains:` registry) |
| Installed brains | Directory per brain | `~/.beam/brains/<slug>/` |
| Personality graph | JSON (v2.2.0) | `~/.beam/brains/<slug>/personality_graph.json` |
| SOUL.md per brain | Markdown | `~/.beam/brains/<slug>/soul.md` |
| Memory per brain | Markdown | `~/.beam/brains/<slug>/memory/{episodic,semantic,procedural}/*.md` |
| Agent identity | Markdown | `~/.hermes/SOUL.md` (auto-regenerated on brain switch) |
| Sessions | SQLite | `~/.hermes/state.db` |
| Config | YAML | `~/.hermes/config.yaml` |
| Neo4j config | env | `~/.hermes/.env` (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) |
| Neo4j graph (brain_platform) | Bolt | required for `brain_platform`; see Neo4j section below |

Override the brain root with `BEAM_HOME=/path/to/.beam` (e.g. for per-profile installs).

---

## Neo4j (required for `brain_platform`)

The `brain_platform/` build path uses Neo4j + Graphiti for bi-temporal graph persistence. **Neo4j is required** if you want to:

- Run `beam interview --adaptive` and persist the result
- Ingest files via `beam brain platform-ingest`
- Search via `beam brain platform-search` (the agent uses this at runtime)
- Generate / export brain files via `beam brain platform-{generate,export}`

**Marketplace brains don't need it** — they're fully offline JSON files.

Three deployment options (any of them work):

### Option 1: Neo4j Aura (managed cloud, free tier) — recommended

1. Sign up at https://neo4j.com/cloud/aura/
2. Create a free instance
3. Copy the connection URI + credentials from the Aura console
4. Run `beam brain setup-neo4j` and paste the values

No local install. The bolt+s:// protocol (with the `+s`) is required for Aura — it's TLS.

### Option 2: Docker (local)

```bash
docker run -d --name beam-neo4j -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password neo4j:5
```

Then `beam brain setup-neo4j` accepts the defaults (bolt://localhost:7687, user `neo4j`, password `password`).

### Option 3: Neo4j Desktop

Download from https://neo4j.com/download/, create a local DBMS, then point `NEO4J_URI` at the bolt port (default `bolt://localhost:7687`).

The `beam brain setup-neo4j` wizard writes `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` to `~/.hermes/.env`. The agent runtime reads these automatically. If Neo4j goes down mid-session, the `GraphBackedBrainRetriever` falls back to the offline keyword retriever transparently.

---

## Development

### Run tests

```bash
# Offline brain runtime
scripts/run_tests.sh tests/beam/ -v

# brain_platform (LLM-powered build path, ported from beam_mind)
scripts/run_tests.sh tests/brain_platform/ -v

# All ported code (212 tests, zero regressions)
scripts/run_tests.sh tests/beam/ tests/brain_platform/ -v
```

The wrapper enforces hermetic environment parity with CI (TZ=UTC, LANG=C.UTF-8, subprocess-per-test isolation, credential-env blanking). See [AGENTS.md](AGENTS.md) for details.

### Rust workspace (reference only)

```bash
cd brain-rust
cargo build --release
```

The Rust crates (`beam-interview`, `beam-brain-builder`, `beam-brain-runtime`) mirror the Python schemas in `brain-rust/src/lib.rs` and provide a JSON-stdin/JSON-stdout subprocess interface. The Python `BrainRetriever` is the runtime path the CLI and gateway invoke today; the Rust binaries are reference implementations and are not currently wired in.

### Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fork base | NousResearch/hermes-agent | 180k stars, 22+ channels, mature plugin system |
| Brain runtime | Python (`BrainRetriever` + `schema_adapter`) | Hot path is offline-only — keyword search on <500-node graphs is microseconds, no FFI complexity |
| Brain schema | v2.2.0 JSON-LD (`BrainFileSchema`) | Portable, versioned, supports both legacy flat and marketplace shapes via the schema adapter |
| Marketplace transport | One-shot download | Brain is read offline from then on — no proxy, no network on the read path |
| Memory | MD files | Human-readable, git-friendly, user-editable |
| Interview | Multi-pass scripted (offline) + LLM-powered adaptive (brain_platform) | Two paths: deterministic default (no LLM) for users without API keys; cloud-quality LLM path for users who want it |
| Build-your-own brain stack | `brain_platform/` (port of `beam_mind`) | Free single-user local port of the cloud's interview + extraction + Neo4j + Graphiti pipeline. No DB, no Celery, no S3 — the LLM calls go through beam-agent's existing `call_llm` BYOK infrastructure |
| Graph persistence | Neo4j + Graphiti (required for brain_platform) | Same schema, same search, same bi-temporal edge model as the cloud. Works with Neo4j Aura (free tier, no Docker), local Docker, or Desktop |
| Graph fallback | `GraphBackedBrainRetriever` auto-falls-back | When Neo4j is down or unconfigured, the agent uses the offline keyword retriever. No user-visible breakage |

---

## License

Proprietary — All rights reserved. See [LICENSE](LICENSE) for terms.

Built on [hermes-agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com).
The original Hermes framework is licensed under the MIT License (see [NOTICE](NOTICE)).
