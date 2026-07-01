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
<tr><td><b>Adaptive Interview</b></td><td>Multi-pass personality interview (30+ questions across 6 domains: identity, relationships, work, emotional, beliefs, procedural). Deterministic scripted progression — no LLM dependency for question selection.</td></tr>
<tr><td><b>Marketplace Personalities</b></td><td>9 installable Pantheon brains (Bill Gates, Elon Musk, Marcus Aurelius, Seneca, Terence Tao, Virginia Woolf, Leonardo da Vinci, Benjamin Franklin, Albert Einstein). Downloaded once, queried forever, fully offline.</td></tr>
<tr><td><b>SOUL.md Identity</b></td><td>Auto-generates a SOUL.md from the active brain's graph. The agent loads it as its identity — it knows who the persona is, how they think, and how they communicate. Regenerated on every brain switch.</td></tr>
<tr><td><b>Brain Search</b></td><td>Two-stage retrieval: keyword-scored node search + result-anchored edge search that surfaces relationships between matched concepts. Returns structured nodes, edges, and a human-readable context string.</td></tr>
<tr><td><b>Multi-Brain Support</b></td><td>Install as many brains as you want at <code>~/.beam/brains/&lt;name&gt;/</code>. Switch between them with <code>beam brain switch &lt;name&gt;</code> or the interactive <code>/brain</code> picker. The marketplace ships new brains for free.</td></tr>
<tr><td><b>24/7 Messaging</b></td><td>Deploy across Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Matrix, Teams, and 15+ more platforms via the Hermes gateway.</td></tr>
</table>

---

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for any OpenAI-compatible provider (OpenRouter, OpenAI, Anthropic, Gemini, DeepSeek, …)
- Neo4j 5.x (optional, for legacy identity-graph storage — not used by marketplace brains)

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

Prefer a custom persona? Run the adaptive interview and the offline brain builder does the rest:

```bash
beam interview          # Start the adaptive interview (interactive)
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

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     beam-agent (forked Hermes)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  CLI (beam)   │  │  Gateway     │  │  22+ Messaging       │   │
│  │  beam chat    │  │  Daemon      │  │  Telegram/Discord/   │   │
│  │  beam install │  │  (Python)    │  │  Slack/WhatsApp/...  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         └─────────────────┼──────────────────────┘               │
│                           ▼                                      │
│                    ┌──────────────┐                               │
│                    │  Agent Loop  │  + SOUL.md + Skills           │
│                    └──────┬───────┘                               │
│         ┌─────────────────┼─────────────────┐                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Brain Search │  │ Memory       │  │ Interview            │   │
│  │ (Python)     │  │ (MD files)   │  │ (Python)             │   │
│  │ BrainRetriever│ │ MDMemory     │  │ InterviewOrchestrator│   │
│  │ +schema_adapter│ └──────┬───────┘  └──────────┬───────────┘   │
│  └──────┬───────┘         │                       │               │
│         ▼                 ▼                       ▼               │
│  ┌────────────────────────────┐  ┌────────────────────────────┐  │
│  │  ~/.beam/brains/<name>/     │  │  ~/.beam/brains/default/    │  │
│  │  personality_graph.json     │  │  personality_graph.json    │  │
│  │  + soul.md                  │  │  (interview-built)          │  │
│  │  (marketplace brains)       │  │                             │  │
│  └────────────────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Reference-only: brain-rust/ contains a Rust workspace (beam-interview,
beam-brain-builder, beam-brain-runtime) that mirrors the Python
schemas. The Python BrainRetriever is the runtime path; the Rust
binaries are not invoked by the CLI/gateway today.
```

The brain runtime is fully offline after install. Search, context building, SOUL.md generation, and stats all run locally against the personality graph file. No network calls on the read path.

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
| `beam interview` | Start the adaptive personality interview |
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

Override the brain root with `BEAM_HOME=/path/to/.beam` (e.g. for per-profile installs).

---

## Docker (Neo4j, optional)

The legacy identity-graph storage used Neo4j 5.26. Marketplace brains don't need it. If you're building a brain from interview and want graph storage:

```bash
docker compose -f docker-compose.beam.yml up -d
```

Starts Neo4j on `bolt://localhost:7687` with default credentials (`neo4j/password`).

---

## Development

### Run tests

```bash
scripts/run_tests.sh tests/beam/ -v
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
| Interview | Multi-pass scripted | Deterministic, no LLM dependency for question selection |

---

## License

Proprietary — All rights reserved. See [LICENSE](LICENSE) for terms.

Built on [hermes-agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com).
The original Hermes framework is licensed under the MIT License (see [NOTICE](NOTICE)).
