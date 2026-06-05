# Beam Agent

<p align="center">
  <a href="https://github.com/fuegocoding/beam-agent"><img src="https://img.shields.io/badge/GitHub-beam--agent-blue?style=for-the-badge" alt="GitHub"></a>
  <a href="https://github.com/fuegocoding/beam-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

**Your personal AI digital clone.** Beam builds a deep understanding of who you are through an adaptive interview, stores it as a personality graph, and deploys a 24/7 agent that thinks like you across all messaging platforms.

Built on [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — the 180k-star agent framework with 22+ messaging channels, a mature plugin/skill system, and a battle-tested conversation loop.

---

## What It Does

<table>
<tr><td><b>Adaptive Interview</b></td><td>Multi-pass personality interview (30+ questions across 6 domains). Follow-up detection for vagueness, contradiction, emotion, and depth. 3 passes: surface → deep → gaps.</td></tr>
<tr><td><b>Personality Graph</b></td><td>Your identity stored as a knowledge graph — traits, beliefs, values, boundaries, life events, people, places, and 22 edge types. Built in Rust for performance.</td></tr>
<tr><td><b>SOUL.md Generation</b></td><td>Auto-generates a SOUL.md from your personality graph. The agent loads it as its identity — it knows who you are, how you think, and how you communicate.</td></tr>
<tr><td><b>Brain Search</b></td><td>Ask your agent about your own personality — "What do I believe about AI safety?" — and it searches your brain graph for relevant context.</td></tr>
<tr><td><b>Memory System</b></td><td>Episodic, semantic, procedural, and style memories stored as human-readable Markdown files. Git-friendly, user-editable.</td></tr>
<tr><td><b>24/7 Messaging</b></td><td>Deploy across Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Matrix, Teams, and 15+ more platforms via the Hermes gateway.</td></tr>
</table>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    beam-agent (forked Hermes)                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  CLI (beam)   │  │  Gateway     │  │  22+ Messaging       │   │
│  │  beam chat    │  │  Daemon      │  │  Telegram/Discord/   │   │
│  │  beam setup   │  │  (Python)    │  │  Slack/WhatsApp/...  │   │
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
│  │ (Python→Rust)│  │ (MD files)   │  │ (Python→Rust)        │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘   │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Rust        │  │  ~/.beam/    │  │  Rust Binaries       │   │
│  │  brain-      │  │  memory/     │  │  beam-interview      │   │
│  │  runtime     │  │  *.md        │  │  beam-brain-builder  │   │
│  └──────────────┘  └──────────────┘  │  beam-brain-runtime  │   │
│                                      └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Rust 1.96+ (for brain binaries)
- Neo4j 5.x (optional, for identity graph storage)

### Install

```bash
# Clone the repo
git clone https://github.com/fuegocoding/beam-agent.git
cd beam-agent

# Install Python dependencies
pip install -e ".[all]"

# Build Rust brain binaries (optional — Python fallback available)
cd brain-rust
cargo build --release
cd ..

# Start beam (single command — handles setup automatically)
beam
```

The `beam` command checks for API keys and config on first run, walks you through setup if needed, then starts the interactive CLI.

### Build Your Brain

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

---

## Project Structure

```
beam-agent/
├── brain/                          # Python brain module
│   ├── brain_schema.py             # Pydantic v2 PersonalityGraph
│   ├── subprocess_bridge.py        # Rust binary caller (JSON stdin/stdout)
│   ├── interview_orchestrator.py   # Multi-pass interview state machine
│   ├── brain_builder.py            # Extract personality from interview
│   ├── brain_retriever.py          # Search/context/export/stats
│   ├── md_memory.py                # MD file read/write (~/.beam/memory/)
│   └── soul_generator.py           # SOUL.md from personality graph
│
├── brain-rust/                     # Rust workspace
│   ├── src/lib.rs                  # PersonalityGraph schema (750 lines)
│   ├── beam-interview/             # Adaptive interview engine
│   ├── beam-brain-builder/         # 14 extractors + edge builder
│   └── beam-brain-runtime/         # Search, context, export engine
│
├── plugins/
│   ├── brain-tools/                # brain_search, brain_export, brain_status
│   ├── interview/                  # start_interview, continue_interview
│   └── memory/beam-memory/         # MD file memory provider
│
├── skills/
│   ├── interview/                  # "build-my-brain" skill
│   ├── brain-search/               # Brain search skill
│   └── brain-export/               # Brain export skill
│
├── beam                            # CLI entry point
├── cli.py                          # CLI (renamed from hermes)
├── hermes_cli/                     # CLI internals (Hermes-derived)
├── gateway/                        # Messaging gateway (22+ platforms)
├── agent/                          # Agent loop, memory, providers
├── tools/                          # Tool registry + implementations
├── docker-compose.beam.yml         # Neo4j 5.26
└── tests/beam/                     # E2E tests
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `beam` | Start interactive chat (default) |
| `beam chat` | Interactive chat |
| `beam gateway` | Run messaging gateway |
| `beam setup` | Setup wizard |
| `/brain status` | Show brain statistics and coverage |
| `/brain export` | Export brain to SOUL.md + memory files |
| `/interview` | Start adaptive brain-building interview |
| `/model` | Switch LLM provider/model |
| `/skills` | Browse available skills |
| `/help` | Show all commands |

---

## Storage

| Layer | Technology | Location |
|-------|-----------|----------|
| Personality Graph | Rust + JSON | `~/.beam/brain/default/personality_graph.json` |
| Episodic Memory | MD files | `~/.beam/memory/default/episodic/*.md` |
| Semantic Memory | MD files | `~/.beam/memory/default/semantic/*.md` |
| Procedural Memory | MD files | `~/.beam/memory/default/procedural/*.md` |
| Style Profile | MD file | `~/.beam/memory/default/style.md` |
| Agent Identity | SOUL.md | `~/.hermes/SOUL.md` (auto-generated) |
| Sessions | SQLite | `~/.hermes/state.db` |
| Config | YAML | `~/.hermes/config.yaml` |

---

## Docker (Neo4j)

```bash
docker compose -f docker-compose.beam.yml up -d
```

Starts Neo4j 5.26 on `bolt://localhost:7687` with default credentials (`neo4j/password`).

---

## Development

### Build Rust binaries

```bash
cd brain-rust
cargo build --release
```

### Run tests

```bash
python -m pytest tests/beam/ -v
```

### Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fork base | NousResearch/hermes-agent | 180k stars, 22+ channels, mature plugin system |
| New components | Rust | Performance, safety, clean JSON serde |
| Rust↔Python bridge | Subprocess JSON | Simplest, no FFI complexity |
| Memory | MD files | Human-readable, git-friendly, user-editable |
| Interview | Multi-pass adaptive | Surface → Deep → Gaps with follow-up detection |

---

## License

MIT — see [LICENSE](LICENSE).

Built on [hermes-agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com).
