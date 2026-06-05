"""beam build - Brain builder mode.

A separate CLI interface for building, testing, and publishing brains.
Activated via 'beam build' from the shell.

Commands:
    new <name>          Create a new brain project
    list                List all brain projects
    open <name>         Open a project (set as working project)
    interview [name]    Run interview for project
    ingest <source>     Ingest data (file, URL, text) into open project
    test [name]         Test brain (coverage, Q&A, consistency)
    export [name]       Export brain as .beam file
    publish [name]      Publish to marketplace
    status [name]       Show project stats
    delete <name>       Delete a project
    help                Show commands
    quit                Exit build mode
"""
import json
import shutil
import sys
from pathlib import Path


class BuildCLI:
    """Interactive REPL for brain building."""

    def __init__(self):
        self.current_project: str | None = None
        self.running = True

    def run(self):
        """Main REPL loop."""
        from brain.paths import ensure_beam_dirs, get_projects_dir

        ensure_beam_dirs()
        self._print_banner()

        while self.running:
            try:
                prompt = self._get_prompt()
                line = input(prompt).strip()
                if not line:
                    continue
                self._dispatch(line)
            except KeyboardInterrupt:
                print("\n")
                continue
            except EOFError:
                print("\nExiting build mode.")
                break

    def _print_banner(self):
        print("\n  ╔══════════════════════════════════════╗")
        print("  ║        Beam Brain Builder             ║")
        print("  ║   Build, test, and publish brains     ║")
        print("  ╚══════════════════════════════════════╝")
        print("\n  Type 'help' for commands, 'quit' to exit.\n")

    def _get_prompt(self) -> str:
        if self.current_project:
            return f"beam-build [{self.current_project}]> "
        return "beam-build> "

    def _dispatch(self, line: str):
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        commands = {
            "help": self._cmd_help,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
            "new": self._cmd_new,
            "list": self._cmd_list,
            "ls": self._cmd_list,
            "open": self._cmd_open,
            "interview": self._cmd_interview,
            "ingest": self._cmd_ingest,
            "test": self._cmd_test,
            "export": self._cmd_export,
            "publish": self._cmd_publish,
            "status": self._cmd_status,
            "delete": self._cmd_delete,
            "rm": self._cmd_delete,
        }

        handler = commands.get(cmd)
        if handler:
            try:
                handler(args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands.")

    def _cmd_help(self, args):
        print("""
Commands:
  new <name>          Create a new brain project
  list                List all brain projects
  open <name>         Open a project as working project
  interview [name]    Run interview for project
  ingest <source>     Ingest data into open project
                      Sources: <file-path>, <url>, "text..."
  test [name]         Test brain (coverage, Q&A, consistency)
  export [name]       Export brain as .beam file
  publish [name]      Publish to marketplace (requires auth)
  status [name]       Show project stats
  delete <name>       Delete a project
  help                Show this help
  quit                Exit build mode
""")

    def _cmd_quit(self, args):
        self.running = False
        print("Exiting build mode.")

    def _cmd_new(self, args):
        if not args:
            print("Usage: new <name>")
            return

        name = args[0]
        from brain.paths import get_project_path, get_project_graph_path, get_project_transcript_path

        project_path = get_project_path(name)
        if project_path.exists():
            print(f"Project '{name}' already exists.")
            return

        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "sources").mkdir(exist_ok=True)

        # Create empty personality graph
        graph_path = get_project_graph_path(name)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_summary": "",
                "traits": [], "beliefs": [], "values": [], "boundaries": [],
                "life_events": [], "memories": [], "patterns": [], "social": [],
                "expertise": [], "style": [], "people": [], "places": [],
                "procedural_patterns": [], "work_loops": [], "prompting_styles": [],
                "technical_gaps": [], "edges": [],
                "voice_dna": {}, "work_dna": {},
                "behavioral_rules": [], "contradiction_patterns": [],
                "emotional_triggers": [], "emotional_profile": {}, "contextual_moods": [],
            }, f, indent=2)

        # Create empty interview transcript
        transcript_path = get_project_transcript_path(name)
        with open(transcript_path, "w") as f:
            json.dump([], f)

        print(f"Project '{name}' created at {project_path}")
        self.current_project = name

    def _cmd_list(self, args):
        from brain.paths import get_projects_dir, get_project_graph_path

        projects_dir = get_projects_dir()
        if not projects_dir.exists():
            print("No projects found.")
            return

        projects = sorted([d.name for d in projects_dir.iterdir() if d.is_dir()])
        if not projects:
            print("No projects found.")
            return

        print("Brain projects:\n")
        for name in projects:
            marker = "●" if name == self.current_project else "○"
            graph_path = get_project_graph_path(name)
            size_info = ""
            if graph_path.exists():
                try:
                    with open(graph_path, encoding="utf-8") as f:
                        graph = json.load(f)
                    node_count = sum(len(graph.get(k, [])) for k in [
                        "traits", "beliefs", "values", "memories", "patterns",
                        "procedural_patterns", "work_loops", "expertise"
                    ])
                    size_info = f" ({node_count} nodes)"
                except Exception:
                    size_info = " (parse error)"
            print(f"  {marker} {name}{size_info}")

    def _cmd_open(self, args):
        if not args:
            print(f"Current project: {self.current_project or 'none'}")
            return

        name = args[0]
        from brain.paths import get_project_path

        project_path = get_project_path(name)
        if not project_path.exists():
            print(f"Project '{name}' not found.")
            return

        self.current_project = name
        print(f"Opened project '{name}'.")

    def _cmd_interview(self, args):
        name = args[0] if args else self.current_project
        if not name:
            print("No project open. Use 'open <name>' or 'interview <name>'.")
            return

        from brain.paths import get_project_path, get_project_graph_path, get_project_transcript_path

        project_path = get_project_path(name)
        if not project_path.exists():
            print(f"Project '{name}' not found.")
            return

        print(f"Starting interview for project '{name}'...")
        print("(Interview system reuses the existing interview orchestrator)")
        print("This will guide you through building the brain's personality graph.\n")

        # Import and run the interview orchestrator
        try:
            from brain.interview_orchestrator import InterviewOrchestrator

            transcript_path = get_project_transcript_path(name)
            graph_path = get_project_graph_path(name)

            # Load existing transcript if any
            existing_transcript = []
            if transcript_path.exists():
                with open(transcript_path) as f:
                    existing_transcript = json.load(f)

            orchestrator = InterviewOrchestrator()
            if existing_transcript:
                print(f"Resuming interview ({len(existing_transcript)} existing answers)...\n")

            # Run interview interactively
            question = orchestrator.start()
            print(f"Q: {question}\n")

            while not orchestrator.is_complete():
                try:
                    answer = input("A: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterview interrupted. Progress saved.")
                    break

                if not answer:
                    continue

                orchestrator.add_answer(answer)
                if orchestrator.is_complete():
                    print("\nInterview complete!")
                    break

                question = orchestrator.next_question()
                print(f"\nQ: {question}\n")

            # Save transcript
            transcript = orchestrator.get_transcript()
            with open(transcript_path, "w") as f:
                json.dump(transcript, f, indent=2)
            print(f"Transcript saved to {transcript_path}")

            # Build brain from transcript
            if orchestrator.is_complete():
                print("Building personality graph from interview...")
                from brain.brain_builder import BrainBuilder
                builder = BrainBuilder()
                graph = builder.extract(transcript)
                with open(graph_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2, default=str)
                print(f"Brain saved to {graph_path}")

        except ImportError as e:
            print(f"Error: Interview system not available: {e}")
        except Exception as e:
            print(f"Error during interview: {e}")

    def _cmd_ingest(self, args):
        name = self.current_project
        if not name:
            print("No project open. Use 'open <name>' first.")
            return

        if not args:
            print("Usage: ingest <file-path|url|text>")
            return

        source = " ".join(args)
        from brain.paths import get_project_path, get_project_graph_path

        project_path = get_project_path(name)
        graph_path = get_project_graph_path(name)

        if not graph_path.exists():
            print(f"Project '{name}' has no brain data. Run 'interview' first.")
            return

        print(f"Ingesting data from: {source}")

        # Determine source type
        content = None
        if source.startswith("http://") or source.startswith("https://"):
            print("(URL ingestion - fetching content...)")
            try:
                import httpx
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(source)
                    resp.raise_for_status()
                    content = resp.text
                print(f"Fetched {len(content)} characters from URL.")
            except Exception as e:
                print(f"Error fetching URL: {e}")
                return
        elif Path(source).exists():
            print(f"(File ingestion - reading {source})")
            try:
                content = Path(source).read_text(encoding="utf-8")
                print(f"Read {len(content)} characters from file.")
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        else:
            # Treat as raw text
            content = source
            print(f"Processing {len(content)} characters of text.")

        if not content:
            print("No content to ingest.")
            return

        # Save source to sources directory
        sources_dir = project_path / "sources"
        sources_dir.mkdir(exist_ok=True)
        source_file = sources_dir / f"source_{len(list(sources_dir.iterdir())) + 1}.txt"
        source_file.write_text(content, encoding="utf-8")
        print(f"Source saved to {source_file}")

        # Ingest into brain (merge with existing)
        print("Processing with brain builder...")
        try:
            from brain.brain_builder import BrainBuilder

            with open(graph_path, encoding="utf-8") as f:
                existing_graph = json.load(f)

            builder = BrainBuilder()
            # Use the builder to extract from the new content
            # This creates a temporary transcript-like structure
            temp_transcript = [{"q": "source", "a": content}]
            new_graph = builder.extract(temp_transcript)

            # Merge new data into existing graph
            for key in ["traits", "beliefs", "values", "memories", "patterns",
                        "expertise", "style", "people", "places"]:
                existing = existing_graph.get(key, [])
                new_items = new_graph.get(key, [])
                # Simple merge: add new items
                existing.extend(new_items)
                existing_graph[key] = existing

            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(existing_graph, f, indent=2, default=str)
            print("Data ingested successfully.")

        except Exception as e:
            print(f"Error during ingestion: {e}")

    def _cmd_test(self, args):
        name = args[0] if args else self.current_project
        if not name:
            print("No project open. Use 'open <name>' or 'test <name>'.")
            return

        from brain.paths import get_project_graph_path

        graph_path = get_project_graph_path(name)
        if not graph_path.exists():
            print(f"Project '{name}' has no brain data.")
            return

        print(f"Testing brain '{name}'...\n")

        try:
            with open(graph_path, encoding="utf-8") as f:
                graph = json.load(f)

            # Coverage check
            print("=== Coverage ===")
            node_types = {
                "traits": "Personality traits",
                "beliefs": "Core beliefs",
                "values": "Values",
                "memories": "Memories",
                "patterns": "Behavioral patterns",
                "expertise": "Expertise areas",
                "procedural_patterns": "Procedural patterns",
                "work_loops": "Work loops",
            }
            total_nodes = 0
            for key, label in node_types.items():
                count = len(graph.get(key, []))
                total_nodes += count
                status = "✓" if count > 0 else "✗"
                print(f"  {status} {label}: {count}")

            kg_nodes = len(graph.get("knowledge_graph", {}).get("nodes", []))
            total_nodes += kg_nodes
            print(f"  {'✓' if kg_nodes > 0 else '✗'} Knowledge graph nodes: {kg_nodes}")
            print(f"\n  Total nodes: {total_nodes}")

            # Edge count
            edges = len(graph.get("edges", []))
            kg_edges = len(graph.get("knowledge_graph", {}).get("edges", []))
            print(f"  Total edges: {edges + kg_edges}")

            # Voice DNA check
            print("\n=== Voice DNA ===")
            voice = graph.get("voice_dna", {})
            if voice:
                for key in ["characteristic_phrases", "humor_style", "response_length_pattern"]:
                    has = bool(voice.get(key))
                    print(f"  {'✓' if has else '✗'} {key}")
            else:
                print("  ✗ No Voice DNA defined")

            # Sensitivity audit
            print("\n=== Sensitivity Audit ===")
            private_count = 0
            for key in node_types:
                for node in graph.get(key, []):
                    if isinstance(node, dict) and node.get("sensitivity") == "private":
                        private_count += 1
            print(f"  Private nodes: {private_count}")
            if private_count > 0:
                print("  ⚠ Review private nodes before publishing")

            # Summary
            print("\n=== Summary ===")
            summary = graph.get("user_summary", "")
            if summary:
                print(f"  {summary[:200]}...")
            else:
                print("  ✗ No user summary defined")

            print(f"\nTest complete for '{name}'.")

        except Exception as e:
            print(f"Error testing brain: {e}")

    def _cmd_export(self, args):
        name = args[0] if args else self.current_project
        if not name:
            print("No project open. Use 'open <name>' or 'export <name>'.")
            return

        from brain.paths import get_project_graph_path

        graph_path = get_project_graph_path(name)
        if not graph_path.exists():
            print(f"Project '{name}' has no brain data.")
            return

        # Export as .beam file
        output_path = Path.cwd() / f"{name}.beam"
        shutil.copy2(graph_path, output_path)
        print(f"Brain exported to {output_path}")

    def _cmd_publish(self, args):
        name = args[0] if args else self.current_project
        if not name:
            print("No project open. Use 'open <name>' or 'publish <name>'.")
            return

        from brain.paths import get_project_graph_path

        graph_path = get_project_graph_path(name)
        if not graph_path.exists():
            print(f"Project '{name}' has no brain data.")
            return

        print(f"Publishing brain '{name}' to marketplace...")
        print("(This requires authentication with beam_mind)\n")

        # Load brain data
        with open(graph_path, encoding="utf-8") as f:
            brain_json = json.load(f)

        # Get metadata from user
        tagline = input("Tagline (max 280 chars): ").strip()
        description = input("Description (optional): ").strip()
        category = input("Category [personal/professional/creative/technical/education/entertainment/other]: ").strip() or "other"
        tags_input = input("Tags (comma-separated): ").strip()
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

        pricing = input("Pricing [free/one_time/subscription/pay_per_use]: ").strip() or "free"
        price_cents = None
        subscription_price_cents = None
        if pricing == "one_time":
            price_str = input("Price in dollars: ").strip()
            price_cents = int(float(price_str) * 100) if price_str else None
        elif pricing == "subscription":
            price_str = input("Monthly price in dollars: ").strip()
            subscription_price_cents = int(float(price_str) * 100) if price_str else None
        elif pricing == "pay_per_use":
            price_str = input("Per-use price in dollars: ").strip()
            price_cents = int(float(price_str) * 100) if price_str else None

        # Confirm
        print(f"\nPublishing:")
        print(f"  Name: {name}")
        print(f"  Tagline: {tagline}")
        print(f"  Category: {category}")
        print(f"  Tags: {', '.join(tags)}")
        print(f"  Pricing: {pricing}")
        confirm = input("\nProceed? [y/N] ").strip()
        if confirm.lower() != "y":
            print("Cancelled.")
            return

        # Publish via API
        import os
        import httpx

        api_url = os.environ.get("BEAM_API_URL", "https://api.openbeam.me")
        token = os.environ.get("BEAM_AUTH_TOKEN")

        if not token:
            print("Error: BEAM_AUTH_TOKEN environment variable required for publishing.")
            print("Set it with: export BEAM_AUTH_TOKEN=<your-jwt-token>")
            return

        payload = {
            "brain_name": name,
            "tagline": tagline or None,
            "description": description or None,
            "category": category,
            "tags": tags or None,
            "pricing_model": pricing,
            "price_cents": price_cents,
            "subscription_price_cents": subscription_price_cents,
            "brain_json": brain_json,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{api_url}/api/v1/marketplace/publish",
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                result = resp.json()
                print(f"\nPublished successfully!")
                print(f"  Slug: {result.get('slug')}")
                print(f"  URL: {api_url}/marketplace/{result.get('slug')}")
        except httpx.HTTPStatusError as e:
            print(f"\nError: API returned {e.response.status_code}")
            try:
                detail = e.response.json().get("detail", "")
                if detail:
                    print(f"  {detail}")
            except Exception:
                pass
        except Exception as e:
            print(f"\nError: {e}")

    def _cmd_status(self, args):
        name = args[0] if args else self.current_project
        if not name:
            print("No project open. Use 'open <name>' or 'status <name>'.")
            return

        self._cmd_test(args)  # Reuse test for now

    def _cmd_delete(self, args):
        if not args:
            print("Usage: delete <name>")
            return

        name = args[0]
        from brain.paths import get_project_path

        project_path = get_project_path(name)
        if not project_path.exists():
            print(f"Project '{name}' not found.")
            return

        confirm = input(f"Delete project '{name}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

        shutil.rmtree(project_path)
        if self.current_project == name:
            self.current_project = None
        print(f"Project '{name}' deleted.")


def cmd_build(args):
    """Handle 'beam build' command."""
    cli = BuildCLI()
    cli.run()


def register_build_command(subparsers):
    """Register the 'build' subcommand with argparse."""
    parser = subparsers.add_parser(
        "build",
        help="Enter brain builder mode",
        description="Interactive brain builder for creating, testing, and publishing brains.",
    )
    parser.set_defaults(func=cmd_build)
