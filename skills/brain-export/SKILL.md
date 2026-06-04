---
name: brain-export
description: Export your digital brain as human-readable Markdown files for review and editing.
version: 1.0.0
author: beam-agent
license: MIT
metadata:
  beam:
    tags: [brain, export, review]
    category: core
---

# Brain Export

## When to Use

Use this skill when the user wants to review, export, or inspect their brain data.

## Procedure

1. Call `brain_status` tool to get current brain statistics
2. Call `brain_export` tool to generate human-readable files
3. Show the user where the files were saved
4. Offer to walk them through the contents

## Exported Files

- `~/.beam/SOUL.md` — complete personality profile
- `~/.beam/memory/<user>/brain-export/brain-summary.md` — human-readable summary
- `~/.beam/memory/<user>/style.md` — communication and work style

## Editing

Users can edit exported files directly. Changes are preserved across sessions but won't be reflected in the graph until the next interview run.
