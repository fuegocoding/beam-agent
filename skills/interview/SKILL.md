---
name: build-my-brain
description: Run an adaptive multi-pass interview to build your digital brain. Asks deep questions about your personality, beliefs, values, work style, and emotional patterns.
version: 1.0.0
author: beam-agent
license: MIT
metadata:
  beam:
    tags: [interview, personality, brain, identity]
    category: core
---

# Build My Brain

## When to Use

Use this skill when the user wants to build or update their digital brain/personality profile. This runs an adaptive interview that builds a PersonalityGraph.

## Procedure

1. Call `start_interview` tool to begin
2. Present each question to the user naturally
3. Record their answer with `continue_interview` tool
4. After each answer, the tool may return follow-up questions — ask those too
5. The interview has 3 passes:
   - Pass 1: Surface — broad questions across all domains
   - Pass 2: Deep — targeted follow-ups on interesting signals
   - Pass 3: Gaps — fill remaining holes
6. When the interview completes, the tool returns a PersonalityGraph
7. Summarize what was learned

## Domains Covered

- **Identity**: core traits, values, beliefs
- **Relationships**: key people, social patterns
- **Work**: procedural patterns, work loops, delegation style
- **Emotional**: triggers, mood patterns, recovery
- **Beliefs**: opinions, contradictions, convictions
- **Procedural**: how they think, debug, decide

## Tips

- Be conversational, not robotic
- If the user gives a short answer, encourage them to elaborate
- If they mention something interesting, probe deeper before moving on
- Don't rush — quality of answers matters more than speed
