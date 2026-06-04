---
name: brain-search
description: Search your digital brain for personality traits, beliefs, memories, and patterns relevant to a query.
version: 1.0.0
author: beam-agent
license: MIT
metadata:
  beam:
    tags: [brain, search, personality, memory]
    category: core
---

# Brain Search

## When to Use

Use this skill when you need to recall information about the user's personality, beliefs, values, work style, or memories. Also use when answering questions that require personalization.

## Procedure

1. Call `brain_search` tool with a relevant query
2. The tool returns matching nodes from the personality graph
3. Use the results to personalize your response
4. If no results, try a broader query

## Query Examples

- "beliefs about AI" — finds belief nodes related to AI
- "work style" — finds procedural patterns and work DNA
- "key relationships" — finds person nodes and social patterns
- "emotional triggers" — finds emotional trigger nodes
- "communication style" — finds voice DNA and style nodes

## Trust Levels

- `visitor`: public nodes only
- `known`: public + personal nodes
- `owner`: all nodes including private

## Brain Power

- `light`: top 3 matches
- `standard`: top 10 matches
- `full`: all matching nodes
