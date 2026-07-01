"""Extraction prompt for user-authored AI instructions files.

Extracts categorized rules verbatim into a DirectiveSet. No BrainExtractor
pass — instructions are an operational config layer, not identity data.
"""

DIRECTIVE_EXTRACTION_PROMPT = """\
You are analyzing a user's custom AI instructions file to extract structured,
actionable directives. These directives will be injected VERBATIM into an AI
agent's system prompt, so preserve the user's exact phrasing.

Categorize every directive using this decision tree:

1. communication_rules — TONE, FORMAT, and STYLE of responses.
   Ask: "Does this control HOW the response looks or sounds?"
   Examples:
   - "Be concise" → communication
   - "Respond in French" → communication
   - "Use markdown formatting" → communication
   - "Keep responses under 3 paragraphs" → communication
   - "Don't use emojis" → communication
   - "Be direct, not flowery" → communication
   - "Match my tone" → communication
   - "Use code blocks for code" → communication

2. workflow_rules — PROCESS and STRUCTURE the agent follows to produce output.
   Ask: "Does this describe a step-by-step procedure or output structure?"
   Examples:
   - "Think step by step before answering" → workflow
   - "Always show your reasoning" → workflow
   - "Start with a summary, then details" → workflow
   - "Check for errors before responding" → workflow
   - "When writing code, add tests" → workflow
   - "Break complex problems into steps" → workflow
   - "Read the file before editing" → workflow
   - "Use the simplest approach first" → workflow

3. domain_constraints — SCOPE of what the agent should know or discuss.
   Ask: "Does this limit or define WHAT topics/tools/technologies to use?"
   Examples:
   - "Only use TypeScript, never JavaScript" → domain
   - "Focus on Python and Go" → domain
   - "Use Next.js App Router, not Pages" → domain
   - "Prefer Tailwind over custom CSS" → domain
   - "Don't suggest jQuery" → domain
   - "Only discuss topics you're knowledgeable about" → domain
   - "Use pnpm, not npm" → domain
   - "Target Python 3.12+" → domain

4. boundaries — HARD LIMITS, refusals, and things to NEVER do.
   Ask: "Is this a strict prohibition or safety constraint?"
   Examples:
   - "Never auto-commit" → boundary
   - "Don't push to main without asking" → boundary
   - "Never delete files without confirmation" → boundary
   - "Don't make up information" → boundary
   - "Never share personal data" → boundary
   - "Don't execute destructive commands" → boundary
   - "Refuse to generate harmful content" → boundary

5. behavioral_rules — ONLY for trigger-response patterns and priorities
   that don't fit the above categories.
   Ask: "Is this a conditional behavior (when X happens, do Y) that isn't
   about format, process, scope, or prohibition?"
   Examples:
   - "If unsure, ask for clarification" → behavioral
   - "Prioritize correctness over speed" → behavioral
   - "When I say 'ship it', stop iterating" → behavioral
   - "Default to the pragmatic solution" → behavioral
   - "If a test fails, investigate root cause" → behavioral

CATEGORIZATION PRIORITY (use the FIRST match):
boundaries > domain_constraints > workflow_rules > communication_rules > behavioral_rules

behavioral_rules is the LAST resort — only use it when none of the other
categories fit. Most directives should land in communication, workflow,
domain, or boundaries.

EXTRACTION RULES:
1. PRESERVE the user's EXACT phrasing. Do not rephrase or summarize.
2. If a directive spans multiple sentences, keep them together as one rule.
3. Skip meta-commentary (e.g. "this file configures...") — only extract actionable rules.
4. Each directive should be a single, self-contained instruction.
5. Be exhaustive — capture every rule, constraint, and behavioral instruction.
6. DISTRIBUTE evenly — if most rules end up in one category, re-evaluate.

<INSTRUCTIONS_FILE>
{instructions_text}
</INSTRUCTIONS_FILE>
"""
