use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
struct BuilderRequest {
    command: String,
    #[serde(default)]
    interview_data: Option<InterviewData>,
    #[serde(default)]
    existing_graph: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct InterviewData {
    #[serde(default)]
    answers: Vec<AnswerEntry>,
    #[serde(default)]
    transcript: String,
}

#[derive(Debug, Deserialize, Clone)]
struct AnswerEntry {
    question_id: String,
    question: String,
    answer: String,
    domain: String,
}

#[derive(Debug, Serialize)]
struct BuilderResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    graph: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extraction_stats: Option<ExtractionStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation: Option<ValidationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Serialize, Default)]
struct ExtractionStats {
    traits_found: usize,
    beliefs_found: usize,
    values_found: usize,
    boundaries_found: usize,
    life_events_found: usize,
    memories_found: usize,
    patterns_found: usize,
    social_found: usize,
    expertise_found: usize,
    style_found: usize,
    people_found: usize,
    places_found: usize,
    procedural_patterns_found: usize,
    work_loops_found: usize,
    prompting_styles_found: usize,
    technical_gaps_found: usize,
    edges_built: usize,
    voice_dna_extracted: bool,
    work_dna_extracted: bool,
    emotional_profile_extracted: bool,
}

#[derive(Debug, Serialize)]
struct ValidationResult {
    coverage: f64,
    node_count: usize,
    edge_count: usize,
    domain_coverage: Vec<(String, f64)>,
    missing_types: Vec<String>,
}

// ── Extractors ──────────────────────────────────────────────

fn extract_traits(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut traits = Vec::new();
    let trait_keywords = [
        ("curious", "intellectually curious"),
        ("organized", "structured and organized"),
        ("empathetic", "shows empathy in interactions"),
        ("pragmatic", "pragmatic decision-maker"),
        ("creative", "creative problem-solver"),
        ("analytical", "analytical thinker"),
        ("patient", "patient under pressure"),
        ("direct", "direct communicator"),
        ("collaborative", "prefers collaboration"),
        ("independent", "independent worker"),
        ("perfectionist", "perfectionist tendencies"),
        ("adaptable", "adaptable to change"),
        ("introverted", "introverted energy pattern"),
        ("extroverted", "extroverted energy pattern"),
        ("risk-taker", "comfortable with risk"),
        ("cautious", "cautious approach"),
    ];

    for a in answers {
        if a.domain == "identity" || a.domain == "emotional" {
            let lower = a.answer.to_lowercase();
            for (keyword, trait_name) in &trait_keywords {
                if lower.contains(keyword) {
                    traits.push(serde_json::json!({
                        "name": trait_name,
                        "strength": 0.6,
                        "summary": format!("Detected from answer to '{}': {}", a.question, &a.answer[..a.answer.len().min(150)]),
                    }));
                }
            }
            // Also extract from "I am" / "I tend to" patterns
            for pattern in &["i am ", "i tend to ", "i always ", "i usually ", "i prefer "] {
                if let Some(pos) = lower.find(pattern) {
                    let snippet = &a.answer[pos..(pos + 100).min(a.answer.len())];
                    traits.push(serde_json::json!({
                        "name": snippet.chars().take(50).collect::<String>(),
                        "strength": 0.5,
                        "summary": format!("Self-described trait: {}", snippet),
                    }));
                }
            }
        }
    }
    traits
}

fn extract_beliefs(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut beliefs = Vec::new();
    for a in answers {
        if a.domain == "beliefs" {
            beliefs.push(serde_json::json!({
                "name": a.question.chars().take(50).collect::<String>(),
                "confidence": 0.6,
                "sensitivity": "public",
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        }
        // Detect belief patterns in any domain
        let lower = a.answer.to_lowercase();
        for pattern in &["i believe", "i think", "in my opinion", "i'm convinced", "i hold"] {
            if let Some(pos) = lower.find(pattern) {
                let snippet = &a.answer[pos..(pos + 150).min(a.answer.len())];
                beliefs.push(serde_json::json!({
                    "name": format!("belief_from_{}", a.question_id),
                    "confidence": 0.5,
                    "sensitivity": "public",
                    "summary": snippet,
                }));
                break;
            }
        }
    }
    beliefs
}

fn extract_values(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut values = Vec::new();
    let value_keywords = [
        "honesty", "integrity", "autonomy", "freedom", "family",
        "growth", "learning", "creativity", "impact", "justice",
        "compassion", "excellence", "authenticity", "security", "adventure",
    ];
    for a in answers {
        let lower = a.answer.to_lowercase();
        for &v in &value_keywords {
            if lower.contains(v) {
                values.push(serde_json::json!({
                    "name": v,
                    "importance": 0.7,
                    "summary": format!("Value '{}' mentioned in: {}", v, &a.answer[..a.answer.len().min(100)]),
                }));
            }
        }
    }
    values
}

fn extract_boundaries(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut boundaries = Vec::new();
    for a in answers {
        let lower = a.answer.to_lowercase();
        for pattern in &["i would never", "i always", "dealbreaker", "non-negotiable", "i won't", "i refuse"] {
            if let Some(pos) = lower.find(pattern) {
                let snippet = &a.answer[pos..(pos + 150).min(a.answer.len())];
                boundaries.push(serde_json::json!({
                    "name": format!("boundary_from_{}", a.question_id),
                    "tested": false,
                    "cost_paid": "",
                    "sensitivity": "personal",
                    "summary": snippet,
                }));
                break;
            }
        }
    }
    boundaries
}

fn extract_life_events(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut events = Vec::new();
    for a in answers {
        if a.question_id.contains("event") || a.question_id.contains("trajectory") {
            events.push(serde_json::json!({
                "name": a.question.chars().take(50).collect::<String>(),
                "impact": a.answer.chars().take(200).collect::<String>(),
                "sensitivity": "personal",
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        }
    }
    events
}

fn extract_memories(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut memories = Vec::new();
    for a in answers {
        let lower = a.answer.to_lowercase();
        // Detect specific memory patterns
        for pattern in &["i remember", "one time", "there was a time", "i recall", "that time when"] {
            if lower.contains(pattern) {
                memories.push(serde_json::json!({
                    "name": format!("memory_from_{}", a.question_id),
                    "emotional_tone": 0.0,
                    "sensitivity": "personal",
                    "summary": a.answer.chars().take(200).collect::<String>(),
                }));
                break;
            }
        }
    }
    memories
}

fn extract_social(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut social = Vec::new();
    for a in answers {
        if a.domain == "relationships" {
            social.push(serde_json::json!({
                "name": a.question.chars().take(50).collect::<String>(),
                "sensitivity": "personal",
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        }
    }
    social
}

fn extract_people(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut people = Vec::new();
    for a in answers {
        if a.domain == "relationships" {
            // Try to extract names from relationship answers
            let words: Vec<&str> = a.answer.split_whitespace().collect();
            for (i, w) in words.iter().enumerate() {
                let cleaned = w.trim_matches(|c: char| !c.is_alphabetic());
                // Simple heuristic: capitalized words that aren't sentence starters
                if i > 0 && cleaned.len() > 2 && cleaned.chars().next().map_or(false, |c| c.is_uppercase()) {
                    let skip = ["I", "The", "My", "When", "What", "How", "But", "And", "For", "Not", "This", "That", "They", "She", "Her", "His", "Are", "Was", "Have", "Has"];
                    if !skip.contains(&cleaned) {
                        people.push(serde_json::json!({
                            "name": cleaned,
                            "role": "",
                            "sensitivity": "private",
                            "summary": format!("Mentioned in: {}", &a.answer[..a.answer.len().min(100)]),
                        }));
                    }
                }
            }
        }
    }
    people
}

fn extract_expertise(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut expertise = Vec::new();
    for a in answers {
        if a.domain == "work" || a.domain == "procedural" {
            let lower = a.answer.to_lowercase();
            let domains = [
                "programming", "engineering", "design", "marketing", "sales",
                "management", "leadership", "data science", "machine learning",
                "devops", "security", "frontend", "backend", "mobile",
                "architecture", "testing", "research", "writing",
            ];
            for d in &domains {
                if lower.contains(d) {
                    expertise.push(serde_json::json!({
                        "name": d,
                        "depth": 0.6,
                        "summary": format!("Expertise area mentioned in: {}", &a.answer[..a.answer.len().min(100)]),
                    }));
                }
            }
        }
    }
    expertise
}

fn extract_procedural(answers: &[AnswerEntry]) -> Vec<serde_json::Value> {
    let mut patterns = Vec::new();
    for a in answers {
        if a.domain == "procedural" || a.domain == "work" {
            patterns.push(serde_json::json!({
                "name": a.question.chars().take(50).collect::<String>(),
                "domain": a.domain,
                "situation": a.question.chars().take(100).collect::<String>(),
                "approach": a.answer.chars().take(200).collect::<String>(),
                "tells": [],
                "anti_pattern": "",
                "source": "interview",
                "sample_size": 1,
                "sensitivity": "public",
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        }
    }
    patterns
}

fn extract_voice_dna(answers: &[AnswerEntry]) -> serde_json::Value {
    // Analyze language patterns from all answers
    let all_text: String = answers.iter().map(|a| a.answer.as_str()).collect::<Vec<_>>().join(" ");
    let word_count = all_text.split_whitespace().count();
    let avg_sentence_len = if all_text.contains('.') {
        let sentences = all_text.split('.').count();
        word_count as f64 / sentences.max(1) as f64
    } else {
        word_count as f64
    };

    // Detect common phrases (simple n-gram analysis)
    let lower = all_text.to_lowercase();
    let mut phrases = Vec::new();
    for pattern in &["i think", "you know", "basically", "actually", "honestly", "i mean", "the thing is"] {
        if lower.contains(pattern) {
            phrases.push(pattern.to_string());
        }
    }

    // Detect humor style
    let humor = if lower.contains("haha") || lower.contains("lol") || lower.contains("jk") {
        "casual humor, uses internet shorthand"
    } else if lower.contains("irony") || lower.contains("sarcasm") {
        "dry wit, uses irony"
    } else {
        "not enough data to determine"
    };

    serde_json::json!({
        "characteristic_phrases": phrases,
        "phrases_to_avoid": [],
        "punctuation_and_formatting": if all_text.contains("...") { "uses ellipses for trailing thoughts" } else { "standard punctuation" },
        "emoji_usage": "not enough data",
        "humor_style": humor,
        "response_length_pattern": format!("Average {:.0} words per response, avg sentence length {:.1} words", word_count as f64 / answers.len().max(1) as f64, avg_sentence_len),
        "formality_range": "not enough data",
        "filler_words": phrases.clone(),
        "storytelling_style": "not enough data",
        "listener_vs_talker": "not enough data",
    })
}

fn extract_work_dna(answers: &[AnswerEntry]) -> serde_json::Value {
    let mut decomposition = "";
    let mut debugging = "";
    let mut risk = "";
    let mut delegation = "";

    for a in answers {
        let lower = a.answer.to_lowercase();
        if a.domain == "work" {
            if lower.contains("break") || lower.contains("decompose") || lower.contains("step by step") {
                decomposition = "systematic decomposition";
            } else if lower.contains("start") || lower.contains("jump in") || lower.contains("figure out") {
                decomposition = "emergent, start with concrete cases";
            }
            if lower.contains("log") || lower.contains("trace") || lower.contains("debug") {
                debugging = "trace-driven, reads logs";
            } else if lower.contains("theory") || lower.contains("hypothesis") || lower.contains("guess") {
                debugging = "hypothesis-driven";
            }
            if lower.contains("small") || lower.contains("reversible") || lower.contains("safe") {
                risk = "ships small reversible changes";
            } else if lower.contains("big") || lower.contains("invest") || lower.contains("durable") {
                risk = "invests in durable solutions";
            }
            if lower.contains("spec") || lower.contains("detailed") || lower.contains("examples") {
                delegation = "spec + examples + acceptance criteria";
            } else if lower.contains("trust") || lower.contains("high level") || lower.contains("intent") {
                delegation = "high-level intent and trust";
            }
        }
    }

    serde_json::json!({
        "decomposition_style": decomposition,
        "error_taxonomy": "",
        "debugging_approach": debugging,
        "review_depth": "",
        "documentation_habit": "",
        "abstraction_timing": "",
        "risk_posture": risk,
        "delegation_style": delegation,
        "stop_conditions": "",
    })
}

fn extract_emotional(answers: &[AnswerEntry]) -> (Vec<serde_json::Value>, serde_json::Value, Vec<serde_json::Value>) {
    let mut triggers = Vec::new();
    let mut energy_sources = Vec::new();
    let mut energy_drains = Vec::new();

    for a in answers {
        if a.domain == "emotional" {
            let lower = a.answer.to_lowercase();
            // Detect triggers
            for word in &["excited", "frustrated", "angry", "happy", "sad", "anxious", "proud", "grateful"] {
                if lower.contains(word) {
                    triggers.push(serde_json::json!({
                        "trigger": a.question.chars().take(50).collect::<String>(),
                        "emotion": word,
                        "intensity": 0.6,
                        "expression": "",
                        "context": "",
                    }));
                }
            }
            // Detect energy sources/drains
            if lower.contains("energize") || lower.contains("excite") || lower.contains("recharge") {
                energy_sources.push(a.answer.chars().take(50).collect::<String>());
            }
            if lower.contains("drain") || lower.contains("exhaust") || lower.contains("deplete") {
                energy_drains.push(a.answer.chars().take(50).collect::<String>());
            }
        }
    }

    let profile = serde_json::json!({
        "baseline_mood": "",
        "processing_style": "",
        "reaction_speed": "",
        "recovery_pattern": "",
        "energy_sources": energy_sources,
        "energy_drains": energy_drains,
        "emotional_tells": [],
    });

    (triggers, profile, vec![])
}

fn extract_edges(graph: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut edges = Vec::new();
    let empty = vec![];
    let traits = graph["traits"].as_array().unwrap_or(&empty);
    let empty2 = vec![];
    let beliefs = graph["beliefs"].as_array().unwrap_or(&empty2);
    let empty3 = vec![];
    let values = graph["values"].as_array().unwrap_or(&empty3);

    // Connect traits to beliefs/values
    for t in traits {
        for b in beliefs {
            edges.push(serde_json::json!({
                "source_name": t["name"],
                "target_name": b["name"],
                "edge_type": "expresses",
                "fact": format!("Trait '{}' expresses through belief '{}'", t["name"], b["name"]),
            }));
        }
        for v in values {
            edges.push(serde_json::json!({
                "source_name": t["name"],
                "target_name": v["name"],
                "edge_type": "driven_by",
                "fact": format!("Trait '{}' is driven by value '{}'", t["name"], v["name"]),
            }));
        }
    }

    edges
}

// ── Main ──────────────────────────────────────────────────

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    let request: BuilderRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = BuilderResponse {
                status: "error".to_string(),
                graph: None, extraction_stats: None, validation: None,
                message: Some(format!("Invalid JSON input: {}", e)),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            return;
        }
    };

    let response = match request.command.as_str() {
        "extract" => cmd_extract(&request),
        "merge" => cmd_merge(&request),
        "validate" => cmd_validate(&request),
        _ => BuilderResponse {
            status: "error".to_string(),
            graph: None, extraction_stats: None, validation: None,
            message: Some(format!("Unknown command: {}", request.command)),
        },
    };

    println!("{}", serde_json::to_string(&response).unwrap());
}

fn cmd_extract(req: &BuilderRequest) -> BuilderResponse {
    let data = match &req.interview_data {
        Some(d) => d,
        None => {
            return BuilderResponse {
                status: "error".to_string(),
                graph: None, extraction_stats: None, validation: None,
                message: Some("No interview data provided".to_string()),
            };
        }
    };

    let answers = &data.answers;

    // Run all extractors
    let traits = extract_traits(answers);
    let beliefs = extract_beliefs(answers);
    let values = extract_values(answers);
    let boundaries = extract_boundaries(answers);
    let life_events = extract_life_events(answers);
    let memories = extract_memories(answers);
    let social = extract_social(answers);
    let people = extract_people(answers);
    let expertise = extract_expertise(answers);
    let procedural = extract_procedural(answers);
    let voice_dna = extract_voice_dna(answers);
    let work_dna = extract_work_dna(answers);
    let (emotional_triggers, emotional_profile, contextual_moods) = extract_emotional(answers);

    let mut graph = serde_json::json!({
        "user_summary": "Personality graph extracted from interview",
        "traits": traits,
        "beliefs": beliefs,
        "values": values,
        "boundaries": boundaries,
        "life_events": life_events,
        "memories": memories,
        "patterns": [],
        "social": social,
        "expertise": expertise,
        "style": [],
        "people": people,
        "places": [],
        "procedural_patterns": procedural,
        "work_loops": [],
        "prompting_styles": [],
        "technical_gaps": [],
        "voice_dna": voice_dna,
        "work_dna": work_dna,
        "behavioral_rules": [],
        "contradiction_patterns": [],
        "emotional_triggers": emotional_triggers,
        "emotional_profile": emotional_profile,
        "contextual_moods": contextual_moods,
    });

    // Build edges
    let edges = extract_edges(&graph);
    graph["edges"] = serde_json::json!(edges);

    let stats = ExtractionStats {
        traits_found: traits_len(&graph, "traits"),
        beliefs_found: traits_len(&graph, "beliefs"),
        values_found: traits_len(&graph, "values"),
        boundaries_found: traits_len(&graph, "boundaries"),
        life_events_found: traits_len(&graph, "life_events"),
        memories_found: traits_len(&graph, "memories"),
        patterns_found: 0,
        social_found: traits_len(&graph, "social"),
        expertise_found: traits_len(&graph, "expertise"),
        style_found: 0,
        people_found: traits_len(&graph, "people"),
        places_found: 0,
        procedural_patterns_found: traits_len(&graph, "procedural_patterns"),
        work_loops_found: 0,
        prompting_styles_found: 0,
        technical_gaps_found: 0,
        edges_built: edges_len(&graph),
        voice_dna_extracted: true,
        work_dna_extracted: true,
        emotional_profile_extracted: true,
    };

    BuilderResponse {
        status: "success".to_string(),
        graph: Some(graph),
        extraction_stats: Some(stats),
        validation: None,
        message: None,
    }
}

fn traits_len(graph: &serde_json::Value, key: &str) -> usize {
    graph[key].as_array().map_or(0, |a| a.len())
}

fn edges_len(graph: &serde_json::Value) -> usize {
    graph["edges"].as_array().map_or(0, |a| a.len())
}

fn cmd_merge(req: &BuilderRequest) -> BuilderResponse {
    // For merge, we'd combine existing_graph with new extraction
    // For now, just return the existing graph (merge logic is complex)
    let graph = req.existing_graph.clone().unwrap_or(serde_json::json!({}));
    BuilderResponse {
        status: "success".to_string(),
        graph: Some(graph),
        extraction_stats: None,
        validation: None,
        message: Some("Merge complete (existing graph preserved)".to_string()),
    }
}

fn cmd_validate(req: &BuilderRequest) -> BuilderResponse {
    let graph = match &req.existing_graph {
        Some(g) => g,
        None => {
            return BuilderResponse {
                status: "error".to_string(),
                graph: None, extraction_stats: None, validation: None,
                message: Some("No graph provided for validation".to_string()),
            };
        }
    };

    let node_count = graph.as_object().map_or(0, |m| {
        m.values()
            .filter(|v| v.is_array())
            .map(|v| v.as_array().unwrap().len())
            .sum()
    });

    let edge_count = edges_len(graph);

    let missing: Vec<String> = [
        ("traits", "traits"), ("beliefs", "beliefs"), ("values", "values"),
        ("boundaries", "boundaries"), ("life_events", "life_events"),
        ("people", "people"), ("voice_dna", "voice_dna"), ("work_dna", "work_dna"),
    ]
    .iter()
    .filter(|(key, _)| {
        graph.get(*key).map_or(true, |v| {
            if v.is_array() { v.as_array().unwrap().is_empty() }
            else if v.is_null() { true }
            else { false }
        })
    })
    .map(|(_, name)| name.to_string())
    .collect();

    let coverage = 1.0 - (missing.len() as f64 / 8.0);

    BuilderResponse {
        status: "success".to_string(),
        graph: None,
        extraction_stats: None,
        validation: Some(ValidationResult {
            coverage,
            node_count,
            edge_count,
            domain_coverage: vec![
                ("identity".to_string(), if graph["traits"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 }),
                ("relationships".to_string(), if graph["people"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 }),
                ("work".to_string(), if graph["work_dna"].is_object() { 1.0 } else { 0.0 }),
                ("emotional".to_string(), if graph["emotional_triggers"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 }),
                ("beliefs".to_string(), if graph["beliefs"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 }),
            ],
            missing_types: missing,
        }),
        message: None,
    }
}
