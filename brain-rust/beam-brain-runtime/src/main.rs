use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
struct RuntimeRequest {
    command: String,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    graph: Option<serde_json::Value>,
    #[serde(default)]
    trust_level: Option<String>,
    #[serde(default)]
    brain_power: Option<String>,
}

#[derive(Debug, Serialize)]
struct RuntimeResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<Vec<SearchResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edges: Option<Vec<EdgeResult>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sections_included: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_estimate: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    soul_md: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sections: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<GraphStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    node_type: String,
    name: String,
    summary: String,
    relevance: f64,
}

#[derive(Debug, Serialize)]
struct EdgeResult {
    source: String,
    target: String,
    relation: String,
    fact: String,
}

#[derive(Debug, Serialize)]
struct GraphStats {
    node_count: usize,
    edge_count: usize,
    coverage: f64,
    domain_coverage: HashMap<String, f64>,
    has_voice_dna: bool,
    has_work_dna: bool,
    has_emotional_profile: bool,
}

// ── Search ──────────────────────────────────────────────────

fn search_graph(
    graph: &serde_json::Value,
    query: &str,
    trust_level: &str,
    brain_power: &str,
) -> (Vec<SearchResult>, Vec<EdgeResult>) {
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();

    let max_results = match brain_power {
        "light" => 3,
        "full" => 100,
        _ => 10, // standard
    };

    let mut results = Vec::new();

    // Search all node arrays
    let node_arrays = [
        ("Trait", "traits"),
        ("Belief", "beliefs"),
        ("Value", "values"),
        ("Boundary", "boundaries"),
        ("LifeEvent", "life_events"),
        ("Memory", "memories"),
        ("Pattern", "patterns"),
        ("Social", "social"),
        ("Expertise", "expertise"),
        ("Style", "style"),
        ("Person", "people"),
        ("Place", "places"),
        ("ProceduralPattern", "procedural_patterns"),
        ("WorkLoop", "work_loops"),
        ("PromptingStyle", "prompting_styles"),
        ("TechnicalGap", "technical_gaps"),
    ];

    for (node_type, key) in &node_arrays {
        if let Some(arr) = graph[key].as_array() {
            for node in arr {
                // Check sensitivity filter
                let sensitivity = node["sensitivity"].as_str().unwrap_or("public");
                if !passes_trust_filter(sensitivity, trust_level) {
                    continue;
                }

                let name = node["name"].as_str().unwrap_or("");
                let summary = node["summary"].as_str().unwrap_or("");
                let combined = format!("{} {}", name, summary).to_lowercase();

                let relevance = compute_relevance(&combined, &query_words);
                if relevance > 0.05 {
                    results.push(SearchResult {
                        node_type: node_type.to_string(),
                        name: name.to_string(),
                        summary: summary.chars().take(200).collect(),
                        relevance,
                    });
                }
            }
        }
    }

    // Search voice_dna and work_dna
    if let Some(voice) = graph.get("voice_dna") {
        let phrases: Vec<String> = voice["characteristic_phrases"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let combined = phrases.join(" ").to_lowercase();
        let relevance = compute_relevance(&combined, &query_words);
        if relevance > 0.05 {
            results.push(SearchResult {
                node_type: "VoiceDna".to_string(),
                name: "Communication Style".to_string(),
                summary: format!("Characteristic phrases: {}", phrases.join(", ")),
                relevance,
            });
        }
    }

    if let Some(work) = graph.get("work_dna") {
        let fields: Vec<String> = [
            "decomposition_style", "debugging_approach", "risk_posture", "delegation_style",
        ]
        .iter()
        .filter_map(|k| work[*k].as_str().map(String::from))
        .collect();
        let combined = fields.join(" ").to_lowercase();
        let relevance = compute_relevance(&combined, &query_words);
        if relevance > 0.05 {
            results.push(SearchResult {
                node_type: "WorkDna".to_string(),
                name: "Work Style".to_string(),
                summary: format!("Work approach: {}", fields.join(", ")),
                relevance,
            });
        }
    }

    // Sort by relevance, take top N
    results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(max_results);

    // Find edges connected to result nodes
    let result_names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
    let mut matched_edges = Vec::new();

    if let Some(edges) = graph["edges"].as_array() {
        for edge in edges {
            let source = edge["source_name"].as_str().unwrap_or("");
            let target = edge["target_name"].as_str().unwrap_or("");
            if result_names.contains(&source) || result_names.contains(&target) {
                matched_edges.push(EdgeResult {
                    source: source.to_string(),
                    target: target.to_string(),
                    relation: edge["edge_type"].as_str().unwrap_or("involves").to_string(),
                    fact: edge["fact"].as_str().unwrap_or("").to_string(),
                });
            }
        }
    }

    (results, matched_edges)
}

fn passes_trust_filter(sensitivity: &str, trust_level: &str) -> bool {
    match trust_level {
        "visitor" => sensitivity == "public",
        "known" => sensitivity == "public" || sensitivity == "personal",
        _ => true, // owner sees all
    }
}

fn compute_relevance(text: &str, query_words: &[&str]) -> f64 {
    if query_words.is_empty() {
        return 0.0;
    }
    let matches = query_words.iter().filter(|w| text.contains(**w)).count();
    matches as f64 / query_words.len() as f64
}

// ── Context Builder ──────────────────────────────────────

fn build_context(
    graph: &serde_json::Value,
    trust_level: &str,
    brain_power: &str,
) -> (String, Vec<String>, usize) {
    let mut sections = Vec::new();
    let mut context = String::new();
    let full = brain_power == "full";

    // Identity section
    if let Some(traits) = graph["traits"].as_array() {
        if !traits.is_empty() {
            context.push_str("## Your Core Identity\n\n");
            let limit = if full { traits.len() } else { 3.min(traits.len()) };
            for t in &traits[..limit] {
                if passes_trust_filter(t["sensitivity"].as_str().unwrap_or("public"), trust_level) {
                    let name = t["name"].as_str().unwrap_or("");
                    let summary = t["summary"].as_str().unwrap_or("");
                    context.push_str(&format!("- **{}**: {}\n", name, summary));
                }
            }
            context.push('\n');
            sections.push("identity".to_string());
        }
    }

    // Values section
    if let Some(values) = graph["values"].as_array() {
        if !values.is_empty() {
            context.push_str("## Your Values\n\n");
            let limit = if full { values.len() } else { 3.min(values.len()) };
            for v in &values[..limit] {
                let name = v["name"].as_str().unwrap_or("");
                let summary = v["summary"].as_str().unwrap_or("");
                context.push_str(&format!("- **{}**: {}\n", name, summary));
            }
            context.push('\n');
            sections.push("values".to_string());
        }
    }

    // Beliefs section
    if let Some(beliefs) = graph["beliefs"].as_array() {
        if !beliefs.is_empty() {
            context.push_str("## Your Beliefs\n\n");
            let limit = if full { beliefs.len() } else { 3.min(beliefs.len()) };
            for b in &beliefs[..limit] {
                if passes_trust_filter(b["sensitivity"].as_str().unwrap_or("public"), trust_level) {
                    let name = b["name"].as_str().unwrap_or("");
                    let summary = b["summary"].as_str().unwrap_or("");
                    context.push_str(&format!("- **{}**: {}\n", name, summary));
                }
            }
            context.push('\n');
            sections.push("beliefs".to_string());
        }
    }

    // Style section
    if let Some(voice) = graph.get("voice_dna") {
        context.push_str("## Communication Style\n\n");
        if let Some(humor) = voice["humor_style"].as_str() {
            if !humor.is_empty() && humor != "not enough data to determine" {
                context.push_str(&format!("- Humor: {}\n", humor));
            }
        }
        if let Some(length) = voice["response_length_pattern"].as_str() {
            if !length.is_empty() && length != "not enough data" {
                context.push_str(&format!("- Response style: {}\n", length));
            }
        }
        if let Some(phrases) = voice["characteristic_phrases"].as_array() {
            if !phrases.is_empty() {
                let p: Vec<&str> = phrases.iter().filter_map(|v| v.as_str()).collect();
                context.push_str(&format!("- Common phrases: {}\n", p.join(", ")));
            }
        }
        context.push('\n');
        sections.push("style".to_string());
    }

    // Work section (for standard + full)
    if brain_power != "light" {
        if let Some(work) = graph.get("work_dna") {
            context.push_str("## Work Style\n\n");
            for field in &["decomposition_style", "debugging_approach", "risk_posture", "delegation_style"] {
                if let Some(val) = work[*field].as_str() {
                    if !val.is_empty() {
                        context.push_str(&format!("- {}: {}\n", field.replace('_', " "), val));
                    }
                }
            }
            context.push('\n');
            sections.push("work".to_string());
        }
    }

    // People section (for full)
    if full {
        if let Some(people) = graph["people"].as_array() {
            if !people.is_empty() {
                context.push_str("## Key People\n\n");
                for p in people {
                    if passes_trust_filter(p["sensitivity"].as_str().unwrap_or("private"), trust_level) {
                        let name = p["name"].as_str().unwrap_or("");
                        let role = p["role"].as_str().unwrap_or("");
                        context.push_str(&format!("- **{}**: {}\n", name, role));
                    }
                }
                context.push('\n');
                sections.push("people".to_string());
            }
        }
    }

    let token_estimate = context.split_whitespace().count() * 4 / 3; // rough token estimate

    (context, sections, token_estimate)
}

// ── SOUL.md Generator ──────────────────────────────────

fn generate_soul_md(graph: &serde_json::Value) -> (String, Vec<String>) {
    let mut soul = String::from("# SOUL.md\n\n");
    let mut sections = Vec::new();

    // Who You Are
    soul.push_str("## Who You Are\n\n");
    if let Some(summary) = graph["user_summary"].as_str() {
        if !summary.is_empty() {
            soul.push_str(&format!("{}\n\n", summary));
        }
    }
    if let Some(traits) = graph["traits"].as_array() {
        if !traits.is_empty() {
            soul.push_str("Key traits: ");
            let trait_names: Vec<&str> = traits.iter()
                .filter_map(|t| t["name"].as_str())
                .collect();
            soul.push_str(&trait_names.join(", "));
            soul.push_str(".\n\n");
        }
    }
    sections.push("identity".to_string());

    // Values
    if let Some(values) = graph["values"].as_array() {
        if !values.is_empty() {
            soul.push_str("## Your Values\n\n");
            for v in values {
                let name = v["name"].as_str().unwrap_or("");
                let summary = v["summary"].as_str().unwrap_or("");
                if !name.is_empty() {
                    soul.push_str(&format!("- **{}**: {}\n", name, summary));
                }
            }
            soul.push('\n');
            sections.push("values".to_string());
        }
    }

    // Communication Style
    if let Some(voice) = graph.get("voice_dna") {
        soul.push_str("## Communication Style\n\n");
        let mut has_content = false;
        if let Some(humor) = voice["humor_style"].as_str() {
            if !humor.is_empty() && humor != "not enough data to determine" {
                soul.push_str(&format!("- Humor: {}\n", humor));
                has_content = true;
            }
        }
        if let Some(phrases) = voice["characteristic_phrases"].as_array() {
            if !phrases.is_empty() {
                let p: Vec<&str> = phrases.iter().filter_map(|v| v.as_str()).collect();
                soul.push_str(&format!("- Common phrases: {}\n", p.join(", ")));
                has_content = true;
            }
        }
        if let Some(length) = voice["response_length_pattern"].as_str() {
            if !length.is_empty() && length != "not enough data" {
                soul.push_str(&format!("- Response style: {}\n", length));
                has_content = true;
            }
        }
        if let Some(avoid) = voice["phrases_to_avoid"].as_array() {
            if !avoid.is_empty() {
                let a: Vec<&str> = avoid.iter().filter_map(|v| v.as_str()).collect();
                soul.push_str(&format!("- Avoid saying: {}\n", a.join(", ")));
                has_content = true;
            }
        }
        if has_content {
            soul.push('\n');
            sections.push("voice".to_string());
        }
    }

    // Boundaries
    if let Some(boundaries) = graph["boundaries"].as_array() {
        if !boundaries.is_empty() {
            soul.push_str("## Boundaries\n\n");
            for b in boundaries {
                let name = b["name"].as_str().unwrap_or("");
                let summary = b["summary"].as_str().unwrap_or("");
                if !name.is_empty() {
                    soul.push_str(&format!("- **{}**: {}\n", name, summary));
                }
            }
            soul.push('\n');
            sections.push("boundaries".to_string());
        }
    }

    // Work Style
    if let Some(work) = graph.get("work_dna") {
        soul.push_str("## Work Style\n\n");
        let mut has_content = false;
        for field in &["decomposition_style", "debugging_approach", "risk_posture", "delegation_style", "stop_conditions"] {
            if let Some(val) = work[*field].as_str() {
                if !val.is_empty() {
                    soul.push_str(&format!("- {}: {}\n", field.replace('_', " "), val));
                    has_content = true;
                }
            }
        }
        if has_content {
            soul.push('\n');
            sections.push("work_style".to_string());
        }
    }

    (soul, sections)
}

// ── Stats ──────────────────────────────────────────────────

fn compute_stats(graph: &serde_json::Value) -> GraphStats {
    let node_arrays = [
        "traits", "beliefs", "values", "boundaries", "life_events", "memories",
        "patterns", "social", "expertise", "style", "people", "places",
        "procedural_patterns", "work_loops", "prompting_styles", "technical_gaps",
    ];

    let node_count: usize = node_arrays.iter()
        .filter_map(|k| graph[*k].as_array())
        .map(|a| a.len())
        .sum();

    let edge_count = graph["edges"].as_array().map_or(0, |a| a.len());

    let types_with_data: usize = [
        graph["traits"].as_array().map_or(0, |a| a.len()) > 0,
        graph["beliefs"].as_array().map_or(0, |a| a.len()) > 0,
        graph["values"].as_array().map_or(0, |a| a.len()) > 0,
        graph["boundaries"].as_array().map_or(0, |a| a.len()) > 0,
        graph["life_events"].as_array().map_or(0, |a| a.len()) > 0,
        graph["memories"].as_array().map_or(0, |a| a.len()) > 0,
        graph["patterns"].as_array().map_or(0, |a| a.len()) > 0,
        graph["social"].as_array().map_or(0, |a| a.len()) > 0,
        graph["expertise"].as_array().map_or(0, |a| a.len()) > 0,
        graph["style"].as_array().map_or(0, |a| a.len()) > 0,
        graph["people"].as_array().map_or(0, |a| a.len()) > 0,
        graph["places"].as_array().map_or(0, |a| a.len()) > 0,
        graph["procedural_patterns"].as_array().map_or(0, |a| a.len()) > 0,
        graph["work_loops"].as_array().map_or(0, |a| a.len()) > 0,
        graph["prompting_styles"].as_array().map_or(0, |a| a.len()) > 0,
        graph["technical_gaps"].as_array().map_or(0, |a| a.len()) > 0,
        graph.get("voice_dna").map_or(false, |v| v.is_object()),
        graph.get("work_dna").map_or(false, |v| v.is_object()),
        graph["emotional_triggers"].as_array().map_or(0, |a| a.len()) > 0,
        graph.get("emotional_profile").map_or(false, |v| v.is_object()),
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    let coverage = types_with_data as f64 / 20.0;

    let mut domain_coverage = HashMap::new();
    domain_coverage.insert("identity".to_string(), if graph["traits"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 });
    domain_coverage.insert("relationships".to_string(), if graph["people"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 });
    domain_coverage.insert("work".to_string(), if graph.get("work_dna").map_or(false, |v| v.is_object()) { 1.0 } else { 0.0 });
    domain_coverage.insert("emotional".to_string(), if graph["emotional_triggers"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 });
    domain_coverage.insert("beliefs".to_string(), if graph["beliefs"].as_array().map_or(0, |a| a.len()) > 0 { 1.0 } else { 0.0 });

    GraphStats {
        node_count,
        edge_count,
        coverage,
        domain_coverage,
        has_voice_dna: graph.get("voice_dna").map_or(false, |v| v.is_object()),
        has_work_dna: graph.get("work_dna").map_or(false, |v| v.is_object()),
        has_emotional_profile: graph.get("emotional_profile").map_or(false, |v| v.is_object()),
    }
}

// ── Main ──────────────────────────────────────────────────

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    let request: RuntimeRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = RuntimeResponse {
                status: "error".to_string(),
                results: None, edges: None, context: None, sections_included: None,
                token_estimate: None, soul_md: None, sections: None, stats: None,
                message: Some(format!("Invalid JSON input: {}", e)),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            return;
        }
    };

    let graph = match &request.graph {
        Some(g) => g,
        None => {
            let resp = RuntimeResponse {
                status: "error".to_string(),
                results: None, edges: None, context: None, sections_included: None,
                token_estimate: None, soul_md: None, sections: None, stats: None,
                message: Some("No graph provided".to_string()),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            return;
        }
    };

    let trust = request.trust_level.as_deref().unwrap_or("owner");
    let power = request.brain_power.as_deref().unwrap_or("standard");

    let response = match request.command.as_str() {
        "search" => {
            let query = request.query.as_deref().unwrap_or("");
            let (results, edges) = search_graph(graph, query, trust, power);
            RuntimeResponse {
                status: "success".to_string(),
                results: Some(results),
                edges: Some(edges),
                context: None, sections_included: None, token_estimate: None,
                soul_md: None, sections: None, stats: None, message: None,
            }
        }
        "context" => {
            let (context, sections_included, token_estimate) = build_context(graph, trust, power);
            RuntimeResponse {
                status: "success".to_string(),
                results: None, edges: None,
                context: Some(context),
                sections_included: Some(sections_included),
                token_estimate: Some(token_estimate),
                soul_md: None, sections: None, stats: None, message: None,
            }
        }
        "export_soul" => {
            let (soul_md, sections) = generate_soul_md(graph);
            RuntimeResponse {
                status: "success".to_string(),
                results: None, edges: None, context: None, sections_included: None,
                token_estimate: None,
                soul_md: Some(soul_md),
                sections: Some(sections),
                stats: None, message: None,
            }
        }
        "stats" => {
            let stats = compute_stats(graph);
            RuntimeResponse {
                status: "success".to_string(),
                results: None, edges: None, context: None, sections_included: None,
                token_estimate: None, soul_md: None, sections: None,
                stats: Some(stats),
                message: None,
            }
        }
        _ => RuntimeResponse {
            status: "error".to_string(),
            results: None, edges: None, context: None, sections_included: None,
            token_estimate: None, soul_md: None, sections: None, stats: None,
            message: Some(format!("Unknown command: {}", request.command)),
        },
    };

    println!("{}", serde_json::to_string(&response).unwrap());
}
