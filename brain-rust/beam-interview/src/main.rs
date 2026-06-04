use serde::{Deserialize, Serialize};
use std::io::{self, Read};

// ── Types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct InterviewRequest {
    command: String,
    #[serde(default)]
    pass_num: u8,
    #[serde(default)]
    #[allow(dead_code)]
    existing_graph: Option<serde_json::Value>,
    #[serde(default)]
    answers: Vec<Answer>,
    #[serde(default)]
    domain: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Answer {
    question_id: String,
    question: String,
    answer: String,
    domain: String,
}

#[derive(Debug, Serialize)]
struct InterviewResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    question_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    question: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pass_num: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    followup_context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    followup_triggers: Option<Vec<String>>,
    // Analysis fields
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gaps: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interesting_signals: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_questions: Option<Vec<NextQuestion>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    graph_draft: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Serialize)]
struct NextQuestion {
    domain: String,
    reason: String,
}

// ── Question Bank ──────────────────────────────────────────

struct Question {
    id: &'static str,
    text: &'static str,
    domain: &'static str,
    pass_num: u8,
    followup_triggers: &'static [&'static str],
}

const QUESTIONS: &[Question] = &[
    // Identity (Pass 1)
    Question { id: "q_identity_001", text: "Tell me about yourself — who are you when nobody's watching?", domain: "identity", pass_num: 1, followup_triggers: &["vague", "interesting_signal"] },
    Question { id: "q_identity_002", text: "What are the 3 words you'd want people to use when describing you? What words do they actually use?", domain: "identity", pass_num: 1, followup_triggers: &["contradiction"] },
    Question { id: "q_identity_003", text: "What's a boundary you've had to enforce that cost you something? What happened?", domain: "identity", pass_num: 1, followup_triggers: &["emotional", "interesting_signal"] },
    Question { id: "q_identity_004", text: "What event in your life changed your trajectory the most? How did it shape who you are now?", domain: "identity", pass_num: 1, followup_triggers: &["emotional", "depth"] },
    Question { id: "q_identity_005", text: "What do you value most in other people? What's a dealbreaker?", domain: "identity", pass_num: 1, followup_triggers: &["vague"] },
    // Identity (Pass 2 — deep)
    Question { id: "q_identity_006", text: "You mentioned {earlier_point} — how does that show up in your daily life? Give me a concrete example.", domain: "identity", pass_num: 2, followup_triggers: &["vague"] },
    Question { id: "q_identity_007", text: "What's something you believe that most people in your life would disagree with?", domain: "identity", pass_num: 2, followup_triggers: &["interesting_signal"] },

    // Relationships (Pass 1)
    Question { id: "q_relationships_001", text: "Who are the 3 most important people in your life? What role does each play?", domain: "relationships", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_relationships_002", text: "How do you handle conflict? Walk me through what happens when someone pushes back on something you care about.", domain: "relationships", pass_num: 1, followup_triggers: &["interesting_signal", "emotional"] },
    Question { id: "q_relationships_003", text: "What's your social energy like? Are you the one who initiates plans or waits for others?", domain: "relationships", pass_num: 1, followup_triggers: &["vague"] },
    Question { id: "q_relationships_004", text: "Tell me about a relationship that ended. What did you learn from it?", domain: "relationships", pass_num: 1, followup_triggers: &["emotional"] },
    Question { id: "q_relationships_005", text: "When you're stressed, do you seek people out or withdraw? Who do you go to?", domain: "relationships", pass_num: 1, followup_triggers: &["depth"] },

    // Work (Pass 1)
    Question { id: "q_work_001", text: "How do you break down a complex problem? Walk me through your approach.", domain: "work", pass_num: 1, followup_triggers: &["depth", "interesting_signal"] },
    Question { id: "q_work_002", text: "When something breaks, what's your first instinct? Do you read logs, form a theory, or start experimenting?", domain: "work", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_work_003", text: "How do you decide when something is 'done'? What's your stopping condition?", domain: "work", pass_num: 1, followup_triggers: &["vague"] },
    Question { id: "q_work_004", text: "How do you delegate work — to people or to AI? What do you include in your instructions?", domain: "work", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_work_005", text: "What's a risk you took at work that paid off? What about one that didn't?", domain: "work", pass_num: 1, followup_triggers: &["emotional", "interesting_signal"] },

    // Emotional (Pass 1)
    Question { id: "q_emotional_001", text: "What reliably gets you excited or energized? What drains you fast?", domain: "emotional", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_emotional_002", text: "How do you process strong emotions? Do you talk it out, journal, exercise, or sit with it?", domain: "emotional", pass_num: 1, followup_triggers: &["vague"] },
    Question { id: "q_emotional_003", text: "What's something that always makes you laugh? What's something that always frustrates you?", domain: "emotional", pass_num: 1, followup_triggers: &["interesting_signal"] },
    Question { id: "q_emotional_004", text: "How do you recover after a really bad day? What's your reset ritual?", domain: "emotional", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_emotional_005", text: "When do you feel most like yourself? What's the context — who's there, what are you doing?", domain: "emotional", pass_num: 1, followup_triggers: &["emotional"] },

    // Beliefs (Pass 1)
    Question { id: "q_beliefs_001", text: "What's something you changed your mind about in the last few years? What shifted?", domain: "beliefs", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_beliefs_002", text: "What's a belief you hold that you'd defend in an argument? How strongly do you hold it?", domain: "beliefs", pass_num: 1, followup_triggers: &["interesting_signal"] },
    Question { id: "q_beliefs_003", text: "Is there a topic where your stated beliefs and your actual behavior don't match?", domain: "beliefs", pass_num: 1, followup_triggers: &["contradiction"] },
    Question { id: "q_beliefs_004", text: "What do you think most people get wrong about {their_field}?", domain: "beliefs", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_beliefs_005", text: "What's a question you wish people would ask you more often?", domain: "beliefs", pass_num: 1, followup_triggers: &["interesting_signal"] },

    // Procedural (Pass 1)
    Question { id: "q_procedural_001", text: "When you're directing an AI or writing instructions, what's your style? Give me an example of a prompt you'd write.", domain: "procedural", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_procedural_002", text: "What's your workflow for a typical work session? What triggers it, what are the steps, when do you stop?", domain: "procedural", pass_num: 1, followup_triggers: &["vague", "depth"] },
    Question { id: "q_procedural_003", text: "What's a knowledge gap you have — something you avoid, don't know about, or have an outdated view on?", domain: "procedural", pass_num: 1, followup_triggers: &["interesting_signal"] },
    Question { id: "q_procedural_004", text: "When you review someone else's work, what do you focus on? How deep do you go?", domain: "procedural", pass_num: 1, followup_triggers: &["depth"] },
    Question { id: "q_procedural_005", text: "What do you document? When do you skip it? What's your philosophy on comments and docs?", domain: "procedural", pass_num: 1, followup_triggers: &["vague"] },
];

// ── Follow-up Templates ──────────────────────────────────

fn generate_followup(trigger: &str, context: &str, answer: &str) -> Option<String> {
    match trigger {
        "vague" => {
            if answer.len() < 50 {
                Some(format!(
                    "Can you give me a specific example? You said '{}' — can you expand on that with a real situation or story?",
                    answer
                ))
            } else {
                None
            }
        }
        "emotional" => {
            let emotional_words = [
                "love", "hate", "passionate", "frustrated", "proud",
                "angry", "excited", "scared", "grateful", "overwhelmed",
                "devastated", "thrilled", "anxious", "peaceful", "hurt",
            ];
            let lower = answer.to_lowercase();
            if emotional_words.iter().any(|w| lower.contains(w)) {
                Some(format!(
                    "I can tell this is something you feel strongly about. {} What makes it resonate so deeply with you?",
                    context
                ))
            } else {
                None
            }
        }
        "contradiction" => Some(format!(
            "I noticed something interesting — {} Can you help me understand how those fit together for you?",
            context
        )),
        "depth" => Some(format!(
            "You mentioned {}. How did you develop that approach? Was there a specific experience that shaped it?",
            context
        )),
        "interesting_signal" => Some(format!(
            "That's interesting — {} Tell me more about that.",
            context
        )),
        _ => None,
    }
}

// ── Gap Analysis ──────────────────────────────────────────

fn analyze_gaps(answers: &[Answer]) -> (Vec<String>, Vec<String>, serde_json::Value) {
    let mut domain_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for a in answers {
        *domain_counts.entry(&a.domain).or_insert(0) += 1;
    }

    let all_domains = ["identity", "relationships", "work", "emotional", "beliefs", "procedural"];
    let mut gaps = Vec::new();
    let mut coverage = serde_json::Map::new();

    for &d in &all_domains {
        let count = domain_counts.get(d).copied().unwrap_or(0);
        let score = (count as f64 / 5.0).min(1.0);
        coverage.insert(d.to_string(), serde_json::json!(score));
        if count < 2 {
            gaps.push(d.to_string());
        }
    }

    // Detect interesting signals from answers
    let mut signals = Vec::new();
    for a in answers {
        let lower = a.answer.to_lowercase();
        if lower.contains("startup") || lower.contains("founded") || lower.contains("failed") {
            signals.push(format!("Startup/entrepreneurial experience in {}", a.domain));
        }
        if lower.contains("never") || lower.contains("always") {
            signals.push(format!("Strong stance detected in {}: '{}'", a.domain, &a.answer[..a.answer.len().min(80)]));
        }
    }

    (
        gaps,
        signals,
        serde_json::Value::Object(coverage),
    )
}

// ── Main ──────────────────────────────────────────────────

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    let request: InterviewRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            let resp = InterviewResponse {
                status: "error".to_string(),
                message: Some(format!("Invalid JSON input: {}", e)),
                question_id: None, question: None, domain: None, pass_num: None,
                followup_context: None, followup_triggers: None,
                coverage: None, gaps: None, interesting_signals: None,
                next_questions: None, graph_draft: None,
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            return;
        }
    };

    let response = match request.command.as_str() {
        "start" => cmd_start(),
        "continue" => cmd_continue(&request),
        "analyze" => cmd_analyze(&request),
        _ => InterviewResponse {
            status: "error".to_string(),
            message: Some(format!("Unknown command: {}", request.command)),
            question_id: None, question: None, domain: None, pass_num: None,
            followup_context: None, followup_triggers: None,
            coverage: None, gaps: None, interesting_signals: None,
            next_questions: None, graph_draft: None,
        },
    };

    println!("{}", serde_json::to_string(&response).unwrap());
}

fn cmd_start() -> InterviewResponse {
    // Start with the first identity question
    let q = &QUESTIONS[0];
    InterviewResponse {
        status: "question".to_string(),
        question_id: Some(q.id.to_string()),
        question: Some(q.text.to_string()),
        domain: Some(q.domain.to_string()),
        pass_num: Some(1),
        followup_triggers: Some(q.followup_triggers.iter().map(|s| s.to_string()).collect()),
        followup_context: None,
        coverage: None, gaps: None, interesting_signals: None,
        next_questions: None, graph_draft: None, message: None,
    }
}

fn cmd_continue(req: &InterviewRequest) -> InterviewResponse {
    let answers = &req.answers;
    let pass = req.pass_num.max(1);

    // Check for follow-up triggers from the last answer
    if let Some(last) = answers.last() {
        for trigger in &["vague", "emotional", "contradiction", "depth", "interesting_signal"] {
            if let Some(followup) = generate_followup(trigger, &last.question, &last.answer) {
                // Only follow up on certain triggers and not too many times
                let followup_count = answers.iter()
                    .filter(|a| a.question_id.contains("followup"))
                    .count();
                if followup_count < 3 && (trigger == &"vague" || trigger == &"emotional" || trigger == &"interesting_signal") {
                    return InterviewResponse {
                        status: "question".to_string(),
                        question_id: Some(format!("{}_followup_{}", last.question_id, trigger)),
                        question: Some(followup),
                        domain: Some(last.domain.clone()),
                        pass_num: Some(pass),
                        followup_context: None,
                        followup_triggers: None,
                        coverage: None, gaps: None, interesting_signals: None,
                        next_questions: None, graph_draft: None, message: None,
                    };
                }
            }
        }
    }

    // Find next unanswered question for current pass
    let answered_ids: std::collections::HashSet<&str> = answers.iter()
        .map(|a| a.question_id.as_str())
        .collect();

    let target_domain = req.domain.as_deref();

    // First try to find a question in the requested domain
    let next = QUESTIONS.iter().find(|q| {
        q.pass_num == pass
        && !answered_ids.contains(q.id)
        && target_domain.map_or(true, |d| q.domain == d)
    });

    if let Some(q) = next {
        InterviewResponse {
            status: "question".to_string(),
            question_id: Some(q.id.to_string()),
            question: Some(q.text.to_string()),
            domain: Some(q.domain.to_string()),
            pass_num: Some(pass),
            followup_triggers: Some(q.followup_triggers.iter().map(|s| s.to_string()).collect()),
            followup_context: None,
            coverage: None, gaps: None, interesting_signals: None,
            next_questions: None, graph_draft: None, message: None,
        }
    } else if pass < 3 {
        // Move to next pass
        let next_pass = pass + 1;
        let next = QUESTIONS.iter().find(|q| {
            q.pass_num == next_pass && !answered_ids.contains(q.id)
        });
        if let Some(q) = next {
            InterviewResponse {
                status: "question".to_string(),
                question_id: Some(q.id.to_string()),
                question: Some(q.text.to_string()),
                domain: Some(q.domain.to_string()),
                pass_num: Some(next_pass),
                followup_triggers: Some(q.followup_triggers.iter().map(|s| s.to_string()).collect()),
                followup_context: Some(format!("Pass {} complete. Moving to deeper questions.", pass)),
                coverage: None, gaps: None, interesting_signals: None,
                next_questions: None, graph_draft: None, message: None,
            }
        } else {
            // All passes complete
            cmd_analyze(req)
        }
    } else {
        // All passes complete
        let mut resp = cmd_analyze(req);
        resp.status = "complete".to_string();
        resp.message = Some("Interview complete! Building your brain...".to_string());
        resp
    }
}

fn cmd_analyze(req: &InterviewRequest) -> InterviewResponse {
    let (gaps, signals, coverage) = analyze_gaps(&req.answers);

    let next_questions: Vec<NextQuestion> = gaps.iter().map(|d| {
        NextQuestion {
            domain: d.clone(),
            reason: format!("Low coverage in {} domain", d),
        }
    }).collect();

    // Build a draft graph from answers
    let graph_draft = build_graph_draft(&req.answers);

    InterviewResponse {
        status: "analysis".to_string(),
        pass_num: Some(req.pass_num),
        coverage: Some(coverage),
        gaps: Some(gaps),
        interesting_signals: Some(signals),
        next_questions: Some(next_questions),
        graph_draft: Some(graph_draft),
        question_id: None, question: None, domain: None,
        followup_context: None, followup_triggers: None,
        message: None,
    }
}

fn build_graph_draft(answers: &[Answer]) -> serde_json::Value {
    // Build a partial PersonalityGraph from answers
    let mut traits = Vec::new();
    let mut beliefs = Vec::new();
    let mut values = Vec::new();

    for a in answers {
        let lower = a.answer.to_lowercase();
        // Simple keyword extraction for draft
        if a.domain == "identity" {
            traits.push(serde_json::json!({
                "name": format!("from_{}", a.question_id),
                "strength": 0.5,
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        } else if a.domain == "beliefs" {
            beliefs.push(serde_json::json!({
                "name": format!("from_{}", a.question_id),
                "confidence": 0.5,
                "sensitivity": "public",
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        } else if a.domain == "identity" && a.question_id.contains("value") {
            values.push(serde_json::json!({
                "name": format!("from_{}", a.question_id),
                "importance": 0.5,
                "summary": a.answer.chars().take(200).collect::<String>(),
            }));
        }
    }

    serde_json::json!({
        "user_summary": "Draft personality graph from interview",
        "traits": traits,
        "beliefs": beliefs,
        "values": values,
        "edges": [],
    })
}
