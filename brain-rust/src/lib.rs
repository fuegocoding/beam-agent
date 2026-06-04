use serde::{Deserialize, Serialize};

// ── Enums ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SensitivityLevel {
    Public,
    Personal,
    Private,
}

impl Default for SensitivityLevel {
    fn default() -> Self {
        Self::Public
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ProceduralSource {
    Interview,
    Artifact,
    Synthesis,
}

impl Default for ProceduralSource {
    fn default() -> Self {
        Self::Interview
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GapType {
    Avoids,
    Unaware,
    Outdated,
}

impl Default for GapType {
    fn default() -> Self {
        Self::Avoids
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContradictionPolicy {
    Aspirational,
    Observed,
    Both,
}

impl Default for ContradictionPolicy {
    fn default() -> Self {
        Self::Both
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HarmGate {
    Off,
    Soften,
    Exclude,
}

impl Default for HarmGate {
    fn default() -> Self {
        Self::Soften
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Shaped,
    EvolvedInto,
    Enforces,
    Guides,
    Taught,
    Expresses,
    Tested,
    Involves,
    HasTrait,
    Holds,
    DrivenBy,
    Experienced,
    ExpertIn,
    HandlesConflictBy,
    CommunicatesVia,
    OperatesIn,
    ExplainedBy,
    AspiresTo,
    Contradicts,
    WorksBy,
    LeadsTo,
    Caused,
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::Involves
    }
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_value(self)
            .unwrap_or_default()
            .as_str()
            .unwrap_or("involves")
            .to_string();
        write!(f, "{}", s)
    }
}

// ── Node Types ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraitNode {
    pub name: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
    #[serde(default)]
    pub summary: String,
}

fn default_strength() -> f64 {
    0.5
}
fn default_confidence() -> f64 {
    0.5
}
fn default_importance() -> f64 {
    0.5
}
fn default_depth() -> f64 {
    0.5
}
fn default_intensity() -> f64 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BeliefNode {
    pub name: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValueNode {
    pub name: String,
    #[serde(default = "default_importance")]
    pub importance: f64,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BoundaryNode {
    pub name: String,
    #[serde(default)]
    pub tested: bool,
    #[serde(default)]
    pub cost_paid: String,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifeEventNode {
    pub name: String,
    #[serde(default)]
    pub impact: String,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryNode {
    pub name: String,
    #[serde(default)]
    pub emotional_tone: f64,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternNode {
    pub name: String,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SocialNode {
    pub name: String,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExpertiseNode {
    pub name: String,
    #[serde(default = "default_depth")]
    pub depth: f64,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StyleNode {
    pub name: String,
    #[serde(default)]
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersonNode {
    pub name: String,
    #[serde(default)]
    pub role: String,
    #[serde(default = "default_private")]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
}

fn default_private() -> SensitivityLevel {
    SensitivityLevel::Private
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlaceNode {
    pub name: String,
    #[serde(default)]
    pub significance: String,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
}

// ── Procedural Memory Layer ────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationKnobs {
    #[serde(default = "default_tone_temp")]
    pub tone_temperature: f64,
    #[serde(default)]
    pub confidence_clip: Option<f64>,
    #[serde(default)]
    pub contradiction_policy: ContradictionPolicy,
    #[serde(default = "default_true")]
    pub factual_correction: bool,
    #[serde(default)]
    pub harm_gate: HarmGate,
}

fn default_tone_temp() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Default for CalibrationKnobs {
    fn default() -> Self {
        Self {
            tone_temperature: 1.0,
            confidence_clip: None,
            contradiction_policy: ContradictionPolicy::Both,
            factual_correction: true,
            harm_gate: HarmGate::Soften,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProceduralPatternNode {
    pub name: String,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub situation: String,
    #[serde(default)]
    pub approach: String,
    #[serde(default)]
    pub tells: Vec<String>,
    #[serde(default)]
    pub anti_pattern: String,
    #[serde(default)]
    pub source: ProceduralSource,
    #[serde(default = "default_sample_size")]
    pub sample_size: i64,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub calibration: Option<CalibrationKnobs>,
}

fn default_sample_size() -> i64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkLoopNode {
    pub name: String,
    #[serde(default)]
    pub trigger: String,
    #[serde(default)]
    pub steps: Vec<String>,
    #[serde(default)]
    pub stop_condition: String,
    #[serde(default)]
    pub recovery: String,
    #[serde(default)]
    pub source: ProceduralSource,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub calibration: Option<CalibrationKnobs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptingStyleNode {
    pub name: String,
    #[serde(default)]
    pub structure: String,
    #[serde(default)]
    pub length_preference: String,
    #[serde(default)]
    pub correction_style: String,
    #[serde(default)]
    pub constraint_phrasing: String,
    #[serde(default)]
    pub examples_excerpts: Vec<String>,
    #[serde(default)]
    pub source: ProceduralSource,
    #[serde(default)]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub calibration: Option<CalibrationKnobs>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TechnicalGapNode {
    pub name: String,
    #[serde(default)]
    pub gap_type: GapType,
    #[serde(default)]
    pub evidence: String,
    #[serde(default)]
    pub aspirational: bool,
    #[serde(default = "default_personal")]
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub calibration: Option<CalibrationKnobs>,
}

fn default_personal() -> SensitivityLevel {
    SensitivityLevel::Personal
}

// ── DNA Types ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VoiceDna {
    #[serde(default)]
    pub characteristic_phrases: Vec<String>,
    #[serde(default)]
    pub phrases_to_avoid: Vec<String>,
    #[serde(default)]
    pub punctuation_and_formatting: String,
    #[serde(default)]
    pub emoji_usage: String,
    #[serde(default)]
    pub humor_style: String,
    #[serde(default)]
    pub response_length_pattern: String,
    #[serde(default)]
    pub formality_range: String,
    #[serde(default)]
    pub filler_words: Vec<String>,
    #[serde(default)]
    pub storytelling_style: String,
    #[serde(default)]
    pub listener_vs_talker: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkDna {
    #[serde(default)]
    pub decomposition_style: String,
    #[serde(default)]
    pub error_taxonomy: String,
    #[serde(default)]
    pub debugging_approach: String,
    #[serde(default)]
    pub review_depth: String,
    #[serde(default)]
    pub documentation_habit: String,
    #[serde(default)]
    pub abstraction_timing: String,
    #[serde(default)]
    pub risk_posture: String,
    #[serde(default)]
    pub delegation_style: String,
    #[serde(default)]
    pub stop_conditions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BehavioralRule {
    pub trigger: String,
    pub response: String,
    #[serde(default)]
    pub exceptions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContradictionPattern {
    pub topic: String,
    pub stance: String,
    #[serde(default)]
    pub how_they_push_back: String,
}

// ── Emotional Types ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalTrigger {
    pub trigger: String,
    pub emotion: String,
    #[serde(default = "default_intensity")]
    pub intensity: f64,
    #[serde(default)]
    pub expression: String,
    #[serde(default)]
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalProfile {
    #[serde(default)]
    pub baseline_mood: String,
    #[serde(default)]
    pub processing_style: String,
    #[serde(default)]
    pub reaction_speed: String,
    #[serde(default)]
    pub recovery_pattern: String,
    #[serde(default)]
    pub energy_sources: Vec<String>,
    #[serde(default)]
    pub energy_drains: Vec<String>,
    #[serde(default)]
    pub emotional_tells: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextualMood {
    pub context: String,
    #[serde(default)]
    pub mood: String,
    #[serde(default = "default_guard")]
    pub guard_level: f64,
    #[serde(default = "default_energy")]
    pub energy_level: f64,
}

fn default_guard() -> f64 {
    0.5
}
fn default_energy() -> f64 {
    0.5
}

// ── Edge ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeSpec {
    pub source_name: String,
    pub target_name: String,
    #[serde(default)]
    pub edge_type: EdgeType,
    #[serde(default)]
    pub fact: String,
}

// ── Complete Graph ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersonalityGraph {
    #[serde(default)]
    pub user_summary: String,

    #[serde(default)]
    pub traits: Vec<TraitNode>,
    #[serde(default)]
    pub beliefs: Vec<BeliefNode>,
    #[serde(default)]
    pub values: Vec<ValueNode>,
    #[serde(default)]
    pub boundaries: Vec<BoundaryNode>,
    #[serde(default)]
    pub life_events: Vec<LifeEventNode>,
    #[serde(default)]
    pub memories: Vec<MemoryNode>,
    #[serde(default)]
    pub patterns: Vec<PatternNode>,
    #[serde(default)]
    pub social: Vec<SocialNode>,
    #[serde(default)]
    pub expertise: Vec<ExpertiseNode>,
    #[serde(default)]
    pub style: Vec<StyleNode>,
    #[serde(default)]
    pub people: Vec<PersonNode>,
    #[serde(default)]
    pub places: Vec<PlaceNode>,

    // Procedural memory layer
    #[serde(default)]
    pub procedural_patterns: Vec<ProceduralPatternNode>,
    #[serde(default)]
    pub work_loops: Vec<WorkLoopNode>,
    #[serde(default)]
    pub prompting_styles: Vec<PromptingStyleNode>,
    #[serde(default)]
    pub technical_gaps: Vec<TechnicalGapNode>,

    #[serde(default)]
    pub edges: Vec<EdgeSpec>,

    // Clone specification
    #[serde(default)]
    pub voice_dna: Option<VoiceDna>,
    #[serde(default)]
    pub work_dna: Option<WorkDna>,
    #[serde(default)]
    pub behavioral_rules: Vec<BehavioralRule>,
    #[serde(default)]
    pub contradiction_patterns: Vec<ContradictionPattern>,

    // Emotional dynamics
    #[serde(default)]
    pub emotional_triggers: Vec<EmotionalTrigger>,
    #[serde(default)]
    pub emotional_profile: Option<EmotionalProfile>,
    #[serde(default)]
    pub contextual_moods: Vec<ContextualMood>,
}

impl PersonalityGraph {
    /// Count total nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.traits.len()
            + self.beliefs.len()
            + self.values.len()
            + self.boundaries.len()
            + self.life_events.len()
            + self.memories.len()
            + self.patterns.len()
            + self.social.len()
            + self.expertise.len()
            + self.style.len()
            + self.people.len()
            + self.places.len()
            + self.procedural_patterns.len()
            + self.work_loops.len()
            + self.prompting_styles.len()
            + self.technical_gaps.len()
    }

    /// Coverage score 0-1 (how many node types have at least one entry).
    pub fn coverage(&self) -> f64 {
        let types_with_data = [
            !self.traits.is_empty(),
            !self.beliefs.is_empty(),
            !self.values.is_empty(),
            !self.boundaries.is_empty(),
            !self.life_events.is_empty(),
            !self.memories.is_empty(),
            !self.patterns.is_empty(),
            !self.social.is_empty(),
            !self.expertise.is_empty(),
            !self.style.is_empty(),
            !self.people.is_empty(),
            !self.places.is_empty(),
            !self.procedural_patterns.is_empty(),
            !self.work_loops.is_empty(),
            !self.prompting_styles.is_empty(),
            !self.technical_gaps.is_empty(),
            self.voice_dna.is_some(),
            self.work_dna.is_some(),
            !self.emotional_triggers.is_empty(),
            self.emotional_profile.is_some(),
        ];
        let filled = types_with_data.iter().filter(|&&b| b).count();
        filled as f64 / types_with_data.len() as f64
    }

    /// Domain coverage map.
    pub fn domain_coverage(&self) -> Vec<(&str, f64)> {
        vec![
            ("identity", self.identity_score()),
            ("relationships", self.relationships_score()),
            ("work", self.work_score()),
            ("emotional", self.emotional_score()),
            ("beliefs", self.beliefs_score()),
            ("procedural", self.procedural_score()),
        ]
    }

    fn identity_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if !self.traits.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.values.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.boundaries.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.life_events.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.memories.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }

    fn relationships_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if !self.people.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.social.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }

    fn work_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if self.work_dna.is_some() {
            s += 1.0;
        }
        c += 1;
        if !self.expertise.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.procedural_patterns.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.work_loops.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }

    fn emotional_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if !self.emotional_triggers.is_empty() {
            s += 1.0;
        }
        c += 1;
        if self.emotional_profile.is_some() {
            s += 1.0;
        }
        c += 1;
        if !self.contextual_moods.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }

    fn beliefs_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if !self.beliefs.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.contradiction_patterns.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }

    fn procedural_score(&self) -> f64 {
        let mut s = 0.0;
        let mut c = 0;
        if !self.procedural_patterns.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.work_loops.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.prompting_styles.is_empty() {
            s += 1.0;
        }
        c += 1;
        if !self.technical_gaps.is_empty() {
            s += 1.0;
        }
        c += 1;
        s / c as f64
    }
}
