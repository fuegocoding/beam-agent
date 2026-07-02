"""Pure-Python writing style metrics — no LLM calls."""

import re
from collections import Counter

from brain_platform.pipeline.brain_file.schema import WritingStyle

SENTENCE_SPLIT = re.compile(r"[.!?]+\s+")

FORMAL_MARKERS = {
    "furthermore", "moreover", "consequently", "nevertheless", "therefore",
    "subsequently", "accordingly", "notwithstanding", "whereas", "hereby",
    "thus", "hence", "regarding", "concerning", "pursuant",
}
CASUAL_MARKERS = {
    "yeah", "gonna", "wanna", "kinda", "sorta", "lol", "btw", "tbh",
    "imo", "imho", "nah", "yep", "ok", "okay", "hey", "hi", "cool",
    "awesome", "stuff", "thing", "basically", "literally", "actually",
}
ANALYTICAL_MARKERS = {
    "analysis", "framework", "hypothesis", "data", "evidence", "correlation",
    "systematic", "methodology", "empirical", "quantitative", "metrics",
}
OPTIMISTIC_MARKERS = {
    "excited", "opportunity", "amazing", "love", "great", "wonderful",
    "fantastic", "promising", "optimistic", "thrilled", "passionate",
}
CRITICAL_MARKERS = {
    "problem", "issue", "concern", "flaw", "failure", "risk", "danger",
    "unfortunately", "disappointing", "skeptical", "doubt", "worried",
}


class StyleAnalyzer:
    def analyze(self, texts: list[str]) -> WritingStyle:
        if not texts:
            return WritingStyle()

        corpus = " ".join(texts)
        words = corpus.lower().split()
        word_set = set(words)
        total_words = len(words)

        if total_words == 0:
            return WritingStyle()

        # Sentence length
        sentences = SENTENCE_SPLIT.split(corpus)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )

        # Vocabulary level
        unique_ratio = len(set(words)) / total_words if total_words > 0 else 0
        if unique_ratio > 0.7 and avg_sentence_length > 20:
            vocabulary_level = "technical"
        elif unique_ratio > 0.55:
            vocabulary_level = "advanced"
        elif unique_ratio > 0.4:
            vocabulary_level = "intermediate"
        else:
            vocabulary_level = "basic"

        # Formality
        formal_count = len(word_set & FORMAL_MARKERS)
        casual_count = len(word_set & CASUAL_MARKERS)
        total_markers = formal_count + casual_count
        formality = formal_count / total_markers if total_markers > 0 else 0.5

        # Common phrases (bigrams and trigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        phrase_counts = Counter(bigrams + trigrams)
        # Filter out very common phrases
        common_phrases = [
            phrase
            for phrase, count in phrase_counts.most_common(40)
            if count >= 2 and len(phrase) > 5
        ][:20]

        # Tone
        analytical_score = len(word_set & ANALYTICAL_MARKERS)
        optimistic_score = len(word_set & OPTIMISTIC_MARKERS)
        critical_score = len(word_set & CRITICAL_MARKERS)
        tone_scores = {
            "analytical": analytical_score,
            "optimistic": optimistic_score,
            "critical": critical_score,
            "neutral": 1,
        }
        tone = max(tone_scores, key=tone_scores.get)

        return WritingStyle(
            avg_sentence_length=round(avg_sentence_length, 1),
            vocabulary_level=vocabulary_level,
            common_phrases=common_phrases,
            tone=tone,
        )
