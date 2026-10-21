"""Stylometric Feature Extraction Engine (Module B).

Extracts the 12 stylometric features specified in Section 4.5 and Appendix A & C
of the project report.
"""

from typing import List, Dict, Set
import re
import math


# Appendix C Generic-phrase lexicon
DEFAULT_GENERIC_LEXICON: Set[str] = {
    "highly recommend",
    "best product ever",
    "must buy",
    "works great",
    "love it",
    "five stars",
    "exceeded my expectations",
    "perfect gift",
    "great quality",
    "would buy again",
    "fast shipping",
    "worth every penny",
    "best purchase ever",
    "do not buy",
    "worst product ever",
    "complete garbage",
    "terrible scam",
    "waste of money",
    "10/10",
    "10 out of 10"
}

FIRST_PERSON_PRONOUNS: Set[str] = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}

SUPERLATIVE_WORDS: Set[str] = {
    "best", "worst", "greatest", "perfect", "flawless", "finest", "highest",
    "lowest", "cheapest", "ultimate", "absolute", "top"
}


def compute_flesch_reading_ease(text: str) -> float:
    """Calculates Flesch Reading Ease score without requiring external binary dependencies:

    Score = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    num_sentences = max(len(sentences), 1)

    words = re.findall(r"\b[A-Za-z]+\b", text)
    num_words = max(len(words), 1)

    # Estimate syllables
    total_syllables = 0
    for word in words:
        w = word.lower()
        count = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith("le") and count > 1:
            count -= 1
        total_syllables += max(count, 1)

    asl = num_words / num_sentences
    asw = total_syllables / num_words
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return float(max(min(score, 100.0), 0.0))


class StylometricEngine:
    """Extracts 12 fine-grained stylometric measures from review text."""

    def __init__(self, generic_lexicon: Set[str] = None):
        self.generic_lexicon = generic_lexicon or DEFAULT_GENERIC_LEXICON

    def extract_features(self, text: str) -> Dict[str, float]:
        """Computes the 12 stylometric features for a given text."""
        if not text or not isinstance(text, str):
            return {
                "n_tokens": 0.0,
                "avg_sent_len": 0.0,
                "ttr": 0.0,
                "rep_trigram": 0.0,
                "max_word_run": 0.0,
                "generic_ratio": 0.0,
                "specificity": 0.0,
                "excl_ratio": 0.0,
                "caps_ratio": 0.0,
                "first_person": 0.0,
                "superlative": 0.0,
                "readability": 50.0,
            }

        # Tokenization & sentences
        char_len = max(len(text), 1)
        tokens = text.split()
        num_tokens = max(len(tokens), 1)

        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        num_sentences = max(len(sentences), 1)

        # 1. n_tokens
        n_tokens = float(num_tokens)

        # 2. avg_sent_len
        avg_sent_len = float(num_tokens / num_sentences)

        # 3. Type-Token Ratio (TTR)
        lower_tokens = [t.lower() for t in tokens]
        unique_tokens = len(set(lower_tokens))
        ttr = float(unique_tokens / num_tokens)

        # 4. Repeated-trigram ratio: rho = 1 - (unique trigrams / total trigrams)
        if len(tokens) >= 3:
            trigrams = [tuple(lower_tokens[i : i + 3]) for i in range(len(tokens) - 2)]
            rep_trigram = float(1.0 - (len(set(trigrams)) / len(trigrams)))
        else:
            rep_trigram = 0.0

        # 5. Longest repeated-word run
        max_run = 1
        current_run = 1
        for i in range(1, len(tokens)):
            if lower_tokens[i] == lower_tokens[i - 1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 1
        max_word_run = float(max_run)

        # 6. Generic phrasing ratio
        lower_text = text.lower()
        matched_tokens = 0
        for phrase in self.generic_lexicon:
            count = len(re.findall(re.escape(phrase), lower_text))
            if count > 0:
                matched_tokens += count * len(phrase.split())
        generic_ratio = float(min(matched_tokens / num_tokens, 1.0))

        # 7. Concrete specificity: (digits + entities/models) / L
        digit_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
        spec_patterns = sum(1 for t in tokens if re.search(r"\b([A-Z0-9]{2,}\b|\$\d+|\d+mm|\d+hz|\d+gb)", t))
        specificity = float(min((digit_tokens + spec_patterns) / num_tokens, 1.0))

        # 8. Exclamation mark ratio
        excl_ratio = float(text.count("!") / char_len)

        # 9. Capital letters ratio
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = float(caps_count / char_len)

        # 10. First-person pronoun ratio
        fp_count = sum(1 for t in lower_tokens if t in FIRST_PERSON_PRONOUNS)
        first_person = float(fp_count / num_tokens)

        # 11. Superlative ratio
        sup_count = sum(1 for t in lower_tokens if t in SUPERLATIVE_WORDS or (t.endswith("est") and len(t) > 4))
        superlative = float(sup_count / num_tokens)

        # 12. Readability score
        readability = compute_flesch_reading_ease(text)

        return {
            "n_tokens": n_tokens,
            "avg_sent_len": avg_sent_len,
            "ttr": ttr,
            "rep_trigram": rep_trigram,
            "max_word_run": max_word_run,
            "generic_ratio": generic_ratio,
            "specificity": specificity,
            "excl_ratio": excl_ratio,
            "caps_ratio": caps_ratio,
            "first_person": first_person,
            "superlative": superlative,
            "readability": readability,
        }
