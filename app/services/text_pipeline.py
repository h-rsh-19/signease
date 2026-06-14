from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import PipelineTraceStep
from app.services.asset_catalog import AssetCatalog, norm_token

STOPWORDS = {
    "am", "are", "is", "was", "were", "be", "being", "been", "have", "has", "had",
    "do", "does", "did", "could", "should", "would", "can", "shall", "will", "may",
    "might", "must", "let", "a", "an", "the", "to", "of", "for", "in", "on", "at",
    "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those",
    "he", "she", "it", "we", "they", "me", "him", "us", "them",
}

QUESTION_WORDS = {"what", "where", "when", "why", "who", "whom", "which", "how"}

VERB_HINTS = {
    "go", "come", "eat", "learn", "study", "work", "play", "help", "see", "talk", "walk",
    "read", "write", "make", "take", "give", "ask", "say", "like", "love", "know", "think",
    "need", "want"
}

SYNONYM_MAP = {
    "dont": "not", "cannot": "not", "thanks": "thank", "tv": "television", "u": "you", "pls": "please", "thx": "thank",
    "hi": "hello", "hey": "hello", "help": "how_can_i_help",
    "coming": "come", "comes": "come", "came": "come",
    "making": "make", "makes": "make", "made": "make",
    "taking": "take", "takes": "take", "took": "take",
    "giving": "give", "gives": "give", "gave": "give",
    "loving": "love", "loves": "love", "loved": "love",
    "saying": "say", "says": "say", "said": "say",
    "asking": "ask", "asks": "ask", "asked": "ask",
    "going": "go", "goes": "go", "went": "go", "gone": "go",
    "eating": "eat", "eats": "eat", "ate": "eat", "eaten": "eat",
    "working": "work", "works": "work", "worked": "work",
    "playing": "play", "plays": "play", "played": "play",
    "helping": "help", "helps": "help", "helped": "help",
    "seeing": "see", "sees": "see", "saw": "see", "seen": "see",
    "talking": "talk", "talks": "talk", "talked": "talk",
    "walking": "walk", "walks": "walk", "walked": "walk",
    "reading": "read", "reads": "read",
    "writing": "write", "writes": "write", "wrote": "write", "written": "write",
    "learning": "learn", "learns": "learn", "learned": "learn", "learnt": "learn",
    "studying": "study", "studies": "study", "studied": "study",
    "knowing": "know", "knows": "know", "knew": "know", "known": "know",
    "thinking": "think", "thinks": "think", "thought": "think",
    "liking": "like", "likes": "like", "liked": "like",
    "needs": "need", "needed": "need", "needing": "need",
    "wants": "want", "wanted": "want", "wanting": "want",
}

# Centralized data-driven structures populated from the catalog.
ASSET_DICT: dict[str, str] = {}
PHRASE_DICT: dict[str, str] = {}
ALPHABET_DICT: dict[str, str] = {}
PHRASE_COMPONENTS: dict[str, set[str]] = {
    "how_can_i_help": {"how", "can", "i", "help"},
}
SUBJECTS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
}
VERBS = set(VERB_HINTS)


@dataclass
class TranslationResult:
    input_text: str
    normalized_text: str
    tokens: list[str]
    token_confidence: list[dict[str, object]]
    unknown_tokens: list[str]
    trace: list[PipelineTraceStep]


class TextPipelineService:
    def __init__(self, catalog: AssetCatalog) -> None:
        self.catalog = catalog
        self._sync_data_structures()

    def _sync_data_structures(self) -> None:
        global ASSET_DICT, PHRASE_DICT, ALPHABET_DICT
        ASSET_DICT = {
            **self.catalog.human_videos,
            **self.catalog.animated_videos,
            **self.catalog.sigml_files,
        }
        PHRASE_DICT = {
            key: value
            for key, value in ASSET_DICT.items()
            if "_" in key
        }
        ALPHABET_DICT = {
            ch: ASSET_DICT[ch]
            for ch in "abcdefghijklmnopqrstuvwxyz0123456789"
            if ch in ASSET_DICT
        }

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    def _synonym_or_self(self, token: str) -> str:
        return SYNONYM_MAP.get(token, token)

    def lemmatize(self, tokens: list[str]) -> list[str]:
        return [self._synonym_or_self(tok) for tok in tokens]

    def phrase_match(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        result: list[str] = []
        i = 0
        total = len(tokens)
        while i < total:
            match_found = False
            # Greedy n-gram matching (3 -> 1)
            for n in range(min(3, total - i), 0, -1):
                phrase = " ".join(tokens[i:i + n])
                key = norm_token(phrase)
                if key in PHRASE_DICT:
                    result.append(key)
                    i += n
                    match_found = True
                    break

            if not match_found:
                result.append(tokens[i])
                i += 1

        return result

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        return [tok for tok in tokens if tok not in STOPWORDS]

    def reorder_sov(self, words: list[str]) -> list[str]:
        subjects: list[str] = []
        objects: list[str] = []
        verbs: list[str] = []
        question_tail: list[str] = []

        for w in words:
            if w in QUESTION_WORDS:
                question_tail.append(w)
            elif w in SUBJECTS:
                subjects.append(w)
            elif w in VERBS:
                verbs.append(w)
            else:
                objects.append(w)

        return subjects + objects + verbs + question_tail

    def resolve_overlaps(self, words: list[str]) -> tuple[list[str], list[str]]:
        if not words:
            return [], []

        active_components: set[str] = set()
        for token in words:
            key = norm_token(token)
            if key in PHRASE_COMPONENTS:
                active_components |= PHRASE_COMPONENTS[key]

        resolved: list[str] = []
        dropped: list[str] = []
        for token in words:
            key = norm_token(token)
            if key in PHRASE_COMPONENTS:
                resolved.append(token)
                continue
            if key in active_components:
                dropped.append(key)
                continue
            resolved.append(token)
        return resolved, sorted(set(dropped))

    def map_assets(self, words: list[str]) -> tuple[list[str], list[str]]:
        mapped_tokens: list[str] = []
        unknown_tokens: list[str] = []
        for w in words:
            key = norm_token(w)
            # O(1) lookup using dictionary
            if key in ASSET_DICT:
                mapped_tokens.append(key)
            else:
                unknown_tokens.append(key)
        return mapped_tokens, unknown_tokens

    def fallback(self, word: str) -> list[str]:
        # Explicit letter-level fallback
        return [ALPHABET_DICT[c] for c in word if c in ALPHABET_DICT]

    @staticmethod
    def score_token(token: str, *, source: str, is_reordered: bool = False) -> dict[str, object]:
        if source == "phrase":
            confidence = 0.95
        elif source == "direct":
            confidence = 0.85
        elif source == "fallback":
            confidence = 0.55
        else:
            confidence = 0.75 if is_reordered else 0.8
        return {"token": token, "confidence": round(confidence, 2), "source": source}

    def process_text(self, text: str) -> TranslationResult:
        trace: list[PipelineTraceStep] = []

        normalized = self.normalize(text)
        trace.append(PipelineTraceStep(stage="normalize", input=text, output=normalized))

        tokens = self.tokenize(normalized)
        trace.append(PipelineTraceStep(stage="tokenize", input=normalized, output=tokens))

        lemmas = self.lemmatize(tokens)
        if lemmas != tokens:
            trace.append(PipelineTraceStep(stage="lemmatize", input=tokens, output=lemmas))

        phrase_tokens = self.phrase_match(lemmas)
        if phrase_tokens != lemmas:
            trace.append(PipelineTraceStep(stage="phrase_match", input=lemmas, output=phrase_tokens))

        overlap_resolved, dropped_tokens = self.resolve_overlaps(phrase_tokens)
        if overlap_resolved != phrase_tokens:
            trace.append(
                PipelineTraceStep(
                    stage="overlap_resolve",
                    input={"before": phrase_tokens},
                    output={"after": overlap_resolved, "dropped": dropped_tokens},
                )
            )

        filtered = self.remove_stopwords(overlap_resolved)
        if filtered != overlap_resolved:
            trace.append(PipelineTraceStep(stage="remove_stopwords", input=overlap_resolved, output=filtered))

        reordered = self.reorder_sov(filtered)
        if reordered != filtered:
            trace.append(PipelineTraceStep(stage="reorder_sov", input=filtered, output=reordered))

        mapped_tokens, unknown = self.map_assets(reordered)
        trace.append(
            PipelineTraceStep(
                stage="map_assets",
                input=reordered,
                output={"mapped": mapped_tokens, "unknown": unknown},
            )
        )

        fallback_map = {tok: self.fallback(tok) for tok in unknown}
        trace.append(PipelineTraceStep(stage="fallback", input=unknown, output=fallback_map))

        token_confidence: list[dict[str, object]] = []
        reordered_set = set(reordered)
        for tok in reordered:
            key = norm_token(tok)
            if key in PHRASE_COMPONENTS:
                token_confidence.append(self.score_token(key, source="phrase"))
            elif key in ASSET_DICT:
                token_confidence.append(
                    self.score_token(
                        key,
                        source="direct",
                        is_reordered=(reordered != filtered and key in reordered_set),
                    )
                )
            elif fallback_map.get(key):
                token_confidence.append(self.score_token(key, source="fallback"))
            else:
                token_confidence.append(self.score_token(key, source="unknown"))

        return TranslationResult(
            input_text=text,
            normalized_text=normalized,
            tokens=[norm_token(tok) for tok in reordered if tok],
            token_confidence=token_confidence,
            unknown_tokens=sorted(set(norm_token(tok) for tok in unknown)),
            trace=trace,
        )

    def translate(self, text: str) -> TranslationResult:
        # Compatibility wrapper for existing route usage.
        return self.process_text(text)

