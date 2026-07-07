"""Transcript store + phrase-occurrence index (§5, M2).

Two jobs:

* **Store** — persist/load the scraped corpus (``data/corpus/transcripts.parquet``) as a
  :class:`~pmlab.data.schemas.CorpusTranscripts` frame; append-dedupe by ``doc_id``.
* **Phrase matcher** — turn a market's ``yes_sub_title`` (the outcome label, e.g.
  ``"Afford / Affordable / Affordability"``, ``"Oil / Gas / Gasoline"``, ``"Inflation (3+ times)"``)
  into a :class:`PhraseSpec` and count its occurrences in a transcript, mirroring Kalshi's
  resolution semantics: ``/`` enumerates accepted alternates, plurals/possessives are accepted
  universally (the ``rules_secondary`` payout clause), and ``(N+ times)`` is a counting threshold.

The matcher is deterministic and the most heavily unit-tested piece in M2 — it is the bridge
between "what the market asked" and "what the speaker actually said" that the base-rate feature
sits on. It is a *feature* input, not the resolution itself, so it errs toward simple, auditable
whole-word matching (case-insensitive) and documents the known soft spots (acronym case,
homographs) rather than trying to reproduce Kalshi's per-market judgement.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pmlab.data.schemas import CORPUS_COLUMNS, CorpusTranscripts, empty_corpus
from pmlab.mentions.resolution import VariantRule, classify_variant

_WS = re.compile(r"\s+")
_COUNT_RE = re.compile(r"\(\s*(\d+)\s*\+?\s*times?\s*\)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

# Speaker aliases → a canonical join key so a registry market's speaker lines up with the corpus
# speaker regardless of how each side labels the same person (series label vs parsed vs scraped).
_SPEAKER_ALIASES: dict[str, str] = {
    "powell": "jerome powell",
    "jerome powell": "jerome powell",
    "chair powell": "jerome powell",
    "the chair of the federal reserve": "jerome powell",
    "the federal reserve chair": "jerome powell",
    "fed chair": "jerome powell",
    "fed": "jerome powell",
}
_TITLE_RE = re.compile(
    r"^(vice )?(president|chair(man)?(\s+for\s+supervision)?|chairwoman|secretary|senator|"
    r"governor|representative|the honorable|mr|mrs|ms|dr|sir|prime minister|pm|king|queen)\.?\s+",
    re.IGNORECASE,
)
_INITIAL_RE = re.compile(r"\b[a-z]\.")  # middle initial, e.g. the "h." in "jerome h. powell"


def canonical_speaker(name: str | None) -> str:
    """Normalise a speaker label to a stable join key (lowercased, titles + middle initials
    stripped, aliased).

    Alignment matters: the corpus base rate only fires when a registry market's speaker maps to
    the same key as the scraped transcripts' speaker (e.g. a Fed byline ``"Chair Jerome H.
    Powell"`` and a Kalshi label ``"Jerome Powell"`` must collapse to one key). Unknown names fall
    through to their normalised form, so non-corpus speakers still get a consistent, if unmatched,
    key."""
    s = _WS.sub(" ", (name or "").strip().lower())
    if not s:
        return ""
    if s in _SPEAKER_ALIASES:
        return _SPEAKER_ALIASES[s]
    prev = None
    while prev != s:  # peel stacked titles ("vice chair for supervision …")
        prev = s
        s = _TITLE_RE.sub("", s).strip()
    s = _WS.sub(" ", _INITIAL_RE.sub("", s)).strip(" .")
    return _SPEAKER_ALIASES.get(s, s)


def normalize_key_text(text: str) -> str:
    """Lowercased, alphanumeric-only, single-spaced — for building stable phrase keys."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def _term_regex(term: str) -> str | None:
    """Whole-word, case-insensitive pattern for one alternate, tolerating a trailing
    plural/possessive on the final token (``Billionaire`` → ``billionaires``; ``World Cup`` →
    ``World Cups``). Word edges use letter lookarounds so ``ice`` doesn't fire inside ``nice`` and
    a trailing comma/apostrophe still closes the match."""
    words = [re.escape(w) for w in term.strip().split()]
    if not words:
        return None
    words[-1] = words[-1] + r"(?:'s|’s|es|s)?"
    core = r"\s+".join(words)
    return r"(?<![A-Za-z])" + core + r"(?![A-Za-z])"


@dataclass
class PhraseSpec:
    """A parsed outcome label ready to count in transcript text."""

    raw: str
    alternates: list[str]
    variant: VariantRule
    min_count: int
    phrase_key: str
    pattern: re.Pattern[str] | None = field(default=None, compare=False, repr=False)

    @property
    def matchable(self) -> bool:
        """False for empty/``fuzzy`` labels with no reliable term to search for."""
        return self.pattern is not None and bool(self.alternates)


def parse_phrase_spec(yes_sub_title: str | None) -> PhraseSpec:
    """Parse a Kalshi outcome label into an alternates set + counting threshold + match regex.

    ``"Inflation (3+ times)"`` → alternates ``["Inflation"]``, ``min_count=3``. ``"Oil / Gas /
    Gasoline"`` → three alternates, ``min_count=1``. Empty label → an unmatchable ``fuzzy`` spec."""
    raw = (yes_sub_title or "").strip()
    variant = classify_variant(raw)
    min_count = 1
    m = _COUNT_RE.search(raw)
    core = raw
    if m:
        min_count = max(1, int(m.group(1)))
        core = _COUNT_RE.sub("", raw).strip()

    alternates = [p.strip() for p in core.split("/") if p.strip()] if core else []
    regexes = [r for r in (_term_regex(a) for a in alternates) if r]
    pattern = re.compile("|".join(regexes), re.IGNORECASE) if regexes else None

    norm_alts = sorted({normalize_key_text(a) for a in alternates if normalize_key_text(a)})
    key = "|".join(norm_alts)
    if min_count > 1 and key:
        key = f"{key}|>={min_count}"
    return PhraseSpec(
        raw=raw,
        alternates=alternates,
        variant=variant,
        min_count=min_count,
        phrase_key=key,
        pattern=pattern,
    )


def phrase_key(yes_sub_title: str | None) -> str:
    """Stable grouping key for phrase recurrence across occasions (alternates + threshold)."""
    return parse_phrase_spec(yes_sub_title).phrase_key


def count_occurrences(text: str | None, spec: PhraseSpec) -> int:
    """Total whole-word occurrences of any alternate in ``text`` (0 for an unmatchable spec)."""
    if spec.pattern is None or not text:
        return 0
    return len(spec.pattern.findall(text))


def phrase_occurs(text: str | None, spec: PhraseSpec) -> bool:
    """True iff the phrase occurs at least ``min_count`` times — the market's Yes condition."""
    if not spec.matchable:
        return False
    return count_occurrences(text, spec) >= spec.min_count


def word_count(text: str | None) -> int:
    return len((text or "").split())


def make_doc_id(source: str, url: str, occasion: str = "") -> str:
    """Stable id for a scraped occasion (dedupe key across re-fetches)."""
    payload = f"{source}|{url}|{occasion}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


# --- store I/O -----------------------------------------------------------------------------------


def to_corpus_frame(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Validate raw transcript dicts into a :class:`CorpusTranscripts` frame."""
    rows = list(records)
    df = pd.DataFrame(rows, columns=CORPUS_COLUMNS) if rows else empty_corpus()
    return CorpusTranscripts.validate(df)


def persist_corpus(corpus: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    CorpusTranscripts.validate(corpus)
    corpus.to_parquet(path, index=False)
    return path


def load_corpus(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_corpus()
    return CorpusTranscripts.validate(pd.read_parquet(path))


def merge_corpus(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append new transcripts, keeping the latest row per ``doc_id`` (idempotent re-fetch)."""
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset="doc_id", keep="last").reset_index(drop=True)
    return CorpusTranscripts.validate(combined)


# --- occurrence index ----------------------------------------------------------------------------

OCCURRENCE_COLUMNS = ["speaker_key", "format", "phrase_key", "date_ts", "occurred", "count"]


SpecGroups = Mapping[tuple[str, str], Iterable[PhraseSpec]]


def build_occurrences(corpus: pd.DataFrame, specs_by_group: SpecGroups) -> pd.DataFrame:
    """Long occurrence frame: for each ``(speaker_key, format, phrase_key)`` × corpus occasion,
    whether the phrase occurred and its raw count.

    ``specs_by_group`` maps a ``(canonical speaker, format)`` to the phrase specs actually asked
    about that speaker (derived from the registry), so a phrase is only ever matched against its
    own speaker's transcripts. Computed once per distinct ``phrase_key``; the base-rate feature
    then reduces to a leakage-gated groupby rather than re-running regexes per target market."""
    if corpus.empty or not specs_by_group:
        return pd.DataFrame(columns=OCCURRENCE_COLUMNS)

    corp = corpus.copy()
    corp["speaker_key"] = corp["speaker"].map(canonical_speaker)
    rows: list[dict[str, object]] = []
    for (speaker_key, fmt), group in corp.groupby(["speaker_key", "format"], sort=False):
        uniq: dict[str, PhraseSpec] = {}
        for spec in specs_by_group.get((str(speaker_key), str(fmt)), ()):
            if spec.matchable and spec.phrase_key not in uniq:
                uniq[spec.phrase_key] = spec
        if not uniq:
            continue
        texts = [
            (int(d), str(t) if t is not None else "")
            for d, t in zip(group["date_ts"], group["text"], strict=True)
        ]
        for key, spec in uniq.items():
            for date_ts, text in texts:
                cnt = count_occurrences(text, spec)
                rows.append(
                    {
                        "speaker_key": speaker_key,
                        "format": fmt,
                        "phrase_key": key,
                        "date_ts": date_ts,
                        "occurred": cnt >= spec.min_count,
                        "count": cnt,
                    }
                )
    return pd.DataFrame(rows, columns=OCCURRENCE_COLUMNS)
