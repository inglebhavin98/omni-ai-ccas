"""Presidio detection for the entity classes regex cannot settle.

Names, places and organisations have no lexical shape, so they need a model. The model
is local and ONNX-backed (CLAUDE.md Rule 5: local inference only) and is consulted after
regex, not instead of it -- a model call on every utterance does not fit the 3 ms slice.

If the analyzer cannot be constructed, ``available`` is False and the pipeline reports
``UNVERIFIED``. It never degrades to "found nothing".
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any, Protocol

from ccas.redaction.engine import DetectedSpan, RedactionEngine
from ccas.schemas.pii import PiiEntityType

__all__ = ["ALLOW_LIST", "ENTITY_MAP", "Analyzer", "PresidioEngine"]

#: Presidio's labels -> our domain-agnostic taxonomy. Anything unmapped is ignored
#: rather than guessed at: a mislabelled entity still gets redacted, but the report
#: would then lie about what it found.
ENTITY_MAP: dict[str, PiiEntityType] = {
    "PERSON": PiiEntityType.PERSON,
    "LOCATION": PiiEntityType.LOCATION,
    "GPE": PiiEntityType.LOCATION,
    "ORGANIZATION": PiiEntityType.ORG,
    "ORG": PiiEntityType.ORG,
    "ADDRESS": PiiEntityType.ADDRESS,
    "STREET_ADDRESS": PiiEntityType.ADDRESS,
    "MEDICAL_LICENSE": PiiEntityType.ACCOUNT_REF,
    "US_DRIVER_LICENSE": PiiEntityType.ACCOUNT_REF,
    "US_PASSPORT": PiiEntityType.NATIONAL_ID,
}

#: Support-protocol vocabulary a general-purpose NER model reliably mislabels.
#: Observed with en_core_web_sm: "SSN" and "CVV" as ORGANIZATION, "Email" as PERSON.
#: Redacting these would make transcripts unreadable for the human agents who inherit
#: them, and would hide *which* identifier the caller was asked for.
ALLOW_LIST: frozenset[str] = frozenset(
    {
        "ssn",
        "cvv",
        "cvc",
        "iban",
        "pin",
        "otp",
        "dob",
        "id",
        "ref",
        "url",
        "email",
        "e-mail",
        "phone",
        "mobile",
        "cell",
        "fax",
        "zip",
        "zipcode",
        "postcode",
        "card",
        "account",
        "number",
        "code",
        "pw",
        "password",
        "am",
        "pm",
        "ok",
        "okay",
        "hi",
        "hello",
        "thanks",
        "yes",
        "no",
        "sir",
        "madam",
        "ma'am",
        "please",
        "sorry",
        "bye",
    }
)

#: spaCy NER labels this module has no use for. Left in, they produce a warning per
#: call ("not mapped to a Presidio entity, but keeping anyway"), which would bury real
#: signal in an ingest log -- and none of them are identifiers.
IGNORED_NER_LABELS: tuple[str, ...] = (
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "PRODUCT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "EVENT",
    "FAC",
    "TIME",
)

#: Presidio labels we ask for, derived from the mapping above.
_REVERSE: dict[PiiEntityType, tuple[str, ...]] = {}
for _label, _entity in ENTITY_MAP.items():
    _REVERSE.setdefault(_entity, ())
    _REVERSE[_entity] = (*_REVERSE[_entity], _label)


class Analyzer(Protocol):
    """The slice of Presidio's ``AnalyzerEngine`` this module uses.

    ``entities`` is typed as ``list`` rather than ``Sequence`` to stay structurally
    compatible with the real ``AnalyzerEngine`` signature.
    """

    def analyze(
        self, text: str, language: str, entities: list[str] | None = ...
    ) -> Sequence[Any]: ...


class PresidioEngine(RedactionEngine):
    name = "presidio_onnx"

    def __init__(
        self,
        entities: tuple[PiiEntityType, ...] = (),
        min_score: float = 0.6,
        language: str = "en",
        analyzer: Analyzer | None = None,
        spacy_model: str = "en_core_web_lg",
    ) -> None:
        self._requested = entities or tuple(_REVERSE)
        self._min_score = min_score
        self._language = language
        self._spacy_model = spacy_model
        self._analyzer = analyzer
        self._labels: tuple[str, ...] | None = None
        self._load_error: str | None = None
        if analyzer is None:
            self._analyzer, self._load_error = _try_build_analyzer(spacy_model, language)

    @property
    def available(self) -> bool:
        return self._analyzer is not None

    @property
    def load_error(self) -> str | None:
        """Why the engine is unavailable. Surfaced in the report's residual_patterns."""
        return self._load_error

    @property
    def handled_entities(self) -> frozenset[PiiEntityType]:
        return frozenset(self._requested)

    @property
    def presidio_labels(self) -> tuple[str, ...]:
        """Labels to ask for, narrowed to what this analyzer actually recognises.

        The map carries aliases (GPE, ORG, STREET_ADDRESS) so results are understood
        whatever a build emits, but *requesting* an unsupported label makes Presidio
        warn on every single call -- which would bury real signal in an ingest log.
        """
        if self._labels is None:
            wanted: list[str] = []
            for entity in self._requested:
                wanted.extend(_REVERSE.get(entity, ()))
            self._labels = _narrow_to_supported(
                tuple(sorted(set(wanted))), self._analyzer, self._language
            )
        return self._labels

    def detect(self, text: str) -> tuple[DetectedSpan, ...]:
        if self._analyzer is None:
            return ()
        results = self._analyzer.analyze(
            text=text, language=self._language, entities=list(self.presidio_labels)
        )
        found: list[DetectedSpan] = []
        for result in results:
            entity = ENTITY_MAP.get(str(result.entity_type))
            if entity is None or float(result.score) < self._min_score:
                continue
            if _is_allow_listed(text[int(result.start) : int(result.end)]):
                continue
            found.append(
                DetectedSpan(
                    start=int(result.start),
                    end=int(result.end),
                    entity_type=entity,
                    score=float(result.score),
                    engine="presidio_onnx",
                    pattern_name=None,
                )
            )
        return tuple(found)


def _narrow_to_supported(
    wanted: tuple[str, ...], analyzer: Analyzer | None, language: str
) -> tuple[str, ...]:
    supported_fn = getattr(analyzer, "get_supported_entities", None)
    if supported_fn is None:
        return wanted
    try:
        supported = set(supported_fn(language))
    except Exception:  # a probe failure must not disable detection
        return wanted
    narrowed = tuple(label for label in wanted if label in supported)
    return narrowed or wanted


def _is_allow_listed(surface: str) -> bool:
    """True when every token of a span is protocol vocabulary rather than an entity."""
    tokens = [t for t in surface.casefold().replace("-", " ").split() if t.isalpha() or "'" in t]
    return bool(tokens) and all(t.strip(".,'") in ALLOW_LIST for t in tokens)


def _try_build_analyzer(spacy_model: str, language: str) -> tuple[Analyzer | None, str | None]:
    """Construct the local analyzer, or explain why it could not be built.

    Import failure is expected in a default install -- Presidio lives in the
    ``redaction`` extra -- and must be reported, never swallowed. A missing model is
    likewise reported rather than silently downgraded to a smaller one.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError:
        return None, "presidio_not_installed"

    # Check before handing the name to Presidio: on a miss it attempts a network
    # download and then calls sys.exit(1). A missing model must make this engine
    # unavailable, not terminate the process.
    if importlib.util.find_spec(spacy_model) is None:
        return None, f"spacy_model_missing:{spacy_model}"

    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": spacy_model}],
                "ner_model_configuration": {"labels_to_ignore": list(IGNORED_NER_LABELS)},
            }
        )
        analyzer: Analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=[language]
        )
    except Exception as exc:
        return None, f"presidio_init_failed:{type(exc).__name__}"
    return analyzer, None
