"""Turn clusters into a named L1/L2/L3 hierarchy (Module 3).

Clustering finds *that* a group of utterances belongs together; it cannot say what the
group is. That is the one step in mining where a language model genuinely helps, and it
is run offline against structured output so the result is validated, not parsed.

The prompt is strictly domain-agnostic (CLAUDE.md Rule 1): the model is told the pack's
declared tool names and nothing else about the vertical, and it invents the vocabulary
from the utterances themselves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from ccas.llm.base import LLMProvider
from ccas.llm.prompt import authored, compose
from ccas.schemas.common import Frozen, JsonValue, RiskTier, Slug
from ccas.schemas.llm import LLMRequest, Message, ModelBinding
from ccas.schemas.pii import RedactedText
from ccas.schemas.taxonomy import SlotType

__all__ = ["LABEL_SCHEMA", "SYSTEM_PROMPT", "ClusterBrief", "ClusterLabel", "IntentLabeler"]

SYSTEM_PROMPT = """\
You are building a customer-care intent taxonomy from clustered caller utterances.

Every utterance has already been redacted: tokens like [PERSON_1] or [ACCOUNT_REF_1] \
stand in for removed identifiers. Treat them as opaque placeholders.

For the cluster you are given, produce a three-level intent path:
  L1  the broad area of the business the caller is contacting about
  L2  the specific intent within that area
  L3  the concrete action or question being asked

Rules:
- Derive the vocabulary from the utterances. Do not import terminology from an industry \
you assume this is; if the utterances do not say it, do not name it.
- Slugs are lowercase, dot-free, and use underscores between words.
- The L1 slug must be broad enough that other clusters can share it. Prefer an existing \
one from the list of already-assigned L1 slugs when it genuinely fits.
- Slots are the pieces of information an agent would have to collect to act. Only list \
ones the utterances actually imply.
- required_tools may only name tools from the provided list. If none fit, return [].
- If you list a tool, you MUST declare a slot for each of its required arguments, using
  exactly the argument name shown. The platform collects slots and passes them to the
  tool by name, so a missing one makes the call fail.
- complexity is how hard this is to automate end to end: a single lookup is low, \
multi-step judgement or a policy decision is high.
- feasibility is how confidently an automated agent could resolve it without a human.
- Set risk_tier to "regulated" only when acting would require a licensed or legally \
accountable person.
- Set is_intent to false when the cluster is conversational filler rather than a reason \
for contact: greetings, closings, hold acknowledgements, thanks, confirmations. These \
are the largest clusters in any real corpus and they are not intents.
- Set is_intent to false when the cluster is the *agent* speaking rather than the caller: \
scripted openings, disclosures, probing questions, hold messages, or anything offering or \
explaining a product. Only a caller's reason for contact is an intent.
"""

USER_TEMPLATE = """\
Cluster {cluster_id} contains {size} utterances ({share} of the corpus).

Already-assigned L1 slugs (reuse one if it fits):
{known_l1}

Available tools:
{tools}

Representative utterances:
{exemplars}
"""

LABEL_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "l1_slug": {"type": "string"},
        "l1_label": {"type": "string"},
        "l1_description": {"type": "string"},
        "l2_slug": {"type": "string"},
        "l2_label": {"type": "string"},
        "l2_description": {"type": "string"},
        "l3_slug": {"type": "string"},
        "l3_label": {"type": "string"},
        "l3_description": {"type": "string"},
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "slot_type": {
                        "type": "string",
                        "enum": [t.value for t in SlotType],
                    },
                    "required": {"type": "boolean"},
                    "elicitation_prompt": {"type": "string"},
                    "pii_sensitive": {"type": "boolean"},
                    "dtmf_capturable": {"type": "boolean"},
                },
                "required": [
                    "name",
                    "slot_type",
                    "required",
                    "elicitation_prompt",
                    "pii_sensitive",
                    "dtmf_capturable",
                ],
                "additionalProperties": False,
            },
        },
        "required_tools": {"type": "array", "items": {"type": "string"}},
        "is_intent": {"type": "boolean"},
        "risk_tier": {"type": "string", "enum": [t.value for t in RiskTier]},
        "complexity": {"type": "number"},
        "feasibility": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": [
        "l1_slug",
        "l1_label",
        "l1_description",
        "l2_slug",
        "l2_label",
        "l2_description",
        "l3_slug",
        "l3_label",
        "l3_description",
        "slots",
        "required_tools",
        "is_intent",
        "risk_tier",
        "complexity",
        "feasibility",
        "rationale",
    ],
    "additionalProperties": False,
}


class ClusterBrief(Frozen):
    """Everything the labeler is allowed to know about one cluster."""

    cluster_id: int
    size: int = Field(ge=1)
    share: float = Field(ge=0.0, le=1.0)
    exemplars: tuple[RedactedText, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _check_exemplars_are_clean(self) -> ClusterBrief:
        if any(not e.egress_permitted for e in self.exemplars):
            raise ValueError(f"cluster {self.cluster_id}: an exemplar is not egress-permitted")
        return self


class MinedSlot(Frozen):
    name: str
    slot_type: SlotType
    required: bool
    elicitation_prompt: str
    pii_sensitive: bool
    dtmf_capturable: bool


class ClusterLabel(Frozen):
    """A validated label for one cluster."""

    cluster_id: int
    l1_slug: Slug
    l1_label: str
    l1_description: str
    l2_slug: Slug
    l2_label: str
    l2_description: str
    l3_slug: Slug
    l3_label: str
    l3_description: str
    slots: tuple[MinedSlot, ...] = ()
    required_tools: tuple[Slug, ...] = ()
    is_intent: bool = True
    """False for greetings, closings and confirmations. Excluded from the taxonomy --
    a stock phrase repeated in every call would otherwise be its highest-volume node."""

    risk_tier: RiskTier = RiskTier.LOW
    complexity: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    rationale: str

    @model_validator(mode="after")
    def _check_slugs_are_single_level(self) -> ClusterLabel:
        """Each level contributes one dot-free segment.

        A model that answers "billing.dispute" for l2_slug would silently produce a
        depth-4 id that fails the taxonomy's level check much later.
        """
        for field_name in ("l1_slug", "l2_slug", "l3_slug"):
            value = getattr(self, field_name)
            if "." in value:
                raise ValueError(f"{field_name}={value!r} must be a single segment without dots")
        return self

    @property
    def l1_id(self) -> str:
        return self.l1_slug

    @property
    def l2_id(self) -> str:
        return f"{self.l1_slug}.{self.l2_slug}"

    @property
    def l3_id(self) -> str:
        return f"{self.l1_slug}.{self.l2_slug}.{self.l3_slug}"


class IntentLabeler:
    """Names clusters via an ``LLMProvider``. Offline; never on the call path."""

    def __init__(
        self,
        provider: LLMProvider,
        binding: ModelBinding,
        tool_names: Sequence[str] = (),
        concurrency: int = 4,
        tool_arguments: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._provider = provider
        self._binding = binding
        self._tool_names = tuple(tool_names)
        self._tool_arguments = dict(tool_arguments or {})
        """Required argument names per tool. Shown to the model so the slots it mines
        line up with what the tool will actually be sent."""
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def model_name(self) -> str:
        """Recorded on the taxonomy: which model named these clusters is provenance."""
        return f"{self._binding.provider.value}:{self._binding.model}"

    def build_request(self, brief: ClusterBrief, known_l1: Sequence[str]) -> LLMRequest:
        exemplars = compose(
            "\n".join(f"- {{e{i}}}" for i in range(len(brief.exemplars))),
            {f"e{i}": text for i, text in enumerate(brief.exemplars)},
        )
        user = compose(
            USER_TEMPLATE,
            {
                "cluster_id": authored(str(brief.cluster_id)),
                "size": authored(str(brief.size)),
                "share": authored(f"{brief.share:.1%}"),
                "known_l1": authored(", ".join(known_l1) or "(none yet)"),
                "tools": authored(self._tool_catalogue()),
                "exemplars": exemplars,
            },
        )
        return LLMRequest(
            binding=self._binding,
            system=authored(SYSTEM_PROMPT),
            messages=(Message(role="user", content=user),),
            response_schema=LABEL_SCHEMA,
        )

    def _tool_catalogue(self) -> str:
        if not self._tool_names:
            return "(none declared)"
        return "\n".join(
            f"- {name}"
            + (
                f" (requires slots: {', '.join(self._tool_arguments[name])})"
                if self._tool_arguments.get(name)
                else ""
            )
            for name in self._tool_names
        )

    async def label(self, brief: ClusterBrief, known_l1: Sequence[str] = ()) -> ClusterLabel:
        async with self._semaphore:
            response = await self._provider.structured(self.build_request(brief, known_l1))
        if response.parsed is None:
            raise ValueError(f"cluster {brief.cluster_id}: labeler returned no parsed output")
        payload = dict(response.parsed)
        payload["cluster_id"] = brief.cluster_id
        # Drop any tool the model invented. The registry is the authority on what
        # exists; a hallucinated name would fail later, at a much worse moment.
        requested = payload.get("required_tools")
        payload["required_tools"] = (
            [str(tool) for tool in requested if str(tool) in self._tool_names]
            if isinstance(requested, list)
            else []
        )
        return ClusterLabel.model_validate(payload)

    async def label_all(self, briefs: Sequence[ClusterBrief]) -> tuple[ClusterLabel, ...]:
        """Label clusters largest-first, feeding assigned L1 slugs forward.

        Order matters: the biggest clusters establish the L1 vocabulary, and later
        clusters are shown it so the hierarchy converges instead of fragmenting into
        one L1 per cluster.
        """
        labels: list[ClusterLabel] = []
        known_l1: list[str] = []
        for brief in sorted(briefs, key=lambda b: -b.size):
            label = await self.label(brief, known_l1)
            labels.append(label)
            if label.l1_slug not in known_l1:
                known_l1.append(label.l1_slug)
        return tuple(labels)
