# Documentation

| Document | Answers |
|---|---|
| [`skills.md`](skills.md) | How do I work in this repo? What are the rules, tools and agent capabilities? |
| [`tech-spec.md`](tech-spec.md) | What shape is the data? What are the API contracts and latency budgets? |
| [`design-doc.md`](design-doc.md) | How does it fit together? What flows through it, and why is it built this way? |
| [`adr/`](adr/README.md) | Why was this chosen over the alternative? |
| [`future-scoped-work.md`](future-scoped-work.md) | What was deliberately left out, and what would bring it back? |

`CLAUDE.md` at the repo root is the binding rulebook. These documents explain and
operationalise it; where they disagree, `CLAUDE.md` wins.

## Reading order

New to the project: `design-doc.md` → `skills.md` → `tech-spec.md`.

Picking up a module: `skills.md` §2 (conventions) → `tech-spec.md` for the contracts you
will touch → the relevant ADR.

Reviewing a change: the ADR it cites, then `tech-spec.md` §5 (enforcement summary).
