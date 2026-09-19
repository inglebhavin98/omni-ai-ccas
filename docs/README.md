# Documentation

| Document | Answers |
|---|---|
| [`how-it-works.md`](how-it-works.md) | **Start here.** What is this, what does it do, how does it work — in plain language, no expertise assumed |
| [`skills.md`](skills.md) | How do I work in this repo? What are the rules, tools and agent capabilities? |
| [`tech-spec.md`](tech-spec.md) | What shape is the data? What are the API contracts and latency budgets? |
| [`design-doc.md`](design-doc.md) | How does it fit together? What flows through it, and why is it built this way? |
| [`adr/`](adr/README.md) | Why was this chosen over the alternative? |
| [`future-scoped-work.md`](future-scoped-work.md) | What was deliberately left out, and what would bring it back? |
| [`resume-here.md`](resume-here.md) | Point-in-time: what to do first next session. Not maintained — delete when stale. |

`CLAUDE.md` at the repo root is the binding rulebook. These documents explain and
operationalise it; where they disagree, `CLAUDE.md` wins.

## Reading order

New to the project, or not an engineer: `how-it-works.md`, and stop there if that is
all you need.

New engineer: `how-it-works.md` → `design-doc.md` → `skills.md` → `tech-spec.md`.

Picking up a module: `skills.md` §2 (conventions) → `tech-spec.md` for the contracts you
will touch → the relevant ADR.

Reviewing a change: the ADR it cites, then `tech-spec.md` §5 (enforcement summary).
