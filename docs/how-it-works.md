# How it works

A plain-language guide to what this project is, what it actually does, and how.

No prior knowledge assumed. Every term of art is explained the first time it appears, and
there is a [glossary](#glossary) at the end. If you want the precise data shapes read
[`tech-spec.md`](tech-spec.md); if you want to know *why* a choice was made read the
[ADRs](adr/README.md). This document is the one that explains the thing itself.

---

## 1. What problem this solves

When you phone a company about a late parcel, you usually meet a machine first:

> *"Press 1 for orders, press 2 for billing…"*

That machine is an **IVR** — Interactive Voice Response. Bigger systems around it are
called **CCaaS** — Contact Centre as a Service. Genesys and Cisco are the two big vendors.

These systems are old-fashioned in a specific way: **someone had to draw the menu in
advance.** Every question a caller might ask has to have been anticipated, given a number,
and wired to an outcome. Anything unanticipated falls through to "let me put you through to
an agent", which is the expensive outcome the menu existed to avoid.

`omni-ai-ccas` replaces that with a system that **listens to what the caller actually
says**, works out what they want, does it, and hands over to a human when — and only
when — it genuinely cannot.

Two things make it unusual:

**It has no idea what industry it is in.** The code in `src/ccas/` contains no mention of
orders, refunds, prescriptions or policies. Those words live in *data files* called
**domain packs**. Three exist today — retail, healthcare and insurance. Load the retail
pack and it handles deliveries and returns; load the healthcare one and it becomes a
health plan's member line, routing between Member Services and a licensed clinical queue
that only verified callers may reach. Same code, different data file.

**It is built so that a privacy leak is impossible rather than unlikely.** More on that in
[§4](#4-the-four-rules-that-shape-everything), because it is the most interesting thing
here.

---

## 2. Follow one call, end to end

This is the whole system in one story. A caller says:

> *"Hi, I need to return the boots I ordered, my card 4242 4242 4242 4242 got charged
> twice."*

### Step 1 — Hearing the words

Audio arrives and a **VAD** (Voice Activity Detection — a small model that spots when
someone stops talking) decides the caller has finished a sentence. **STT**
(Speech-To-Text) turns the sound into words.

If the caller interrupts the system mid-sentence, that is **barge-in**: playback stops,
the half-spoken reply is cancelled, and the system listens instead. Handled, because
people interrupt machines constantly.

### Step 2 — Removing the private bits, immediately

Before *anything* else touches those words, they pass through the redactor:

```
in   my card 4242 4242 4242 4242 got charged twice
out  my card [PAYMENT_CARD_1] got charged twice
```

The card number is replaced with a **placeholder**. The original is kept in a vault in
memory that is deliberately impossible to accidentally save or log.

This happens **first**, before the AI sees the sentence, before anything is written to a
log, before any record is stored. Everything downstream works on the redacted version.

### Step 3 — Working out what they want

The redacted sentence goes to an **LLM** (Large Language Model — ChatGPT-style AI) whose
only job is to pick one label from a fixed list:

> `order.track_order` · `refund.get_refund` · `payment.payment_issue` · …27 in total

This is **intent routing**. Note what the AI is *not* allowed to do: it cannot invent a
new category, write a database query, or decide to call an external service. It picks from
a list. That constraint is deliberate and is covered in [§4](#rule-4--the-ai-picks-from-a-list-it-never-improvises).

Measured accuracy: **88.5% exact, 100% category** — see [§6](#6-what-the-numbers-actually-say)
for what those mean and why the second number matters more than it looks.

### Step 4 — Collecting what is missing

To look up an order you need an order number. The system knows which pieces of
information — **slots** — each intent requires, and asks for whatever is missing. On a
phone line the caller can also key digits instead of speaking, which helps enormously with
noisy lines and accents.

A keypad answer to a private question gets replaced wholesale. Six bare digits match no
pattern and would otherwise sail straight through, but the system asked for a reference
and the caller gave one, so it is treated as private **because of the context**, not
because of its shape.

### Step 5 — Actually doing something

Now it calls a real backend — a **tool**, in this system's vocabulary. Nine are available
on the retail pack: four shared across every industry, and five the pack adds itself
(`get_order_status`, `start_return`, and so on). A pack may add tools; it may not quietly
replace a shared one.

Two things happen at this boundary that are easy to miss:

- The **real card number is put back** at the last possible instant, because the
  company's own order system already holds the caller's card and looking it up is the
  entire point of the call. The placeholder goes back immediately afterwards.
- Whatever the backend returns is **redacted on the way in** before the AI is allowed to
  read it. A system of record will happily hand back a full name and address; that does
  not mean an AI model should see them.

### Step 6 — Saying something back

The reply is composed and spoken. It has to be **grounded**: every factual claim about the
caller's account must come from a tool result. If the data is missing, the system says so
and escalates. It is not permitted to produce a plausible-sounding guess, which is the
failure mode that makes AI unusable for customer service.

### Step 7 — Handing to a human, properly

If the system escalates — low confidence, an angry caller, a failing backend, a regulated
topic, too many turns, or the caller simply asking — it builds a **handoff**: a package
containing the caller's journey, what was collected, what was tried, a summary, and the
redacted transcript.

That package is checked again on the way out, pushed to the **CRM** (Customer Relationship
Management — the system a human agent has open), and the agent picks up a call they
already understand. **The caller does not repeat themselves.** That is the single most
hated thing about phone support, and removing it is most of the value here.

---

## 3. The parts

The system is six modules over a shared foundation. Each is independently testable and
has a **gate test** — one test file that proves the module works as a whole, not just in
pieces.

| # | What it is called | What it does, plainly |
|---|---|---|
| **M1** | Ingestion | Reads call transcripts from various file formats, redacts them, turns them into one standard record |
| **M2** | Redaction | Finds and replaces private information. A shared library, used by everything |
| **M3** | Mining | Reads thousands of past calls and *discovers* what people actually call about, rather than someone guessing the menu |
| **M4** | Orchestration | The decision-maker: routes, asks, calls tools, replies, escalates |
| **M5** | Voice | Microphone to speaker: the real-time call machinery |
| **M6** | Copilot & evals | The handoff to a human, and the machinery that measures whether any of this works |

Supporting these: **contracts** (the data shapes everything agrees on), **LLM bindings**
(so no AI vendor is hard-wired in), **config** (loads domain packs), and
**observability** (structured logs and an audit trail).

### M3 deserves a longer explanation

This is the least obvious part. Traditional IVR menus are written by someone in a meeting
guessing at what callers want. M3 does it from evidence:

1. Take thousands of real call transcripts (already redacted).
2. Turn each sentence into an **embedding** — a list of numbers positioning it in a space
   where similar meanings sit close together, so *"where's my parcel"* and *"delivery
   hasn't arrived"* land near each other even though they share almost no words.
3. **Cluster** them — find the natural clumps without being told what to look for.
4. Ask an LLM to name each clump.

Out comes a three-level menu structure derived from what people actually say.

There is an iron rule about this, worth stating because breaking it is an easy and
invisible mistake: **you may not discover categories from a dataset that already has
categories, and then grade yourself on that dataset.** You would rediscover the labels you
were given and call it a result. So the corpora have fixed, enforced roles, and the code
refuses at the point of the mistake rather than producing a flattering number.

---

## 4. The four rules that shape everything

Most of the unusual design decisions come from four rules that are treated as
non-negotiable. They are in [`CLAUDE.md`](../CLAUDE.md) and they are binding.

### Rule 1 — The core knows nothing about any industry

The words `order`, `refund`, `claim`, `patient`, `prescription` appear nowhere in
`src/ccas/`. **A test fails the build if they do.**

Why so strict? Because the usual outcome is a codebase with `if industry == "healthcare"`
scattered through it, where adding a seventh vertical means touching forty files and
breaking the other six. Making it structurally impossible is cheaper than discipline.

A feature is not finished until it works for two different packs.

### Rule 2 — Private data cannot leak, structurally

This is the one to understand if you only read one section.

The usual approach is *"remember to redact before logging"*. That fails, because someone
eventually forgets, and the failure is silent and permanent.

Instead, redaction is enforced by the **type system**. In plain terms: there is a special
kind of text that is the *only* kind allowed to leave the process, and it cannot exist
without proof that it was redacted. Records refuse to be created from unproven text. The
refusal happens at construction, not at review time.

You can watch it refuse:

```bash
uv run python -m cli.demo gate
```

```
  redaction status     unverified
  egress permitted     False

  attempting RedactedText.require_egress() ...
    refused: egress blocked: redaction status is unverified
  attempting to build a CallLog from it ...
    refused: utterance 0 carries redaction status unverified
  attempting to build a HandoffContext from it ...
    refused: summary redaction status is unverified

  The gate is structural: there is no code path that emits unredacted text.
```

Three consequences that follow from taking this seriously:

- **If the redaction engine breaks, the call fails.** It never carries on unredacted. An
  outage must never quietly become a data breach.
- **Logs record types and counts, never content.** `{payment_card: 1, email: 1}` — never
  the values. Trying to log raw text raises an error.
- **Private data recognition runs locally.** It is never sent to a cloud service, because
  sending data to a third party to ask whether it is private is self-defeating.

This rule is also why the three gaps found in the handoff package (documented in
[§7](#7-what-is-honestly-not-finished)) were treated as serious rather than tidied away
quietly.

### Rule 3 — The whole round trip must fit in 800 milliseconds

From the caller finishing their sentence to the first sound of the reply: **800 ms at the
95th percentile** — meaning 95 calls in 100 must beat it, not merely the average.

Above roughly a second, people think the line has dropped and start saying "hello?".

Each stage gets a slice:

| Stage | Budget |
|---|---|
| Detecting end of speech | 100 ms |
| Speech to text | 180 ms |
| Redaction | 3 ms |
| Working out the intent | 90 ms |
| Calling the backend | 150 ms |
| First word out of the AI | 180 ms |
| First audio out | 120 ms |
| **Declared total** | **823 ms** |

That total is **larger than the 800 ms ceiling, on purpose.** The negative slack means any
regression shows up immediately instead of being quietly absorbed by spare room. Raising
any of those numbers requires a written, reviewed decision record — you cannot fix a
failing performance test by loosening the target.

The 3 ms redaction slice is why redaction runs in two modes: fast pattern-matching on the
live call (measured 111 µs — 27× headroom), and a slower, more thorough pass with local
AI (~7 ms) for anything stored, mined, or shown to a human, where nobody is waiting.

### Rule 4 — The AI picks from a list, it never improvises

A popular way to build this would be to hand an AI model a set of tools and let it decide
what to do in a loop. This project forbids that.

Instead the conversation moves through an explicit map of nine states — `greet`,
`identify`, `route`, `clarify`, `slot_fill`, `tool_exec`, `respond`, `escalate`, `close` —
with explicit rules for each transition and a hard limit on turns.

The AI's job is narrow: **choose which of the declared tools fits.** It never constructs a
web address, writes a query, or calls something that was not registered in advance.

Why refuse the flexible approach? Because in customer service you must be able to say what
the system will do before it does it. An open-ended loop can always surprise you, and
"the AI decided to try something" is not an acceptable sentence in a regulated
conversation. Every route must also have a reachable path to a human.

Four **policies** run in a fixed order on every turn — `risk`, `retry`, `sentiment`,
`confidence` — and any of them can stop the conversation and hand over.

---

## 5. What is actually working

Honesty matters more than a tidy table here, so this distinguishes *built and proven*
from *built but never run against a real vendor*.

| Part | State |
|---|---|
| Contracts, config, AI bindings, logging | **Working** |
| M1 Ingestion | **Working** |
| M2 Redaction | **Working** — the strongest-tested part of the system |
| M3 Mining | **Working**, but see the honest caveat in [§7](#7-what-is-honestly-not-finished) |
| M4 Orchestration | **Working** |
| M5 Voice | **Built, then deliberately paused.** Complete and kept passing its tests, but the vendor adapters have never met a live service. Voice needs paid accounts and a media server; the text channel proves the same core without either |
| M6a Copilot handoff | **Working** — handoff package, egress gate, CRM adapter, demo, gate test |
| M6b Evaluation | **Partly.** The two measurement harnesses work and have produced real numbers. The automated quality-judging half is not built; its data shapes exist, its runtime does not |

**The text channel runs end to end on one AI key.** No Docker, no databases, no vendor
accounts.

Two honest qualifications:

- **The backends are mocks.** No real retail company is plugged in. The *boundary* is real
  and tested; what sits behind it is a stand-in, and every real integration is
  customer-specific.
- **The AI models are free-tier.** Fine for proving correctness, far too slow for the
  800 ms budget — one of them measured 12.5 seconds. That gap is measured and recorded
  rather than hidden.

---

## 6. What the numbers actually say

### Does it understand what callers want?

Tested on 30 held-out customer-service messages from a public labelled dataset —
*held-out* meaning these rows played no part in building the category list, which is the
only kind of test that means anything. (The dataset is template-generated rather than
transcribed from real calls, so read it as a fair grader of understanding, not as proof of
performance on live speech.)

```
23/26 exact (88.5%), category 100.0%, 4 unmeasured
```

Three numbers, three different claims:

- **88.5% exact** — it chose precisely the right one of 27 labels.
- **100% category** — *every single answer was in the right broad area.* When it was
  wrong, it was wrong between neighbours: it said "edit account" where the answer was
  "switch account". Those are arguably the same request phrased differently. A system that
  is never wildly wrong is a very different proposition from one that is occasionally
  lost, and this distinction is invisible in the headline figure.
- **4 unmeasured** — the free AI account hit its daily cap and never answered. Those are
  excluded rather than counted as mistakes, which matters more than it sounds and is the
  subject of [ADR-0021](adr/0021-a-timeout-is-read-against-the-run.md).

**Honest limit:** 26 answered questions across 27 categories is about one each. The
headline is sound; per-category detail from this run is not.

### Does it depend on one AI vendor?

A rule here says no part of the system may depend on a single AI model. This is checked by
asking two different models the same questions:

```
8 cases · agreement 1.0 · 0 divergent · 0 unmeasured
openrouter      (nex-agi)          12,525 ms
openrouter_alt  (ling-3.0-flash)    2,347 ms
```

They agreed on everything — but one is **five times slower**. Nobody had compared them
until this test printed both, and that comparison turned out to matter a great deal, as
the next section explains.

### A cautionary tale worth reading

An earlier run of the accuracy test returned **52.5%**, which looks like a system getting
half its answers wrong.

It wasn't. Every single failure was the AI *never answering at all* — not one wrong
answer in the entire run. The free-tier model was so overloaded it ran out of time on 19
of 40 questions, and the test was counting "never answered" as "answered wrongly".

Two lessons the project now encodes in code:

1. **A question nobody answered is not a wrong answer.** It leaves the denominator
   entirely. Conflating the two makes a working system look broken, which trains everyone
   to ignore the test — the worst possible outcome for a safety check.
2. **The test must record *why* something failed.** It didn't, so the diagnosis had to be
   reconstructed from arithmetic about how long the run took. It records the cause now.

The fix for the underlying slowness turned out to be embarrassing in a useful way: a fast
model was already configured. Nobody had looked, because no measurement had ever put the
two side by side.

### Is it tested?

**963 automated tests** pass, plus **95 more** for the paused voice channel.

Not all tests are equal, and this project weights them deliberately:

- The privacy rules get **100% coverage** and are additionally tested by generating
  thousands of random fake identifiers and confirming none escapes.
- Every module has a **gate test** proving it works as a whole.
- Performance budgets are tested, and a test may never be loosened to make a build pass.

---

## 7. What is honestly not finished

Kept deliberately visible. The full list with reasons is in
[`future-scoped-work.md`](future-scoped-work.md).

**The judging half of M6.** Automated quality-scoring of conversations. Data shapes exist,
the runtime does not.

**The agent's screen.** The handoff package is built and delivered, but the web interface
for a human agent to open it is not written.

**A mystery nobody has solved.** When mining categories from a real corpus of insurance
calls, the system only manages to place **19.7%** of sentences into a cluster. Three
explanations have been tested and disproved — and a fourth apparent finding was retracted
when the measurement behind it turned out to be an artefact of how it was calculated. **The cause is
genuinely unknown**, and it is written down as unknown rather than papered over.

**Three privacy gaps, found and fixed during this work.** The handoff package promised in
its own documentation that every text field carried proof of redaction. Three fields did
not — one of them the field that travels to the vendor system, and one carrying
caller-verified details like a date of birth, which was being copied out unredacted. No
data escaped, because the only code populating it happened to use safe values. But the
*contract* permitted it, and a contract that relies on the caller being careful is not a
contract. Now enforced by the type system like everything else.

**The web console has no browser test.** Its wiring is checked; whether it visibly renders
is not.

**Infrastructure never started.** The container setup is valid but has never run, having
been written on a machine without Docker.

---

## 8. Try it yourself

Needs Python and one free AI key. No Docker, no databases, no vendor accounts.

```bash
uv sync --all-extras          # install
cp .env.example .env          # add one AI key; everything else has a working default
```

### Watch every stage of the pipeline

```bash
uv run python -m cli.demo pipeline "where is my delivery"
```

Prints every stage with its real output. Anything not yet built says so and names when it
lands — **a stage never invents output it did not produce.** A demo that faked a redaction
would be worse than no demo.

### Watch the privacy gate refuse

```bash
uv run python -m cli.demo gate
```

### Talk to it

```bash
make workbench                # http://127.0.0.1:8000
```

Binds to your own machine only. It is a development tool, never a production surface.

### Check everything still works

```bash
make check     # formatting, types, and the full test suite
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Barge-in** | The caller interrupting the system mid-sentence; playback stops and it listens |
| **CCaaS** | Contact Centre as a Service — the vendor platform running a call centre |
| **Clustering** | Finding natural groupings in data without being told what to look for |
| **CRM** | Customer Relationship Management — the system a human agent works in |
| **CTI** | Computer Telephony Integration — how call data reaches an agent's screen |
| **Domain pack** | A data file describing one industry: its words, tools, rules. Swapping it changes what the system handles, with no code change |
| **DTMF** | Keypad tones — pressing digits instead of speaking |
| **Embedding** | A list of numbers representing a sentence's meaning, so similar meanings sit close together |
| **Escalation** | Handing the conversation to a human |
| **Grounded** | Every factual claim traceable to a real tool result, never invented |
| **Handoff** | The package of context passed to a human agent so the caller need not repeat themselves |
| **Intent** | What the caller wants, as one label from a fixed list |
| **IVR** | Interactive Voice Response — the "press 1 for…" system |
| **LLM** | Large Language Model — ChatGPT-style AI |
| **p95** | The 95th percentile: 95 of 100 calls beat this number. Stricter and more honest than an average |
| **PII / PHI** | Personally Identifiable / Protected Health Information — private data |
| **Placeholder** | The token replacing private data, e.g. `[PAYMENT_CARD_1]` |
| **Redaction** | Finding private information and replacing it |
| **Slot** | A piece of information needed to complete a request, e.g. an order number |
| **STT / TTS** | Speech-To-Text / Text-To-Speech |
| **Taxonomy** | The three-level tree of everything callers might want |
| **Tool** | A registered backend the system may call, declared in advance |
| **VAD** | Voice Activity Detection — spotting when someone stops speaking |
| **WebRTC** | The browser standard for real-time audio |

---

## Where to go next

| You want | Read |
|---|---|
| How it fits together, with diagrams | [`design-doc.md`](design-doc.md) |
| Exact data shapes and API contracts | [`tech-spec.md`](tech-spec.md) |
| Why a particular choice was made | [`adr/`](adr/README.md) |
| The rules, as binding text | [`../CLAUDE.md`](../CLAUDE.md) |
| What was left out and why | [`future-scoped-work.md`](future-scoped-work.md) |
