# Third-party data

The MIT licence in `LICENSE` covers this repository's own source, configuration and
documentation. It does **not** cover the upstream datasets described below, which have
their own terms.

## Corpora

No corpus data is committed. `data/**` is gitignored; only the READMEs describing each
corpus are tracked. `scripts/fetch_datasets.py` downloads them on demand from Hugging Face,
and their terms bind whoever runs it.

| Corpus | Upstream | Terms |
|---|---|---|
| AIxBlock | [`AIxBlock/92k-real-world-call-center-scripts-english`](https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english) | **Tagged `cc-by-nc-4.0`; the card's prose is stricter** — see below |
| Bitext | [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing-1.0 |
| NatCS | [`splevine/dstc11-intent`](https://huggingface.co/datasets/splevine/dstc11-intent) | Apache-2.0 |

## Derived artefacts committed here

**`domains/retail/taxonomy.json` is derived from the Bitext corpus** and is the only
committed file that is. It was produced by `scripts/adopt_taxonomy.py`, which takes that
dataset's published category and intent labels, counts how often each appears, and reads
the placeholder annotations marking where parameters belong. The file records this in its
own `provenance` and `adopted_from` fields. See `docs/adr/0020-adopt-a-published-label-set.md`.

It contains no utterance text from the corpus — no exemplars, no transcripts, no call ids.

CDLA-Sharing-1.0 distinguishes redistributed *Data* from *Results* of computational
analysis, and a file holding label names and aggregate counts is arguably the latter. This
notice does not assert which it is: attribution is given because it is owed under that
licence in either reading, and because naming a source costs nothing.

### AIxBlock: the card contradicts itself

Its two licence statements disagree, and anyone checking will see both:

- the YAML front matter declares **`license: cc-by-nc-4.0`**, which *permits*
  redistribution for non-commercial purposes with attribution;
- the prose declares *"Provided **strictly for research and AI model development**.
  **Commercial use, resale, or redistribution is prohibited.**"*, which forbids
  redistribution outright.

**This project follows the stricter reading** and redistributes none of it. That is a
choice between two conflicting statements in the source, not a settled interpretation of
one — recorded here so a reader sees the choice rather than only its result. Attribution
is required under either reading: the corpus is credited to **AIxBlock and three
independent researchers — Gaurav Chawla, Raghu Banda and Caleb DeLeeuw.**

`docs/future-scoped-work.md` 9.6 tracks the commercial-use restriction.

**Nothing derived from AIxBlock is committed.** No taxonomy was mined from it — that work
is blocked and tracked in `docs/future-scoped-work.md` 9.12. Its restrictions therefore
attach to nothing in this repository, however the contradiction above is resolved.

`tests/fixtures/` is synthetic throughout, per CLAUDE.md Rule 2.

## Dependencies

Runtime and development dependencies are declared in `pyproject.toml` and pinned in
`uv.lock`, each under its own licence.
