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
| AIxBlock | [`AIxBlock/92k-real-world-call-center-scripts-english`](https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english) | **Research use only.** The card forbids commercial use, resale and redistribution — see `docs/future-scoped-work.md` 9.6 |
| Bitext | [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing-1.0 |
| NatCS | [`splevine/dstc11-intent`](https://huggingface.co/datasets/splevine/dstc11-intent) | Per the upstream card |

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

**Nothing derived from AIxBlock is committed.** No taxonomy was mined from it — that work
is blocked and tracked in `docs/future-scoped-work.md` 9.12. Its research-only restriction
therefore attaches to nothing in this repository.

`tests/fixtures/` is synthetic throughout, per CLAUDE.md Rule 2.

## Dependencies

Runtime and development dependencies are declared in `pyproject.toml` and pinned in
`uv.lock`, each under its own licence.
