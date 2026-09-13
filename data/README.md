# data/

Corpora and derived artifacts. **Nothing in here is committed** except these READMEs —
see `.gitignore`.

```
data/
├── raw/          the three ground-truth corpora, as published
│   ├── aixblock/     mining corpus        (unlabelled, high volume, undiarized)
│   ├── bitext/       supervision corpus   (labelled, single-turn)
│   └── natcs/        dialogue corpus      (caller turns only in the current release)
├── interim/      redacted + normalised CallLog, gzipped JSONL, partitioned by source
└── processed/    mined taxonomies, embeddings, eval fixtures
```

Hydrate `raw/` with:

```bash
uv run python scripts/fetch_datasets.py --all
```

Each corpus README records what the live download actually contains — which in two cases
is not what the adapter originally assumed. Read them before trusting a number derived
from one.

**`interim/` partitions by source, not by domain.** Ingesting two verticals of the same
corpus on the same day writes them into the same partition, and a later mining run pools
them. Keep them apart with `--out` until that is fixed.

## The corpora are not interchangeable

Each answers a different question, and pooling them silently produces eval numbers that
look fine and mean nothing. The separation is enforced in code, not convention:

```python
from ccas.ingestion.datasets import DatasetRole, require_role
require_role(source, DatasetRole.INTENT_MINING)   # raises DatasetRoleError on misuse
```

See `docs/adr/0004-dataset-role-separation.md` for the reasoning, and each corpus
README for what it may and may not be used for.

## Nothing here reaches an external boundary unredacted

Raw corpora are inputs to Module 2, never to a model. Only `data/interim/` — which holds
`CallLog` records, and a `CallLog` cannot be constructed from unredacted text — is safe
to feed downstream (CLAUDE.md Rule 2).
