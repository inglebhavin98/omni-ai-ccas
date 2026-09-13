# Corpus fixtures

Hand-written, synthetic samples in the shape each real corpus publishes. No real
customer data (CLAUDE.md Rule 2); the PII-shaped values are generated and invalid
(card numbers are Luhn-valid but in test ranges, names are invented).

These exist so adapter tests run with no download. The *real* file schemas are
documented in each adapter's `expected_layout` and in `data/raw/*/README.md`; they have
not been verified against a live download, so an adapter may need adjusting on first
contact with the actual corpus.
