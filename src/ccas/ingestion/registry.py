"""Adapter lookup by source."""

from __future__ import annotations

from ccas.ingestion.adapters.aixblock import AixBlockAdapter
from ccas.ingestion.adapters.bitext import BitextAdapter
from ccas.ingestion.adapters.generic_csv import GenericCsvAdapter
from ccas.ingestion.adapters.natcs import NatcsAdapter
from ccas.ingestion.adapters.synthetic import SyntheticAdapter
from ccas.ingestion.base import SourceAdapter
from ccas.schemas.call_log import DatasetSource

__all__ = ["ADAPTERS", "adapter_for"]

ADAPTERS: dict[DatasetSource, type[SourceAdapter]] = {
    DatasetSource.AIXBLOCK: AixBlockAdapter,
    DatasetSource.BITEXT: BitextAdapter,
    DatasetSource.NATCS: NatcsAdapter,
    DatasetSource.GENERIC_CSV: GenericCsvAdapter,
    DatasetSource.SYNTHETIC: SyntheticAdapter,
}


def adapter_for(source: DatasetSource) -> SourceAdapter:
    try:
        return ADAPTERS[source]()
    except KeyError as exc:
        raise KeyError(
            f"no adapter for {source.value!r}; available: {sorted(s.value for s in ADAPTERS)}"
        ) from exc
