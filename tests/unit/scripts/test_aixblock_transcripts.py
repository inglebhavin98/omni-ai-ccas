"""Turn recovery from undiarized AIxBlock transcripts.

The corpus publishes word-level timing and no speaker labels, so these tests pin the two
decisions that shape every downstream CallLog: where a turn ends, and when a turn is
called the agent's rather than the caller's. See ADR-0010.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from scripts.aixblock_transcripts import (
    ARCHIVES,
    DEFAULT_GAP_MS,
    MIN_TURNS,
    read_archive,
    record_id,
    segment_turns,
)


def _words(*spans: tuple[str, int, int]) -> list[dict[str, Any]]:
    return [{"text": t, "start": s, "end": e, "speaker": None} for t, s, e in spans]


class TestSegmentTurns:
    def test_a_long_silence_ends_a_turn(self) -> None:
        turns = segment_turns(
            _words(("Hello", 0, 300), ("there", 300, 600), ("Yes", 9000, 9300)), gap_ms=500
        )
        assert [t["text"] for t in turns] == ["Hello there", "Yes"]

    def test_words_inside_the_gap_stay_in_one_turn(self) -> None:
        turns = segment_turns(_words(("I", 0, 100), ("need", 200, 400), ("help", 450, 700)), 500)
        assert len(turns) == 1
        assert turns[0]["text"] == "I need help"

    def test_boundary_is_inclusive(self) -> None:
        """A gap exactly at the threshold splits; one millisecond under does not."""
        exact = segment_turns(_words(("a", 0, 100), ("b", 100 + DEFAULT_GAP_MS, 900)))
        under = segment_turns(_words(("a", 0, 100), ("b", 100 + DEFAULT_GAP_MS - 1, 900)))
        assert len(exact) == 2
        assert len(under) == 1

    def test_timings_are_carried_through(self) -> None:
        turns = segment_turns(_words(("a", 40, 100), ("b", 120, 900)), 500)
        assert turns[0]["start_ms"] == 40
        assert turns[0]["end_ms"] == 900

    def test_blank_and_malformed_words_are_skipped(self) -> None:
        words: list[Any] = [
            {"text": "  ", "start": 0, "end": 10},
            "not-a-dict",
            {"text": "real", "start": 20, "end": 40},
        ]
        turns = segment_turns(words, 500)
        assert [t["text"] for t in turns] == ["real"]

    def test_missing_timings_do_not_raise(self) -> None:
        turns = segment_turns([{"text": "hi"}, {"text": "there"}], 500)
        assert turns[0]["start_ms"] is None
        assert turns[0]["text"] == "hi there"

    def test_empty_input(self) -> None:
        assert segment_turns([]) == []


class TestSpeakerAttribution:
    """The filter is conservative on purpose: it excludes boilerplate, not speakers."""

    @pytest.mark.parametrize(
        "text",
        [
            "Thank you for calling, how may I help you today?",
            "Your call may be monitored or recorded for training purposes.",
            "If you know your party's extension, please dial it at this time.",
            "Press one for sales.",
            "Let me pull up your details.",
            "May I have your name please?",
            "Is there anything else I can help you with?",
        ],
    )
    def test_boilerplate_is_attributed_to_the_agent(self, text: str) -> None:
        assert segment_turns(_words((text, 0, 100)))[0]["speaker"] == "agent"

    @pytest.mark.parametrize(
        "text",
        [
            "I need to check on something I bought last week.",
            "My device stopped working after the update.",
            "Yes, that is correct.",
            "I was calling because nobody got back to me.",
        ],
    )
    def test_everything_else_stays_with_the_caller(self, text: str) -> None:
        assert segment_turns(_words((text, 0, 100)))[0]["speaker"] == "customer"

    def test_attribution_is_per_turn_not_per_call(self) -> None:
        turns = segment_turns(
            _words(("Thank you for calling.", 0, 100), ("I need some help.", 5000, 6000))
        )
        assert [t["speaker"] for t in turns] == ["agent", "customer"]


class TestRecordId:
    def test_is_stable(self) -> None:
        assert record_id("a.zip", "m.json") == record_id("a.zip", "m.json")

    def test_distinguishes_members_and_archives(self) -> None:
        assert record_id("a.zip", "m.json") != record_id("a.zip", "n.json")
        assert record_id("a.zip", "m.json") != record_id("b.zip", "m.json")

    def test_does_not_carry_the_dialled_number(self) -> None:
        """Rule 2: AIxBlock file names embed a phone number; it must not survive."""
        member = "DMEMED_rmbpo11_20211119-111629_6149286164_4712234-all_transcript.json"
        generated = record_id("medical_equipment_outbound.zip", member)
        assert "6149286164" not in generated
        assert "4712234" not in generated
        assert generated.startswith("aix-")


class TestReadArchive:
    @staticmethod
    def _archive(tmp_path: Path, payloads: dict[str, Any]) -> Path:
        path = tmp_path / "sample.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, payload in payloads.items():
                bundle.writestr(name, json.dumps(payload))
        return path

    def test_short_dialogues_are_dropped(self, tmp_path: Path) -> None:
        two_turns = {"words": _words(("hello", 0, 100), ("yes", 90000, 90100))}
        archive = self._archive(tmp_path, {"a/one.json": two_turns})
        lines, stats = read_archive(archive)
        assert lines == []
        assert stats["too_short"] == 1
        assert stats["kept"] == 0

    def test_a_real_dialogue_is_kept(self, tmp_path: Path) -> None:
        words = _words(*[(f"word{i}", i * 9000, i * 9000 + 100) for i in range(MIN_TURNS)])
        archive = self._archive(
            tmp_path, {"a/one.json": {"words": words, "audio_duration": 12, "confidence": 0.9}}
        )
        lines, stats = read_archive(archive)
        assert stats["kept"] == 1
        record = json.loads(lines[0])
        assert len(record["turns"]) == MIN_TURNS
        assert record["diarization"] == "pause_segmented"
        assert record["audio_duration_s"] == 12

    def test_mac_resource_forks_and_junk_are_ignored(self, tmp_path: Path) -> None:
        words = _words(*[(f"w{i}", i * 9000, i * 9000 + 100) for i in range(MIN_TURNS)])
        archive = self._archive(tmp_path, {"a/one.json": {"words": words}})
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr("__MACOSX/a/._one.json", b"\x00\x01")
            bundle.writestr("a/notes.txt", "ignored")
        lines, stats = read_archive(archive)
        assert stats["members"] == 1
        assert len(lines) == 1

    def test_unparsable_members_are_counted_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.zip"
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("a/broken.json", "{not json")
            bundle.writestr("a/wrong_shape.json", json.dumps({"text": "no words list"}))
        lines, stats = read_archive(path)
        assert lines == []
        assert stats["unparsable"] == 2


class TestArchiveManifest:
    def test_every_archive_names_an_output_and_a_campaign(self) -> None:
        assert all(a.output and a.campaign and a.filename.endswith(".zip") for a in ARCHIVES)

    def test_filenames_are_unique(self) -> None:
        assert len({a.filename for a in ARCHIVES}) == len(ARCHIVES)

    def test_every_output_file_has_a_source(self) -> None:
        assert {"retail", "healthcare"} <= {a.output for a in ARCHIVES}

    def test_the_large_archives_are_opt_in(self) -> None:
        default = [a for a in ARCHIVES if not a.large]
        assert {a.output for a in default} == {"retail", "healthcare"}

    def test_an_unverified_campaign_says_so_rather_than_guessing(self) -> None:
        """The archive names have been wrong three times; a guess must not replace them.

        An archive whose transcripts have not been profiled carries the literal string
        ``unverified`` -- never a topic inferred from its file name, which is how
        "automotive-stereo" (3% car audio) and "medical_equipment" (89% Medicare) got
        their original labels.
        """
        for archive in ARCHIVES:
            if not archive.verified:
                assert archive.campaign == "unverified", archive.filename

    def test_verified_campaigns_are_not_echoes_of_the_filename(self) -> None:
        for archive in ARCHIVES:
            if archive.verified:
                stem = archive.filename.lower()
                # A measured label may share a word with the name, but must not be the
                # name's own topic when measurement contradicted it.
                assert archive.campaign != "unverified"
                assert not (
                    archive.campaign == "medical_equipment_sales" and "medical_equipment" in stem
                )

    def test_output_is_a_path_not_a_domain_claim(self) -> None:
        """ADR-0013: the ZIP names describe BPO campaigns, not verticals.

        `retail.jsonl` is 100% insurance calls. Reading `output` as a domain pack is the
        exact mistake that produced an insurance taxonomy labelled retail, so the
        campaign a file is built from must stay visible and must not agree with its name.
        """
        campaigns: dict[str, set[str]] = {}
        for archive in ARCHIVES:
            campaigns.setdefault(archive.output, set()).add(archive.campaign)
        # The file called "retail" is fed by an insurance campaign.
        assert any("insurance" in c for c in campaigns["retail"])
        # No file's campaigns merely restate its own name.
        assert all(output not in found for output, found in campaigns.items())
