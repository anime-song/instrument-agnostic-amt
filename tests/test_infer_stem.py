from __future__ import annotations

from pathlib import Path
import pretty_midi

from infer_stem import (
    merge_midis_logic,
    resolve_stem_model_type,
    resolve_stem_paths,
)


def test_resolve_stem_model_type() -> None:
    assert resolve_stem_model_type("drums_stem") == "drums"
    assert resolve_stem_model_type("bass_stem") == "bass_v2"
    assert resolve_stem_model_type("vocal_stem") == "vocal_harmony"
    assert resolve_stem_model_type("guitar_stem") == "guitar_v1_5"
    assert resolve_stem_model_type("other_stem") == "other"
    assert resolve_stem_model_type("piano_stem") == "default"


def test_merge_midis_logic(tmp_path: Path) -> None:
    midi1 = pretty_midi.PrettyMIDI()
    inst1 = pretty_midi.Instrument(program=0, name="Piano")
    inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    midi1.instruments.append(inst1)
    path1 = tmp_path / "stem1.mid"
    midi1.write(str(path1))

    midi2 = pretty_midi.PrettyMIDI()
    inst2 = pretty_midi.Instrument(program=33, name="Bass")
    inst2.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=0.0, end=1.0))
    midi2.instruments.append(inst2)
    path2 = tmp_path / "stem2.mid"
    midi2.write(str(path2))

    output_path = tmp_path / "merged.mid"
    merge_midis_logic([path1, path2], output_path, max_melodic_instruments=15)

    assert output_path.exists()
    merged_midi = pretty_midi.PrettyMIDI(str(output_path))
    assert len(merged_midi.instruments) == 2
