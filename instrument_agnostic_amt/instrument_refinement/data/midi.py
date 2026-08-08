"""MIDI を「ノートの平らな表」に変換する。

モデルはトラック構造を見ず、ノートの列だけを扱う。一方で結果を書き戻すときは元の
トラックのどのノートだったかを知る必要があるため、(トラック番号, ノート番号) も
一緒に持ち回る（inference/refine.py の _rewrite_midi がこれを使う）。

学習データには壊れかけの MIDI が混ざるので、読めるものは直して読み、どうしても
読めなければ空の表を返して 1 ファイル分だけ捨てる（学習全体を落とさない）。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import pretty_midi

from ...taxonomy.instrument_classes import (
    get_instrument_class_id,
    get_instrument_class_id_by_name,
)

# program 番号では主旋律とコーラスを区別できないので、トラック名から推定する。
_VOCAL_CLASS_BY_INSTRUMENT_NAME = {
    "vocal": "melody",
    "vocals": "melody",
    "lead_vocal": "melody",
    "lead_vocals": "melody",
    "vocal_harmony": "vocal_harmony",
    "back_vocal": "vocal_harmony",
    "back_vocals": "vocal_harmony",
    "backing_vocal": "vocal_harmony",
    "backing_vocals": "vocal_harmony",
}


@dataclass(frozen=True)
class RefinementNoteTable:
    """ドラム以外の全ノートを並べた表。各配列は同じ長さで、i 番目が同じノートを指す。

    Attributes:
        start_seconds/end_seconds: 発音区間（秒）。
        pitch: MIDI ノート番号。
        prior_class_id: 元 MIDI が主張していた楽器クラス。推論では使わず、
            学習で正解ラベルとして使う場合がある（note_target_mode="midi"）。
        instrument_index/note_index: 元 MIDI での位置。書き戻し時の対応づけに使う。
    """

    start_seconds: np.ndarray
    end_seconds: np.ndarray
    pitch: np.ndarray
    prior_class_id: np.ndarray
    instrument_index: np.ndarray
    note_index: np.ndarray

    @property
    def note_count(self) -> int:
        return int(self.pitch.size)


def empty_note_table() -> RefinementNoteTable:
    """ノート 0 個の表。読み込み失敗時のフォールバックとして使う。"""
    return RefinementNoteTable(
        start_seconds=np.zeros(0, dtype=np.float32),
        end_seconds=np.zeros(0, dtype=np.float32),
        pitch=np.zeros(0, dtype=np.int64),
        prior_class_id=np.zeros(0, dtype=np.int64),
        instrument_index=np.zeros(0, dtype=np.int64),
        note_index=np.zeros(0, dtype=np.int64),
    )


def _repair_time_signatures(midi_data: mido.MidiFile) -> int:
    """拍子が 0/x や 2 のべき乗でない分母だと pretty_midi が失敗するので 4/4 に直す。"""
    repaired = 0
    for track in midi_data.tracks:
        for message_index, message in enumerate(track):
            if message.type != "time_signature":
                continue
            numerator = int(message.numerator)
            denominator = int(message.denominator)
            valid_denominator = denominator > 0 and (denominator & (denominator - 1)) == 0
            if numerator > 0 and valid_denominator:
                continue
            track[message_index] = message.copy(
                numerator=numerator if numerator > 0 else 4,
                denominator=denominator if valid_denominator else 4,
            )
            repaired += 1
    return repaired


def _parse_repaired_midi(midi_path: Path, *, warn_repairs: bool) -> pretty_midi.PrettyMIDI | None:
    """壊れかけの MIDI を可能な範囲で直して読む。読めなければ None。

    学習データには読めない MIDI が混ざるので、1 ファイル分だけ捨てて学習は続ける。

    warn_repairs=False にすると「直せた」通知だけを出さない。学習では 8 万件以上を
    読むので、直せたものまで 1 件ずつ報告すると進捗表示が埋まってしまう。読めずに
    捨てた MIDI（= 学習データが減る）の警告は、この指定でも必ず出す。
    """
    try:
        midi_data = mido.MidiFile(str(midi_path))
        repaired_time_signatures = _repair_time_signatures(midi_data)
        # 末尾の end_of_track が極端に先の時刻を指しているとき、その delta を 0 にする。
        # （曲本体は正常なのに、末尾だけで読み込みが落ちるケースがある。）
        repaired_end_ticks = 0
        try:
            midi = pretty_midi.PrettyMIDI(mido_object=midi_data)
        except ValueError as error:
            if "largest tick" not in str(error):
                raise
            for track in midi_data.tracks:
                if track and track[-1].type == "end_of_track" and int(track[-1].time) > 0:
                    track[-1] = track[-1].copy(time=0)
                    repaired_end_ticks += 1
            if not repaired_end_ticks:
                raise
            midi = pretty_midi.PrettyMIDI(mido_object=midi_data)
    except Exception as error:
        warnings.warn(
            f"Skipping unreadable MIDI note targets in {midi_path}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if warn_repairs and repaired_end_ticks:
        warnings.warn(
            f"Repaired {repaired_end_ticks} oversized end-of-track delta(s) in {midi_path}",
            RuntimeWarning,
            stacklevel=3,
        )
    if warn_repairs and repaired_time_signatures:
        warnings.warn(
            f"Repaired {repaired_time_signatures} invalid time signature(s) in {midi_path}",
            RuntimeWarning,
            stacklevel=3,
        )
    return midi


def _instrument_class_id(instrument: pretty_midi.Instrument) -> int:
    """トラックの楽器クラス。program 番号では主旋律とコーラスを区別できないので名前も見る。"""
    normalized_name = str(instrument.name).strip().casefold().replace("-", "_").replace(" ", "_")
    class_name = _VOCAL_CLASS_BY_INSTRUMENT_NAME.get(normalized_name)
    if class_name is not None:
        return get_instrument_class_id_by_name(class_name)
    return get_instrument_class_id(int(instrument.program), is_drum=False)


def _note_rows(midi: pretty_midi.PrettyMIDI) -> list[tuple[float, float, int, int, int, int]]:
    rows: list[tuple[float, float, int, int, int, int]] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        # ドラムはこのモデルの対象外。instrument_index は元のまま進めるので、
        # 書き戻しの対応づけはずれない。
        if instrument.is_drum:
            continue
        prior_class_id = _instrument_class_id(instrument)
        for note_index, note in enumerate(instrument.notes):
            rows.append(
                (
                    float(note.start),
                    float(note.end),
                    int(note.pitch),
                    int(prior_class_id),
                    int(instrument_index),
                    int(note_index),
                )
            )
    return rows


def load_refinement_note_table(
    path: str | Path | None, *, warn_repairs: bool = True
) -> RefinementNoteTable:
    """MIDI を読み、ドラム以外のノートを並べた表にする。読めなければ空表を返す。

    warn_repairs=False で「自動で直せた」通知を止める（大量に読む学習用）。
    """
    if path is None:
        return empty_note_table()
    midi_path = Path(path)
    if not midi_path.is_file():
        return empty_note_table()
    midi = _parse_repaired_midi(midi_path, warn_repairs=warn_repairs)
    if midi is None:
        return empty_note_table()
    rows = _note_rows(midi)
    if not rows:
        return empty_note_table()
    # 発音時刻・音高の順に整列する。トラックの並び順に結果が左右されないようにするため。
    rows.sort(key=lambda row: (row[0], row[2], row[1], row[4], row[5]))
    values = np.asarray(rows, dtype=np.float64)
    return RefinementNoteTable(
        start_seconds=values[:, 0].astype(np.float32),
        end_seconds=values[:, 1].astype(np.float32),
        pitch=values[:, 2].astype(np.int64),
        prior_class_id=values[:, 3].astype(np.int64),
        instrument_index=values[:, 4].astype(np.int64),
        note_index=values[:, 5].astype(np.int64),
    )
