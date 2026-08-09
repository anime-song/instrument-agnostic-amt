"""楽器クラスと「ステム」の対応づけ。

このパッケージには 3 種類の粒度が出てくるので、混同しないよう注意:

1. 楽器クラス   … taxonomy 側の細かいラベル（"acoustic_guitar" など）。最終出力。
2. ファミリー   … 楽器クラスをざっくりまとめた括り（"guitar" など）。補助タスク用。
3. ステム       … 分離器が出す音源の種類（"vocals" / "bass" / "guitar" / "other" …）。
                   楽器候補の絞り込みと、モデルに渡す文脈 ID に使う。

学習データ側のステム名は細かい（"bowed_strings" など）が、推論時に手に入るのは分離器の
6 ステムだけなので、``inference_stem_group`` で推論時の粒度へ寄せる。
"""

from __future__ import annotations

from ...taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)

# 補助タスク（ファミリー分類）のラベル。細かい楽器クラスを間違えても、
# ファミリーが当たっていれば部分点が入るようにして学習を安定させる狙い。
FAMILY_NAMES = (
    "bass",
    "brass",
    "effects",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "vocal",
    "woodwind",
    "other",
)

_FAMILY_BY_CLASS_NAME = {
    "acoustic_bass": "bass",
    "electric_bass": "bass",
    "slap_bass": "bass",
    "synth_bass": "bass",
    "brass": "brass",
    "orchestra_hit": "effects",
    "percussive_fx": "effects",
    "sound_fx": "effects",
    "synth_fx": "effects",
    "wind_chimes": "effects",
    "acoustic_guitar": "guitar",
    "distorted_guitar": "guitar",
    "electric_guitar_clean": "guitar",
    "electric_guitar_muted": "guitar",
    "guitar_harmonics": "guitar",
    "electric_piano": "keyboard",
    "organ": "keyboard",
    "piano": "keyboard",
    "plucked_keyboard": "keyboard",
    "chromatic_percussion": "percussion",
    "drums": "percussion",
    "timpani": "percussion",
    "orchestral_harp": "strings",
    "pizzicato_strings": "strings",
    "strings": "strings",
    "synth_lead": "synth",
    "synth_pad": "synth",
    "choir": "vocal",
    "melody": "vocal",
    "vocal_harmony": "vocal",
    "flute_pipe": "woodwind",
    "orchestral_woodwind": "woodwind",
    "sax": "woodwind",
    "accordion_family": "other",
    "ethnic": "other",
    "harmonica": "other",
}

# モデルに「今どのステムを聴いているか」を教えるための語彙（stem_embedding の入力）。
# 学習データの細かいステム名も含むため、上の 6 ステムより種類が多い。
STEM_CONTEXT_NAMES = (
    "bass",
    "bowed_strings",
    "guitar",
    "other",
    "other_keys",
    "other_plucked",
    "percussion",
    "piano",
    "vocals",
    "wind",
    "unknown",
)

REFINEMENT_STEM_EXTRA_CLASSES = {
    # The separator can place these sounds in the corresponding deployment
    # stem even though its base candidate mask is narrower.
    "other": ("acoustic_guitar",),
}


def inference_stem_group(stem_name: str | None) -> str:
    """Map a dataset stem name to the separated stem seen at inference time.

    部分一致で判定するので "drums_stem" や "song_vocals" もそのまま渡せる。
    どれにも当てはまらないものは "other" に落ちる（分離器の other ステム相当）。
    """
    normalized = str(stem_name or "").strip().casefold()
    if "drum" in normalized:
        return "drums"
    if "bass" in normalized:
        return "bass"
    if "vocal" in normalized:
        return "vocals"
    if "guitar" in normalized:
        return "guitar"
    if normalized == "piano":
        return "piano"
    return "other"


def refinement_stem_extra_class_ids(stem_name: str | None) -> tuple[int, ...]:
    """そのステムに実際は紛れ込みうる、候補マスク外の楽器クラスを返す。"""
    group = inference_stem_group(stem_name)
    return tuple(
        get_instrument_class_id_by_name(class_name) for class_name in REFINEMENT_STEM_EXTRA_CLASSES.get(group, ())
    )


def stem_context_id(stem_name: str | None) -> int:
    """ステム名を stem_embedding の入力 ID に変換する。未知の名前は "unknown"。"""
    normalized = str(stem_name or "").strip().casefold()
    try:
        return STEM_CONTEXT_NAMES.index(normalized)
    except ValueError:
        return STEM_CONTEXT_NAMES.index("unknown")


def family_id_for_class_name(class_name: str) -> int:
    """楽器クラス名 -> ファミリー ID。対応表に無いものは "other" 扱い。"""
    family = _FAMILY_BY_CLASS_NAME.get(class_name, "other")
    return FAMILY_NAMES.index(family)


def family_ids_for_class_ids(class_ids: tuple[int, ...]) -> tuple[int, ...]:
    """複数の楽器クラスをファミリー ID の集合へ畳む（重複は除去、昇順）。"""
    return tuple(sorted({family_id_for_class_name(INSTRUMENT_CLASSES[class_id]) for class_id in class_ids}))
