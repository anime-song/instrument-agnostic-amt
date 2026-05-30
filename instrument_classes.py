import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# このファイルと同じディレクトリ、またはプロジェクトルートにある
# instrument_merge.json を読む。
_MERGE_JSON_PATH = Path(__file__).parent / "instrument_merge.json"
if not _MERGE_JSON_PATH.exists():
    _MERGE_JSON_PATH = Path(__file__).parent.parent / "instrument_merge.json"

# 既存の33楽器クラスは sorted(labels.keys()) のIDで保存済みnpzに入っている。
# 新しい非GMクラスは末尾に追加して、既存IDをずらさない。
EXTRA_INSTRUMENT_CLASS_PROGRAMS = {
    "melody": 65,  # 推論結果をMIDIに戻すときの代表音色は Alto Sax にする。
    "vocal_harmony": 65,
}

INSTRUMENT_CLASSES: List[str] = []
NUM_INSTRUMENT_CLASSES: int = 0
_PROGRAM_TO_CLASS_ID: Dict[int, int] = {}
_CLASS_NAME_TO_ID: Dict[str, int] = {}
_CLASS_ID_TO_PROGRAM: Dict[int, int] = {}
_DRUM_CLASS_ID: Optional[int] = None


def _load_instrument_mappings() -> None:
    global INSTRUMENT_CLASSES
    global NUM_INSTRUMENT_CLASSES
    global _PROGRAM_TO_CLASS_ID
    global _CLASS_NAME_TO_ID
    global _CLASS_ID_TO_PROGRAM
    global _DRUM_CLASS_ID

    INSTRUMENT_CLASSES = []
    NUM_INSTRUMENT_CLASSES = 0
    _PROGRAM_TO_CLASS_ID = {}
    _CLASS_NAME_TO_ID = {}
    _CLASS_ID_TO_PROGRAM = {}
    _DRUM_CLASS_ID = None

    if not _MERGE_JSON_PATH.exists():
        logger.warning(
            "Could not find %s. Instrument mapping may be incomplete.",
            _MERGE_JSON_PATH,
        )
        return

    with open(_MERGE_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = data.get("labels", {})

    # ここは既存のID互換性に関わるため、既存クラスの並びは変えない。
    # melody は instrument_merge.json にない特殊クラスとして末尾に足す。
    INSTRUMENT_CLASSES = sorted(labels.keys())
    for class_name in EXTRA_INSTRUMENT_CLASS_PROGRAMS:
        if class_name not in labels:
            INSTRUMENT_CLASSES.append(class_name)

    NUM_INSTRUMENT_CLASSES = len(INSTRUMENT_CLASSES)
    _CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(INSTRUMENT_CLASSES)}

    for class_name, class_info in labels.items():
        class_id = _CLASS_NAME_TO_ID[class_name]
        for member in class_info.get("members", []):
            if member.get("instrument_key") == "drums" or class_name == "drums":
                _DRUM_CLASS_ID = class_id

            prog_num = member.get("program_number")
            if prog_num is None:
                continue

            prog_num = int(prog_num)
            _PROGRAM_TO_CLASS_ID[prog_num] = class_id
            _CLASS_ID_TO_PROGRAM.setdefault(class_id, prog_num)

    # melody はGM programから自動判定するクラスではない。
    # prepare_dataset.py 側でトラック名を見て明示的にこのクラスへ割り当てる。
    for class_name, program_number in EXTRA_INSTRUMENT_CLASS_PROGRAMS.items():
        class_id = _CLASS_NAME_TO_ID.get(class_name)
        if class_id is not None:
            _CLASS_ID_TO_PROGRAM[class_id] = program_number


_load_instrument_mappings()


def get_instrument_class_id(program_number: int, is_drum: bool = False) -> int:
    """
    GM program number と drum フラグから楽器クラスIDを返す。
    """
    if is_drum and _DRUM_CLASS_ID is not None:
        return _DRUM_CLASS_ID

    class_id = _PROGRAM_TO_CLASS_ID.get(program_number)
    if class_id is not None:
        return class_id

    logger.debug(
        "program_number %s not found in instrument_merge.json; using class 0.",
        program_number,
    )
    return 0


def get_instrument_class_id_by_name(class_name: str) -> int:
    """
    "melody" など、クラス名から楽器クラスIDを返す。
    """
    class_id = _CLASS_NAME_TO_ID.get(class_name)
    if class_id is None:
        raise KeyError(f"Unknown instrument class: {class_name}")
    return class_id


def get_program_number_from_class_id(class_id: int) -> int:
    """
    クラスIDから、MIDI出力時に使う代表GM program numberを返す。
    """
    if class_id == _DRUM_CLASS_ID:
        return 0

    return _CLASS_ID_TO_PROGRAM.get(class_id, 0)
