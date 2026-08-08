"""Post-AMT instrument refinement from separated audio and predicted MIDI.

AMT 本体が出した MIDI の「楽器の割り当てだけ」を、分離済みステム音声を聴き直して
付け直すための後処理モデル一式。ノートの時刻・音高は一切変更しない。

処理の流れ（推論時）::

    ステム音声 (wav) + AMT の MIDI
        -> data/midi.py         MIDI をノート表 (RefinementNoteTable) に変換
        -> modeling/model.py    窓ごとに「ノート単位のロジット + 音色埋め込み」を推論
        -> inference/refine.py  窓をまたいで平均し、音色でクラスタリングして楽器を決定
        -> 楽器を振り直した MIDI

各サブパッケージの役割:

- ``data``      MIDI/データセットの読み込みと、楽器クラス・ステム名の対応表
- ``modeling``  モデル本体と、AMT backbone の流用/チェックポイント入出力
- ``training``  学習用の Dataset・collate・loss
- ``inference`` 推論本体（窓の集約、クラスタリング、MIDI 書き戻し）
- ``cli``       コマンドラインの入口（manifest 作成 / 学習 / 推論）

ドラムは対象外である点に注意。楽器候補からドラムを必ず除外するため、ドラムステムを
渡すと候補が空になる（呼び出し側でスキップする必要がある）。
"""

from .modeling.model import InstrumentRefinementConfig, InstrumentRefinementModel

__all__ = ["InstrumentRefinementConfig", "InstrumentRefinementModel"]
