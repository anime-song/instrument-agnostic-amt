import argparse
import glob
import os
import numpy as np
import pretty_midi
import mir_eval


def extract_notes(midi_path, target_names=None, exclude_names=None):
    """
    MIDIファイルから指定されたトラックのノートを抽出します。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals = []
    pitches = []

    for inst in pm.instruments:
        name = inst.name.lower().strip()

        if exclude_names is not None:
            if any(excl.lower() in name for excl in exclude_names):
                continue

        if target_names is not None:
            if not any(targ.lower() in name for targ in target_names):
                continue

        for note in inst.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)

    if len(intervals) == 0:
        return np.empty((0, 2)), np.empty(0)

    intervals = np.array(intervals)
    pitches = np.array(pitches)

    # Hzに変換
    pitches_hz = mir_eval.util.midi_to_hz(pitches)

    return intervals, pitches_hz


def evaluate_file(ref_path, pred_path):
    # MIR-ST500の正解データは全てのトラックを対象とする（または必要に応じて調整）
    ref_intervals, ref_pitches = extract_notes(ref_path)

    # 推論結果は 'melody' または 'vocal' のみ対象とし、'vocal_harmony' を除外する
    pred_intervals, pred_pitches = extract_notes(
        pred_path, target_names=["melody", "vocal"], exclude_names=["vocal_harmony"]
    )

    if len(ref_intervals) == 0 and len(pred_intervals) == 0:
        # 正解も予測も空の場合は満点とする
        return {"COnPOff_F1": 1.0, "COnP_F1": 1.0, "COn_F1": 1.0}
    elif len(ref_intervals) == 0 or len(pred_intervals) == 0:
        # どちらかが空の場合は0点
        return {"COnPOff_F1": 0.0, "COnP_F1": 0.0, "COn_F1": 0.0}

    # COnPOff (Onset, Pitch, Offset)
    _, _, conpoff_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        pitch_tolerance=50.0,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )

    # COnP (Onset, Pitch) - offset_ratio=None に設定
    _, _, conp_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        pitch_tolerance=50.0,
        offset_ratio=None,
    )

    # COn (Onset) - onset_precision_recall_f1 を使用
    _, _, con_f1 = mir_eval.transcription.onset_precision_recall_f1(
        ref_intervals, pred_intervals, onset_tolerance=0.05
    )

    return {"COnPOff_F1": conpoff_f1, "COnP_F1": conp_f1, "COn_F1": con_f1}


def main():
    parser = argparse.ArgumentParser(
        description="ボーカル採譜モデルのF1評価を行います。"
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="正解データ(MIR-ST500)のMIDIディレクトリ",
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True, help="推論結果のMIDIディレクトリ"
    )

    args = parser.parse_args()

    # pred_dir のファイル一覧を取得
    pred_files = glob.glob(os.path.join(args.pred_dir, "*.mid"))
    pred_files += glob.glob(os.path.join(args.pred_dir, "*.midi"))

    if not pred_files:
        print(f"推論結果のファイルが {args.pred_dir} に見つかりません。")
        return

    results = []

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        # MIR-ST500の正解ファイルと同名であると仮定（必要に応じて拡張子等の処理を調整）
        ref_path = os.path.join(args.ref_dir, filename)

        if not os.path.exists(ref_path):
            # 拡張子が違った場合のフォールバック (.mid <-> .midi など)
            base, ext = os.path.splitext(filename)
            alt_ext = ".midi" if ext.lower() == ".mid" else ".mid"
            ref_path_alt = os.path.join(args.ref_dir, base + alt_ext)
            if os.path.exists(ref_path_alt):
                ref_path = ref_path_alt
            else:
                print(f"[スキップ] 対応する正解ファイルが見つかりません: {ref_path}")
                continue

        try:
            metrics = evaluate_file(ref_path, pred_path)
            metrics["filename"] = filename
            results.append(metrics)
            print(
                f"[{filename}] COnPOff: {metrics['COnPOff_F1']:.4f}, COnP: {metrics['COnP_F1']:.4f}, COn: {metrics['COn_F1']:.4f}"
            )
        except Exception as e:
            print(f"[{filename}] 評価中にエラーが発生しました: {e}")

    if not results:
        print("評価可能なファイルがありませんでした。")
        return

    # 平均を計算
    avg_conpoff = np.mean([r["COnPOff_F1"] for r in results])
    avg_conp = np.mean([r["COnP_F1"] for r in results])
    avg_con = np.mean([r["COn_F1"] for r in results])

    print("\n" + "=" * 40)
    print("全体評価結果 (Average F1 Score)")
    print("=" * 40)
    print(f"Total files : {len(results)}")
    print(f"COnPOff F1  : {avg_conpoff:.4f}")
    print(f"COnP F1     : {avg_conp:.4f}")
    print(f"COn F1      : {avg_con:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
