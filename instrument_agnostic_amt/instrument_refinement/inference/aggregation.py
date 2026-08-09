"""窓ごと・ノートごとのモデル出力を、扱いやすい単位へまとめ直すための道具。

- ``viterbi_smooth_classes``: 時間方向に楽器がぶれるのを抑えて 1 本の系列にする。
- ``cluster_note_embeddings``: 音色が近いノートを同じグループにまとめる。
- ``top_candidates``: ロジットを候補クラスだけの確率に直し、上位を返す。
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def viterbi_smooth_classes(
    logits: np.ndarray,
    *,
    switch_penalty: float = 2.0,
) -> np.ndarray:
    """Decode a piecewise-constant class sequence from window logits.

    窓ごとに argmax を取ると 1 窓だけ別楽器に飛ぶような揺れが出る。楽器が切り替わる
    たびに switch_penalty を課す Viterbi で、区間ごとに一定な系列へならす。
    switch_penalty が大きいほど切り替わりにくい。
    """

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [windows, classes]")
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if switch_penalty < 0.0:
        raise ValueError("switch_penalty must be nonnegative")
    # 窓ごとに log_softmax する（最大値を引いてから正規化＝オーバーフロー対策）。
    shifted = values - values.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    window_count, class_count = log_prob.shape
    score = np.empty_like(log_prob)
    back = np.zeros((window_count, class_count), dtype=np.int64)
    score[0] = log_prob[0]
    # 遷移コスト: 同じクラスに留まれば 0、別クラスへ移ると -switch_penalty。
    transition = np.full((class_count, class_count), -float(switch_penalty))
    np.fill_diagonal(transition, 0.0)
    # 前向き: 各窓・各クラスについて最良スコアと、その直前クラス (back) を記録する。
    for index in range(1, window_count):
        candidates = score[index - 1][:, None] + transition
        back[index] = candidates.argmax(axis=0)
        score[index] = log_prob[index] + candidates.max(axis=0)
    # 後ろ向き: 最終窓の最良クラスから back をたどって経路を復元する。
    path = np.zeros(window_count, dtype=np.int64)
    path[-1] = int(score[-1].argmax())
    for index in range(window_count - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return path


def cluster_note_embeddings(
    embeddings: np.ndarray,
    *,
    distance_threshold: float = 0.25,
    min_cluster_notes: int = 2,
) -> np.ndarray:
    """Agglomerative cosine clustering with deterministic small-cluster merging.

    音色埋め込みのコサイン距離で階層クラスタリングし、distance_threshold より近い
    ノートを同じ楽器のグループとみなす。min_cluster_notes 未満の小さすぎるクラスタは
    ノイズ扱いし、最も音色が近い大きいクラスタへ吸収する（結果は入力順に決定的）。
    """

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape [notes, dimensions]")
    note_count = int(values.shape[0])
    if note_count == 0:
        return np.zeros(0, dtype=np.int64)
    # ノートが 1 つ、またはしきい値 0 なら分けようがないので全部同じクラスタ。
    if note_count == 1 or distance_threshold <= 0.0:
        return np.zeros(note_count, dtype=np.int64)
    # ゼロベクトル（観測なしなど）はコサイン距離が定義できないので、
    # 有効なベクトルの平均で埋めてから正規化する。
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 1e-8
    if not np.any(nonzero):
        return np.zeros(note_count, dtype=np.int64)
    fallback = values[nonzero].mean(axis=0)
    values = np.where(nonzero[:, None], values, fallback[None, :])
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)
    distances = pdist(values, metric="cosine")
    if not np.all(np.isfinite(distances)):
        return np.zeros(note_count, dtype=np.int64)
    # fcluster のラベルは 1 始まりなので 0 始まりに直す。
    labels = (
        fcluster(
            linkage(distances, method="average"),
            t=float(distance_threshold),
            criterion="distance",
        ).astype(np.int64)
        - 1
    )
    if min_cluster_notes <= 1:
        return _reindex(labels)

    # 小さすぎるクラスタを、音色が最も近い（内積が最大の）大クラスタへ移す。
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= int(min_cluster_notes)]
    if large.size == 0:
        # 全部が小クラスタなら分割をあきらめ、1 つにまとめる。
        return np.zeros(note_count, dtype=np.int64)
    centroids = {int(label): values[labels == label].mean(axis=0) for label in large.tolist()}
    for label, count in zip(unique.tolist(), counts.tolist()):
        if count >= int(min_cluster_notes):
            continue
        members = np.flatnonzero(labels == label)
        for member in members:
            target = max(
                centroids,
                key=lambda candidate: float(values[member] @ centroids[candidate]),
            )
            labels[member] = int(target)
    return _reindex(labels)


def _reindex(labels: np.ndarray) -> np.ndarray:
    """吸収などで歯抜けになったラベルを 0, 1, 2, ... に振り直す。"""
    mapping = {int(value): index for index, value in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def top_candidates(
    logits: np.ndarray,
    *,
    allowed_class_ids: tuple[int, ...],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """候補クラスだけで softmax を取り、確率の高い順に (クラス ID, 確率) を返す。"""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    # 候補外を含めずに正規化するので、返る確率は候補集合の中での比率になる。
    allowed = np.asarray(allowed_class_ids, dtype=np.int64)
    selected = values[allowed]
    selected -= selected.max()
    probabilities = np.exp(selected)
    probabilities /= probabilities.sum().clip(min=1e-12)
    order = np.argsort(-probabilities)[: max(1, int(top_k))]
    return [(int(allowed[index]), float(probabilities[index])) for index in order.tolist()]
