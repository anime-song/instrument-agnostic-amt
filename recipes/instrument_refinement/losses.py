"""学習の loss。6 つの項を重み付きで足す。

- window_class / note_class … 楽器分類の本体。正解が候補集合なので部分ラベル loss を使う。
- family                     … 楽器ファミリーの分類。細かいクラスを外しても方向性を学べる。
- contrastive                … 同じ音源のノート/窓の音色埋め込みを近づける。
                                クラスタリングが効くのはこの項のおかげ。
- consistency                … 同じ音源の別の窓で、予測分布が揃うようにする。
- agreement                  … ノート単位の予測を窓全体の予測に寄せる（単独音源のときだけ）。
"""

from __future__ import annotations

# Losses are recipe concerns and are not imported by runtime inference.

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstrumentRefinementLossConfig:
    """各 loss 項の重み。0 にするとその項を無効化できる。"""

    window_class_weight: float = 1.0
    note_class_weight: float = 0.5
    family_weight: float = 0.3
    contrastive_weight: float = 0.2
    consistency_weight: float = 0.2
    note_window_agreement_weight: float = 0.1
    contrastive_temperature: float = 0.1
    max_contrastive_notes: int = 256

    def __post_init__(self) -> None:
        weights = (
            self.window_class_weight,
            self.note_class_weight,
            self.family_weight,
            self.contrastive_weight,
            self.consistency_weight,
            self.note_window_agreement_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be nonnegative")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")
        if self.max_contrastive_notes <= 0:
            raise ValueError("max_contrastive_notes must be positive")


def partial_label_loss(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """正解が「候補集合」のときの交差エントロピー。

    普通の交差エントロピーは正解 1 つの確率を上げるが、ここでは候補集合に入った
    クラスの確率の合計を上げる（-log Σ p）。候補が 1 つなら通常の交差エントロピーと一致し、
    複数なら「どれか当たっていればよい」という緩い教師になる。

    候補が空の行（=ノートが無い、埋め込み用のダミー）は loss から除外する。
    """
    if logits.shape != target_mask.shape:
        raise ValueError("logits and target_mask must have the same shape")
    valid = target_mask.any(dim=-1)
    if not torch.any(valid):
        return logits.sum() * 0.0
    log_prob = F.log_softmax(logits.float(), dim=-1)
    allowed = log_prob.masked_fill(~target_mask.bool(), -torch.inf)
    losses = -torch.logsumexp(allowed, dim=-1)
    if sample_weight is not None:
        weights = sample_weight.to(device=logits.device, dtype=losses.dtype)
        losses = losses * weights
        return losses[valid].sum() / weights[valid].sum().clamp_min(1e-8)
    return losses[valid].mean()


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    *,
    source_hash: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """同じ音源同士を近づけ、違う音源とは離す教師あり対照学習 (SupCon)。

    「同じクラス」ではなく「同じ音源」を正例にしているのがポイント。同じピアノでも
    録音や奏法で音色は違うので、音源単位でまとめた方が音色空間として素直になり、
    推論時のクラスタリングが効きやすい。
    """
    count = int(embeddings.shape[0])
    if count < 2:
        return embeddings.sum() * 0.0
    values = F.normalize(embeddings.float(), dim=-1)
    similarity = values @ values.transpose(0, 1) / float(temperature)
    eye = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    same_source = source_hash[:, None] == source_hash[None, :]
    positive = same_source & ~eye
    has_positive = positive.any(dim=1)
    if not torch.any(has_positive):
        return embeddings.sum() * 0.0
    # 最大値を引いてから exp する（数値安定化。勾配には影響させないので detach）。
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(similarity) * (~eye).to(similarity.dtype)
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * positive.to(log_prob.dtype)).sum(dim=1) / positive_count
    return per_anchor[has_positive].mean()


def _same_source_consistency(logits: torch.Tensor, source_hash: torch.Tensor) -> torch.Tensor:
    """同じ音源の窓同士で予測分布をそろえる（対称 KL）。

    同じ音源なら、どの窓を切り出しても楽器は同じはず。窓によって判定がぶれるのを
    直接罰することで、推論時の窓ごとの揺れを減らす。
    """
    count = int(logits.shape[0])
    if count < 2:
        return logits.sum() * 0.0
    same = source_hash[:, None] == source_hash[None, :]
    pair_mask = torch.triu(same, diagonal=1)
    left, right = torch.nonzero(pair_mask, as_tuple=True)
    if left.numel() == 0:
        return logits.sum() * 0.0
    log_p = F.log_softmax(logits[left].float(), dim=-1)
    log_q = F.log_softmax(logits[right].float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    return 0.5 * (
        F.kl_div(log_p, q, reduction="batchmean", log_target=False)
        + F.kl_div(log_q, p, reduction="batchmean", log_target=False)
    )


def _note_window_agreement(
    note_logits: torch.Tensor,
    window_logits: torch.Tensor,
    note_mask: torch.Tensor,
) -> torch.Tensor:
    """ノート単位の予測を、窓全体の予測へ寄せる。

    単独音源の窓では全ノートが同じ楽器のはずなので、窓の予測を教師にできる。
    混合サンプルでは成り立たないため、呼び出し側でマスクして除外している。
    """
    if not torch.any(note_mask):
        return note_logits.sum() * 0.0
    window = window_logits[:, None, :].expand_as(note_logits)
    note_log_prob = F.log_softmax(note_logits[note_mask].float(), dim=-1)
    window_prob = F.softmax(window[note_mask].float(), dim=-1)
    return F.kl_div(note_log_prob, window_prob, reduction="batchmean")


def _classification_pair(
    window_logits: torch.Tensor,
    note_logits: torch.Tensor,
    *,
    window_target: torch.Tensor,
    note_target: torch.Tensor,
    note_mask: torch.Tensor,
    window_weight: torch.Tensor,
    note_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """窓単位とノート単位の部分ラベル loss を、同じ組み立てで 2 本返す。

    楽器分類とファミリー分類でまったく同じ形なので共通化してある。
    """
    window_loss = partial_label_loss(window_logits, window_target, sample_weight=window_weight)
    if torch.any(note_mask):
        note_loss = partial_label_loss(
            note_logits[note_mask], note_target[note_mask], sample_weight=note_weights
        )
    else:
        note_loss = note_logits.sum() * 0.0
    return window_loss, note_loss


def _note_contrastive_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    note_mask: torch.Tensor,
    config: InstrumentRefinementLossConfig,
) -> torch.Tensor:
    """ノート側の対照学習。「どの音源か特定できるノート」だけを使う。"""
    usable = note_mask & batch.get("note_contrastive_mask", note_mask).bool()
    if not torch.any(usable):
        return outputs["note_embedding"].sum() * 0.0
    embeddings = outputs["note_embedding"][usable]
    source_hash = batch.get(
        "note_source_hash", batch["source_hash"][:, None].expand_as(note_mask)
    )[usable]
    # ノート数の 2 乗で類似度行列が膨らむので、多すぎる場合は間引く。
    if int(embeddings.shape[0]) > config.max_contrastive_notes:
        selected = torch.randperm(int(embeddings.shape[0]), device=embeddings.device)[
            : config.max_contrastive_notes
        ]
        embeddings = embeddings[selected]
        source_hash = source_hash[selected]
    return _supervised_contrastive_loss(
        embeddings, source_hash=source_hash, temperature=config.contrastive_temperature
    )


def _window_accuracy(
    window_logits: torch.Tensor, primary_class_id: torch.Tensor, fallback: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """精度は「正解が 1 つに確定している窓」だけで測る（混合サンプルは対象外）。"""
    exact = primary_class_id >= 0
    if not torch.any(exact):
        return fallback * 0.0, fallback * 0.0
    logits = window_logits[exact]
    target = primary_class_id[exact]
    accuracy = (logits.argmax(dim=-1) == target).float().mean()
    topk = logits.topk(min(3, int(window_logits.shape[-1])), dim=-1).indices
    return accuracy, (topk == target[:, None]).any(dim=-1).float().mean()


def compute_refinement_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    config: InstrumentRefinementLossConfig = InstrumentRefinementLossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """全 loss 項を計算して合計と内訳を返す。

    Returns:
        (逆伝播する合計 loss, ログ用の内訳と精度)。内訳はすべて detach 済み。
    """
    note_logits = outputs["note_logits"]
    window_logits = outputs["window_logits"]
    note_mask = batch["note_mask"].bool()
    sample_weight = batch["sample_weight"].float()
    class_target = batch["class_target_mask"].bool()
    family_target = batch["family_target_mask"].bool()
    window_weight = batch.get("window_sample_weight", sample_weight)
    note_weights = batch.get("note_sample_weight", sample_weight[:, None].expand_as(note_mask))[note_mask]

    # 1. 楽器分類。窓全体とノート単位の 2 本立て。
    #    混合サンプルは window_weight が 0 なので、窓分類には効かない。
    window_class, note_class = _classification_pair(
        window_logits,
        note_logits,
        window_target=class_target,
        note_target=batch.get("note_class_target_mask", class_target[:, None, :].expand_as(note_logits)),
        note_mask=note_mask,
        window_weight=window_weight,
        note_weights=note_weights,
    )
    # 2. ファミリー分類（補助タスク）。窓とノートの平均を取る。
    note_family_logits = outputs["note_family_logits"]
    window_family, note_family = _classification_pair(
        outputs["window_family_logits"],
        note_family_logits,
        window_target=family_target,
        note_target=batch.get(
            "note_family_target_mask", family_target[:, None, :].expand_as(note_family_logits)
        ),
        note_mask=note_mask,
        window_weight=window_weight,
        note_weights=note_weights,
    )
    family_loss = 0.5 * (window_family + note_family)

    # 3. 音色の対照学習。
    window_contrastive = _supervised_contrastive_loss(
        outputs["window_embedding"],
        source_hash=batch["source_hash"],
        temperature=config.contrastive_temperature,
    )
    note_contrastive = _note_contrastive_loss(outputs, batch, note_mask=note_mask, config=config)
    contrastive = 0.5 * (window_contrastive + note_contrastive)
    # 4. 同一音源の窓間の一貫性。
    consistency = _same_source_consistency(window_logits, batch["source_hash"])
    # 5. ノートと窓の一致。混合サンプルでは「窓 = 1 楽器」が成り立たないので外す。
    agreement_mask = note_mask
    if "is_composite" in batch:
        agreement_mask = agreement_mask & ~batch["is_composite"][:, None]
    agreement = _note_window_agreement(note_logits, window_logits, agreement_mask)
    loss = (
        config.window_class_weight * window_class
        + config.note_class_weight * note_class
        + config.family_weight * family_loss
        + config.contrastive_weight * contrastive
        + config.consistency_weight * consistency
        + config.note_window_agreement_weight * agreement
    )

    exact = batch["primary_class_id"] >= 0
    accuracy, top3 = _window_accuracy(window_logits, batch["primary_class_id"], loss.detach())
    metrics = {
        "loss": loss.detach(),
        "window_class_loss": window_class.detach(),
        "note_class_loss": note_class.detach(),
        "family_loss": family_loss.detach(),
        "contrastive_loss": contrastive.detach(),
        "consistency_loss": consistency.detach(),
        "agreement_loss": agreement.detach(),
        "window_accuracy": accuracy.detach(),
        "window_top3": top3.detach(),
        "note_count": note_mask.sum().detach(),
        "exact_window_count": exact.sum().detach(),
    }
    return loss, metrics
