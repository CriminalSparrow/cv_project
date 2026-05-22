"""Модуль с функциями для расчета метрик качества рекомендаций"""

import numpy as np


def average_precision_at_k_for_one(y_true_row, y_prob_row, k=5):
    topk = np.argsort(y_prob_row)[-k:][::-1]

    num_relevant_total = int(y_true_row.sum())
    if num_relevant_total == 0:
        return None

    denom = min(num_relevant_total, k)

    hits = 0
    precision_sum = 0.0

    for rank, idx in enumerate(topk, start=1):
        if y_true_row[idx] == 1:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / denom


def map_at_k(y_true, y_prob, k=5):
    values = []

    for i in range(y_true.shape[0]):
        ap = average_precision_at_k_for_one(
            y_true_row=y_true[i],
            y_prob_row=y_prob[i],
            k=k,
        )

        if ap is not None:
            values.append(ap)

    return float(np.mean(values)) if values else 0.0


def compute_topk_multilabel_metrics(y_true, y_prob, k=5):
    num_classes = y_prob.shape[1]
    k = min(k, num_classes)

    topk_indices = np.argsort(y_prob, axis=1)[:, -k:][:, ::-1]

    precisions = []
    recalls = []
    hits = []
    average_precisions = []

    for i in range(y_true.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0])

        if len(true_set) == 0:
            continue

        topk = topk_indices[i]
        pred_set = set(topk)

        intersection = true_set & pred_set

        precision = len(intersection) / k
        recall = len(intersection) / len(true_set)
        hit = float(len(intersection) > 0)

        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)

        # AP@K: учитывает позиции релевантных треков
        num_hits = 0
        precision_sum = 0.0
        denom = min(len(true_set), k)

        for rank, idx in enumerate(topk, start=1):
            if idx in true_set:
                num_hits += 1
                precision_sum += num_hits / rank

        average_precisions.append(precision_sum / denom)

    precision_at_k = float(np.mean(precisions)) if precisions else 0.0
    recall_at_k = float(np.mean(recalls)) if recalls else 0.0
    hit_at_k = float(np.mean(hits)) if hits else 0.0
    map_at_k_value = float(np.mean(average_precisions)) if average_precisions else 0.0

    if precision_at_k + recall_at_k == 0:
        f1_at_k = 0.0
    else:
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)

    return {
        f"precision@{k}": precision_at_k,
        f"recall@{k}": recall_at_k,
        f"hit@{k}": hit_at_k,
        f"f1@{k}": f1_at_k,
        f"map@{k}": map_at_k_value,
    }