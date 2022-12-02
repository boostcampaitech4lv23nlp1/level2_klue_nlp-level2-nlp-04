import pickle as pickle
import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]  # 원핫 인코딩 해주는 부분

    score = np.zeros((30,))
    for c in range(30):  # c는 각 클래스
        targets_c = labels.take([c], axis=1).ravel()  # c에 대한 정답 레이블 확인 (열 요소 추출)
        if targets_c.sum() == 0:  # 해당하는 클래스가 하나도 없다면 skip
            continue
        preds_c = probs.take([c], axis=1).ravel()  # c에 대한 확률(만약 logit이라면?) 레이블 확인 (열 요소 추출)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)  # 세번째 반환 값이 바로 임계값, logit의 범위에 따라 자동으로 반환해주는 듯
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0
