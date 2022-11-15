import pickle as pickle
import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0



def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels] # 원핫 인코딩 해주는 부분

    score = np.zeros((30,))
    for c in range(30):
            targets_c = labels.take([c], axis=1).ravel()
            preds_c = probs.take([c], axis=1).ravel()
            precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
            score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(logits, labels):
  """ validation을 위한 metrics function """
  
  preds = logits.argmax(-1) # 하나의 값
  probs = logits  # 전체

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels) # 하나의 값만 넣어줌
  auprc = klue_re_auprc(probs, labels) # 전체 값을 넣어줌

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
  }