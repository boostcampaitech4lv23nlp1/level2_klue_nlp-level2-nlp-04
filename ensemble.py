import numpy as np
import pandas as pd
from ast import literal_eval
import Utils.labels_ids as labels_ids
import os


# 모든 prob 리스트 받아오기
def soft_vote_ensemble(file_list, weight=None):
    if len(file_list) == 0:
        print("파일 리스트 확인!")
        exit(1)
    if weight is None:
        weight = [1 / len(file_list)] * len(file_list)
    if sum(weight) != 1:
        weight = [i / sum(weight) for i in weight]
    if len(file_list) != len(weight):
        print("가중치 확인!")
        exit(1)

    total_prob_list = []
    for i in file_list:
        data_set = pd.read_csv(f"./prediction/{i}.csv")
        probs = data_set["probs"].to_list()
        int_prob_list = []
        for i in range(len(probs)):
            int_prob_list.append(literal_eval(probs[i]))

        total_prob_list.append(np.array(int_prob_list))

    # 평균 prob 구하기
    average_prob = None
    for idx, value in enumerate(total_prob_list):
        if average_prob is None:
            average_prob = weight[idx] * value
        else:
            average_prob += weight[idx] * value

    # prob를 통해서 pred 구하기
    average_pred = [np.argmax(logit, axis=-1).item() for logit in average_prob]
    pred_answer = labels_ids.num_to_label(average_pred)

    average_prob = [list(i) for i in average_prob]  # 리스트 형태로 변환

    # output 만들고 저장하기
    predict_id = range(len(probs))  # id
    output = pd.DataFrame(
        {
            "id": predict_id,
            "pred_label": pred_answer,
            "probs": average_prob,
        }
    )

    # 만약 prediction 폴더가 없다면 생성합니다
    dir_path = "./prediction"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    output.to_csv(f"./prediction/submission_ensemble.csv", index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장


file_list = []  # .csv파일을 제외한 이름을 입력합니다

# 가중치가 있다면 가중평균을 없다면 산술평균을 구합니다. 가중치의 합이 1이 아니라면 합으로 나눠 합이 1로 다시 재조정 됩니다
soft_vote_ensemble(file_list=file_list)
