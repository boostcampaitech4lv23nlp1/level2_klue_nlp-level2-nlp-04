import pickle as pickle
import os
import pandas as pd
import torch


# 여기도 나중에 바꿀 수 있음
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):  # 총 데이터 셋의 길이를 반환합니다
        return len(self.labels)
