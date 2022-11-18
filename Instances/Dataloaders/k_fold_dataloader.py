import pandas as pd
import pytorch_lightning as pl
import torch
import transformers

from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from Instances.Dataloaders.dataset import RE_Dataset

import Utils.labels_ids as labels_ids

# (train+dev), test, predict  # train 데이터의 일부를 dev 데이터 셋으로 사용합니다
class KFoldDataloader(pl.LightningDataModule):
    def __init__(self, conf, k):
        super().__init__()
        self.model_name = conf.model.model_name  # 토크나이저를 받기 위한 backbone 모델의 이름
        self.batch_size = conf.train.batch_size  # 배치 사이즈
        self.shuffle = conf.data.shuffle  # shuffle 유무
        self.k = k  # 현재 해당하는 dataloader
        self.num_split = conf.k_fold.num_folds
        self.seed = conf.utils.seed  # seed

        self.train_path = conf.path.train_path  # train+dev data set 경로
        self.test_path = conf.path.test_path  # test data set 경로
        self.predict_path = conf.path.predict_path  # predict data set 경로

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer
        # deadlock에 걸리는 경우가 존재해서 use_fast를 False로 둠
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # https://www.youtube.com/watch?v=7q5NyFT8REg
        # https://huggingface.co/course/chapter3/2?fw=pt
        self.data_collator = transformers.DataCollatorWithPadding(self.tokenizer)  # 다이나믹 패딩 유튜브 -> 잘안되는거 같음 (train 237로만 잘르고 dev 241)

        tokens = ['""']  # 추가할 토큰들 지정 ex) "" 토큰
        self.new_token_count = self.tokenizer.add_tokens(tokens)  # vocab에 추가를 하며 실제로 새롭게 추가된 토큰의 수를 반환해줍니다.

    def tokenizing(self, dataframe):
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataframe["subject_entity"], dataframe["object_entity"]):
            temp = ""
            temp = e01 + self.tokenizer.sep_token + e02  # [SEP] -> self.tokenizer.sep_token
            concat_entity.append(temp)
        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataframe["sentence"]),  # [CLS](sub[SEP]object_entity)[SEP]sentence[SEP] 형태, sentence 부분이 1로, 나머지는 0으로 세그먼트 임베딩이 될 것 같습니다
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        return tokenized_sentences

    # predict 빼고 전부 동일한 전처리
    def preprocessing(self, dataframe, labels_exist=True):  # 전체 전처리 과정을 모두 거쳐서 dataset input 형태로 구성할 수 있도록 하고 predict일 땐 빈 배열 반환
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i, j in zip(dataframe["subject_entity"], dataframe["object_entity"]):
            i = i[1:-1].split(",")[0].split(":")[1]
            j = j[1:-1].split(",")[0].split(":")[1]

            subject_entity.append(i)
            object_entity.append(j)

        # preprocessing을 거쳐서 subject_entity와 object_entity 형태로 넣어줍니다
        preprocessing_dataframe = pd.DataFrame(
            {
                "id": dataframe["id"],
                "sentence": dataframe["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "label": dataframe["label"],
            }
        )

        # 현재 train_dataset = load_data("../dataset/train/train.csv")까지 거친 상태

        if labels_exist:  # train, dev, test
            labels = labels_ids.label_to_num(preprocessing_dataframe["label"].values)  # labels를 붙여줍니다
        else:  # predict
            labels = preprocessing_dataframe["label"].values
        inputs = self.tokenizing(preprocessing_dataframe)  # input 데이터를 토큰화해줍니다

        return inputs, labels  # 전처리한 inputs와 labels 반환합니다

    def setup(self, stage="fit"):  # train, dev, test는 모두 동일한 전처리 과정을 거칩니다
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)
            kfold = KFold(n_splits=self.num_split, shuffle=self.shuffle, random_state=self.seed)
            all_splits = [d_i for d_i in kfold.split(total_data)]

            # 데이터 split
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            print("Number of splits: \n", self.num_split)
            print("Fold: \n", self.k)
            print("Train data len: \n", len(train_indexes))
            print("Valid data len: \n", len(val_indexes))

            train_inputs, train_labels = self.preprocessing(total_data.loc[train_indexes])
            val_inputs, val_labels = self.preprocessing(total_data.loc[val_indexes])

            self.train_dataset = RE_Dataset(train_inputs, train_labels)
            self.val_dataset = RE_Dataset(val_inputs, val_labels)

            print("train data len : ", len(train_labels))
            print("valid data len : ", len(val_labels))

        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_labels = self.preprocessing(test_data)
            self.test_dataset = RE_Dataset(test_inputs, test_labels)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_labels = self.preprocessing(predict_data, False)  # predict는 label이 없으므로 False를 넘겨줍니다

            self.predict_dataset = RE_Dataset(predict_inputs, predict_labels)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.data_collator)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

    def new_vocab_size(self):  # 임베딩 사이즈를 맞춰줘야함 -> 5개추가 원래의 vocab + 5
        return self.new_token_count + self.tokenizer.vocab_size  # 새로운 vocab 사이즈를 반환합니다
