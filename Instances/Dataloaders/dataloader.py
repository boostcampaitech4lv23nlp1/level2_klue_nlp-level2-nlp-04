import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from Instances.Dataloaders.dataset import RE_Dataset
import Utils.labels_ids as labels_ids
from ast import literal_eval

# (train+dev), test, predict  # train 데이터의 일부를 dev 데이터 셋으로 사용합니다
class Dataloader(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name  # 토크나이저를 받기 위한 backbone 모델의 이름
        self.batch_size = conf.train.batch_size  # 배치 사이즈
        self.shuffle = conf.data.shuffle  # shuffle 유무
        self.train_ratio = conf.data.train_ratio  # train과 dev 셋의 데이터 떼올 양
        self.seed = conf.utils.seed  # seed
        self.entity_marker_type = conf.data.entity_marker_type  # 엔티티 위치 표현 유형

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

        special_tokens = self.find_special_token(self.entity_marker_type)  # ({"additional_special_tokens":['aba','adsf']})
        self.new_special_token_count = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def find_special_token(self, entity_marker_type):
        special_tokens = []
        subject_entity_types = ["PER", "ORG"]
        object_entity_types = ["PER", "ORG", "LOC", "DAT", "POH", "NOH"]
        if entity_marker_type == "typed_entity_marker":
            for i in subject_entity_types:
                for j in ["", "/"]:
                    special_tokens.append(f"[{j}SUBJ-{i}]")
            for i in object_entity_types:
                for j in ["", "/"]:
                    special_tokens.append(f"[{j}OBJ-{i}]")

        elif entity_marker_type == "typed_entity_marker_punct":
            special_tokens = ["@", "#"]

        return special_tokens

    def tokenizing(self, dataframe, entity_marker_type):
        """
        entity_marker_type:
            - baseline : entity_marker_type 미사용
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        concat_entity = []

        for sent, subj, subj_start, subj_end, subj_type, obj, obj_start, obj_end, obj_type in zip(
            dataframe["sentence"], dataframe["subject_entity"], dataframe["subject_start"], dataframe["subject_end"], dataframe["subject_type"], dataframe["object_entity"], dataframe["object_start"], dataframe["object_end"], dataframe["object_type"]
        ):
            if entity_marker_type == "typed_entity_marker":
                temp_subj_type_start = f"[SUBJ-{str(subj_type)}]"
                temp_subj_type_end = f"[/SUBJ-{str(subj_type)}]"
                temp_obj_type_start = f"[OBJ-{str(obj_type)}]"
                temp_obj_type_end = f"[/OBJ-{str(obj_type)}]"

                temp_subj = f"{temp_subj_type_start} {str(subj)} {temp_subj_type_end}"
                temp_obj = f"{temp_obj_type_start} {str(obj)} {temp_obj_type_end}"

            elif entity_marker_type == "typed_entity_marker_punct":
                temp_subj = f"@ * {str(subj_type)} * {str(subj)} @"
                temp_obj = f"# ^ {str(obj_type)} ^ {str(obj)} #"

            elif entity_marker_type == "baseline":
                temp_subj = str(subj)
                temp_obj = str(obj)

            if subj_start < obj_start:
                sent = sent[:subj_start] + temp_subj + sent[subj_end + 1 : obj_start] + temp_obj + sent[obj_end + 1 :]
            else:
                sent = sent[:obj_start] + temp_obj + sent[obj_end + 1 : subj_start] + temp_subj + sent[subj_end + 1 :]

            sents.append(sent)
            concat_entity.append(str(subj) + self.tokenizer.sep_token + str(obj))

        tokenized_sentences = self.tokenizer(
            concat_entity,
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        return tokenized_sentences

    # predict 빼고 전부 동일한 전처리
    def preprocessing(self, dataframe):  # 전체 전처리 과정을 모두 거쳐서 dataset input 형태로 구성할 수 있도록 하고 predict일 땐 빈 배열 반환
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity, subject_start, subject_end, subject_type = [], [], [], []
        object_entity, object_start, object_end, object_type = [], [], [], []

        for s, o in zip(dataframe["subject_entity"], dataframe["object_entity"]):
            s_dict = literal_eval(s)
            o_dict = literal_eval(o)

            subject_entity.append(s_dict["word"])
            subject_start.append(s_dict["start_idx"])
            subject_end.append(s_dict["end_idx"])
            subject_type.append(s_dict["type"])

            object_entity.append(o_dict["word"])
            object_start.append(o_dict["start_idx"])
            object_end.append(o_dict["end_idx"])
            object_type.append(o_dict["type"])

        entity_dataset = pd.DataFrame(
            {
                "subject_entity": subject_entity,
                "subject_start": subject_start,
                "subject_end": subject_end,
                "subject_type": subject_type,
                "object_entity": object_entity,
                "object_start": object_start,
                "object_end": object_end,
                "object_type": object_type,
            }
        )
        entity_dataset.reset_index(drop=True, inplace=True)
        all_dataset = dataframe[["id", "sentence", "label"]]
        all_dataset.reset_index(drop=True, inplace=True)
        preprocessing_dataframe = pd.concat([all_dataset, entity_dataset], axis=1)

        # 현재 train_dataset = load_data("../dataset/train/train.csv")까지 거친 상태

        if type(preprocessing_dataframe["label"].values[0]) == str:  # train, dev, test
            labels = labels_ids.label_to_num(preprocessing_dataframe["label"].values)  # labels를 붙여줍니다
        else:  # predict
            labels = preprocessing_dataframe["label"].values
        inputs = self.tokenizing(preprocessing_dataframe, self.entity_marker_type)  # input 데이터를 토큰화해줍니다

        return inputs, labels  # 전처리한 inputs와 labels 반환합니다

    def setup(self, stage="fit"):  # train, dev, test는 모두 동일한 전처리 과정을 거칩니다
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)
            split = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.train_ratio, random_state=self.seed)
            for train_idx, val_idx in split.split(total_data, total_data["label"]):
                train_data = total_data.loc[train_idx]
                val_data = total_data.loc[val_idx]

            # train_data = total_data.sample(frac=self.train_ratio)  # csv 파일을 불러서 train과 dev로 나눕니다, 기본 baseline의 load_data 과정
            train_inputs, train_labels = self.preprocessing(train_data)
            self.train_dataset = RE_Dataset(train_inputs, train_labels)

            # val_data = total_data.drop(train_data.index)  # dev
            val_inputs, val_labels = self.preprocessing(val_data)
            self.val_dataset = RE_Dataset(val_inputs, val_labels)

            print("train data len : ", len(train_labels))
            print("valid data len : ", len(val_labels))

        else:  # inference 완성 오늘 할 애정
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_labels = self.preprocessing(test_data)
            self.test_dataset = RE_Dataset(test_inputs, test_labels)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_labels = self.preprocessing(predict_data)

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
        return self.new_token_count + self.new_special_token_count + self.tokenizer.vocab_size  # 새로운 vocab 사이즈를 반환합니다
