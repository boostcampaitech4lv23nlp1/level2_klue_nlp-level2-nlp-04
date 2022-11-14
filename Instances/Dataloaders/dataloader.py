import pickle as pickle

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

import Utils.utils as utils
from Instances.Dataloaders.dataset import RE_Dataset


# (train+dev), test, predict  # train 데이터의 일부를 dev 데이터 셋으로 사용합니다
class Dataloader(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.model_name = conf.model.model_name  # backbone 모델
        self.batch_size = conf.train.batch_size  # 배치 사이즈
        self.shuffle = conf.data.shuffle  # shuffle 유무
        self.train_ratio = conf.data.train_ratio  # train과 dev 셋의 데이터 떼올 양
        self.seed = conf.utils.seed  # seed

        self.train_path = conf.path.train_path  # train+dev data set 경로
        self.test_path = conf.path.test_path # test data set 경로
        self.predict_path = conf.path.predict_path  # predict data set 경로

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer
        # deadlock에 걸리는 경우가 존재해서 use_fast를 False로 둠
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        self.tokenizer.model_max_length = 256 # 곧 다이나믹 패딩으로 바꿀 예정

        tokens = ['""']  # 추가할 토큰들 지정 ex) "" 토큰
        self.new_token_count = self.tokenizer.add_tokens(tokens)  # vocab에 추가를 하며 실제로 새롭게 추가된 토큰의 수를 반환해줍니다.

    def label_to_num(self, label):
        """
            문자열 라벨을 숫자로 변환 합니다.
        """
        num_label = []
        with open('dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
        for v in label:
            num_label.append(dict_label_to_num[v])
        
        return num_label
    
    def num_to_label(self, label):
        """
            숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
        """
        origin_label = []
        with open('dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
        for v in label:
            origin_label.append(dict_num_to_label[v])
        
        return origin_label


    def tokenizing(self, dataframe):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataframe['subject_entity'], dataframe['object_entity']):
            temp = ''
            temp = e01 + self.tokenizer.sep_token + e02  # [SEP] -> self.tokenizer.sep_token
            concat_entity.append(temp)
        tokenized_sentences =self.tokenizer(
            concat_entity,
            list(dataframe['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            )
        return tokenized_sentences

    # predict 빼고 전부 동일한 전처리
    def preprocessing(self, dataframe, labels_exist=True): #전체 전처리 과정을 모두 거쳐서 dataset input 형태로 구성할 수 있도록 하고 predict일 땐 빈 배열 반환ㄴ
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i,j in zip(dataframe['subject_entity'], dataframe['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        
        # preprocessing을 거쳐서 subject_entity와 object_entity 형태로 넣어줍니다
        preprocessing_dataframe = pd.DataFrame({'id':dataframe['id'], 'sentence':dataframe['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataframe['label'],})
        
        # 현재 train_dataset = load_data("../dataset/train/train.csv")까지 거친 상태
        
        if labels_exist: #train, dev, test
            labels = self.label_to_num(preprocessing_dataframe['label'].values) # labels를 붙여줍니다
        else: # predict
            labels=[] 
        inputs = self.tokenizing(preprocessing_dataframe) # input 데이터를 토큰화해줍니다
        
        return inputs, labels # 전처리한 inputs와 labels 반환합니다

    def setup(self, stage="fit"): # train, dev, test는 모두 동일한 전처리 과정을 거칩니다
        if stage == "fit":
            total_data = pd.read_csv(self.train_path)

           
            train_data = total_data.sample(frac=self.train_ratio)  # csv 파일을 불러서 train과 dev로 나눕니다, 기본 baseline의 load_data 과정
            train_inputs, train_labels= self.preprocessing(train_data) 
            self.train_dataset = RE_Dataset(train_inputs, train_labels)
            
            
            val_data = total_data.drop(train_data.index)
            val_inputs, val_labels = self.preprocessing(val_data)
            self.val_dataset = RE_Dataset(val_inputs, val_labels)
           
        
            print("train data len : ", len(train_labels))
            print("valid data len : ", len(val_labels))

            
            
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_labels = self.preprocessing(test_data)
            self.test_dataset = RE_Dataset(test_inputs, test_labels)
        
            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data, False)
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def new_vocab_size(self):
        return self.new_token_count + self.tokenizer.vocab_size # 새로운 vocab 사이즈를 반환합니다
