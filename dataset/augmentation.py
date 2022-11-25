import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import random
import string
from konlpy.tag import Mecab
import hgtk
from collections import defaultdict
from changer import Changer
from omegaconf import OmegaConf
import argparse
import time
from py_hanspell.hanspell import spell_checker
mecab = Mecab()

"""
Data Augmentation For Relation Extraction task

RE task는 문장의 주체(subject)와 객체(object) 간의 관계를 예측하는 task입니다.
만약 RE dataset 원본을 변형하여 증강하고자 한다면, 주체와 객체를 나타내는 단어는 변형되면 안 됩니다. 
그러면서도 원본 문장과 너무 같지 않은 문장을 만들어내야 합니다.

Augmentation은 다음과 같은 과정을 거칩니다.
1. 반말 -> 존댓말 변형
2. 문장 내 부사 -> 사전 뜻풀이로 교체
   원 문장에 부사가 없을 경우에는 문장의 첫 위치에 랜덤 부사 삽입 
4. AEDA 적용

"""

if __name__ == "__main__":
    # python augmentation.py -c dataset_xlmr_base_config
    print('start')
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")
    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"../Config/{args.config}.yaml")

    original_df = pd.read_csv('../'+conf.path.train_path)
    aug_df = label_filtering(original_df)
    aug_df = sentence_with_entity_marker(aug_df)

    # 반말 -> 존댓말
    honorific_model = Changer()
    aug_df['honorific'] =  aug_df['sentence_with_entity_marker'].apply(lambda x: informal_to_honorific(x, honorific_model))

    # entity check
    aug_df = entity_check(aug_df, 'honorific')

    # entity_indexing
    aug_df = entity_indexing(aug_df, 'honorific')

    # entity_check 2
    for i in range(len(aug_df.index)):
        if aug_df['sub_word'].iloc[i] != aug_df['honorific_clean'].iloc[i][aug_df['honorific_sub_start_idx'].iloc[i]:aug_df['honorific_sub_end_idx'].iloc[i]] or \
            aug_df['obj_word'].iloc[i] != aug_df['honorific_clean'].iloc[i][aug_df['honorific_obj_start_idx'].iloc[i]:aug_df['honorific_obj_end_idx'].iloc[i]]:
            print('error: ', i, aug_df['sentence'].iloc[i], aug_df['honorific_clean'].iloc[i], aug_df['sub_word'].iloc[i], aug_df['obj_word'].iloc[i], 
                aug_df['honorific_clean'].iloc[i][aug_df['honorific_sub_start_idx'].iloc[i]:aug_df['honorific_sub_end_idx'].iloc[i]],
                aug_df['honorific_clean'].iloc[i][aug_df['honorific_obj_start_idx'].iloc[i]:aug_df['honorific_obj_end_idx'].iloc[i]])

    total_df = concat_dataset(original_df, aug_df, 'honorific')
    print('origin: ', original_df.columns)
    print('total: ', total_df.columns)

    # na check
    print('NA: ', total_df.isna().sum())
    # ducplicated check
    print('duplicated: ', total_df[total_df[['sentence', 'label', 'subject_entity', 'object_entity']].duplicated(keep=False)==True].sort_values('sentence'))
    total_df.to_csv('train/train_honorific_augmentation.csv')

