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
2. 문장 내 부사 -> 사전 뜻풀이로 교체 | 랜덤 삭제
3. 문장 내 랜덤 부사 랜덤 위치 삽입 (원 문장에 부사가 없을 경우)
4. AEDA 적용

"""
# utils
def label_filtering(df):
    """label이 2000개 이상인 것들 제외하고 augmentation 수행하기 위해, 해당 label들을 filtering
    
    Args:
        df (dataframe): dataframe

    Returns:
        filterd_df (dataframe): label filtering이 적용된 dataframe
    """
    delete_list = ['no_relation', 'org:top_members/employees', 
               'per:employee_of', 'per:title']
    filter_df = df[df['label'].apply(lambda x: x not in delete_list) == True]
    return filter_df

def sentence_with_entity_marker(df):
    """증강 과정에서 엔티티가 변형되지 않도록 보호하기 위해 문장에 엔티티 위치를 직접 표시

    Args:
        df (dataframe): dataframe

    Returns:
        df (dataframe): 엔티티 위치가 표시된 문장이 있는 칼럼을 추가한 dataframe
    """
    # entity를 dictionary 형으로 변환하고, 각 key-value 쌍을 별도의 칼럼으로 분리합니다.
    df['subject_entity'] = df['subject_entity'].apply(lambda x: literal_eval(x))
    df['object_entity'] = df['object_entity'].apply(lambda x: literal_eval(x))

    df['sub_word'] = df['subject_entity'].apply(lambda x: x['word'])
    df['obj_word'] = df['object_entity'].apply(lambda x: x['word'])

    df['sub_start_idx'] = df['subject_entity'].apply(lambda x: x['start_idx'])
    df['sub_end_idx'] = df['subject_entity'].apply(lambda x: x['end_idx'])

    df['obj_start_idx'] = df['object_entity'].apply(lambda x: x['start_idx'])
    df['obj_end_idx'] = df['object_entity'].apply(lambda x: x['end_idx'])

    df['sub_type'] = df['subject_entity'].apply(lambda x: x['type'])
    df['obj_type'] = df['object_entity'].apply(lambda x: x['type'])

    # 엔티티의 문장 내 위치를 subject: @, #@, object: #, !# 로 표시해줍니다.
    df['sentence_with_entity_marker'] = None
    df['sub_extracted'] = None
    df['obj_extracted'] = None 

    for i in df.index:
        sub_start_idx = df['sub_start_idx'].loc[i]
        sub_end_idx = df['sub_end_idx'].loc[i]
        
        obj_start_idx = df['obj_start_idx'].loc[i]
        obj_end_idx = df['obj_end_idx'].loc[i]
        
        df['sub_extracted'].loc[i] = df['sentence'].loc[i][sub_start_idx:sub_end_idx+1]
        df['obj_extracted'].loc[i] = df['sentence'].loc[i][obj_start_idx:obj_end_idx+1]
        
        df['sentence_with_entity_marker'].loc[i] = list(df['sentence'].loc[i])
        
        # sub_entity, obj_entity 중 앞에 있는 엔티티에 먼저 마커를 붙여주면, 뒤에 있는 엔티티는 인덱스가 2만큼 밀려나는 것을 반영합니다.  
        if sub_start_idx > obj_start_idx:
        
            df['sentence_with_entity_marker'].loc[i].insert(sub_start_idx, '@')
            df['sentence_with_entity_marker'].loc[i].insert(sub_end_idx+2, '!@')

            df['sentence_with_entity_marker'].loc[i].insert(obj_start_idx, '#')
            df['sentence_with_entity_marker'].loc[i].insert(obj_end_idx+2, '!#')

        elif sub_start_idx < obj_start_idx:
            df['sentence_with_entity_marker'].loc[i].insert(obj_start_idx, '#')
            df['sentence_with_entity_marker'].loc[i].insert(obj_end_idx+2, '!#')
            
            df['sentence_with_entity_marker'].loc[i].insert(sub_start_idx, '@')
            df['sentence_with_entity_marker'].loc[i].insert(sub_end_idx+2, '!@')

        df['sentence_with_entity_marker'].loc[i] = ''.join(df['sentence_with_entity_marker'].loc[i]).strip()

    return df

def extract_entity(sentence):
    sub_word = sentence[sentence.find('@')+1:sentence.find('!@')]
    obj_word = sentence[sentence.find('#')+1:sentence.find('!#')]
    return sub_word, obj_word

def entity_check(df, aug_method_name):
    """변형 과정에서 entity 부분이 변형되었는지 확인합니다.
    만약 변형되었다면 원래의 entity로 다시 고쳐줍니다.

    Args:
        df (dataframe): dataframe
        aug_method_name (str): augmentation 방법의 이름 ex) 'honorific'
    
    Returns:
        df (dataframe): 변형된 entity를 고쳐준 dataframe
    """
    df['sub_extracted_2'] = df[aug_method_name].apply(lambda x: extract_entity(x)[0])
    df['obj_extracted_2'] = df[aug_method_name].apply(lambda x: extract_entity(x)[1])
    
    # entity가 변형되었다면 강제로 고쳐줍니다.
    for i in range(len(df.index)):        
        if df['sub_extracted'].iloc[i] != df['sub_extracted_2'].iloc[i]:
            df[aug_method_name].iloc[i] = df[aug_method_name].iloc[i].replace('@'+df['sub_extracted_2'].iloc[i]+'!@', '@'+df['sub_extracted'].iloc[i]+'!@')
            
            
        if df['obj_extracted'].iloc[i] != df['obj_extracted_2'].iloc[i]:
            df[aug_method_name].iloc[i] = df[aug_method_name].iloc[i].replace('#'+df['obj_extracted_2'].iloc[i]+'!#', '#'+df['obj_extracted'].iloc[i]+'!#')
            # print(f'{i} th row does not match with original entity')
            
    return df

def entity_indexing(df, aug_method_name):
    """변형 과정에서 바뀐 entity index를 entity 정보 dict에 업데이트해줍니다.

    Args:
        df (dataframe): dataframe
        aug_method_name (str): augmentation 방법의 이름 ex) 'honorific'

    Returns:
        df (dataframe): 변형 과정에서 바뀐 entity index를 반영한 dictionary 칼럼을 추가한 dataframe

    """
    df[aug_method_name + '_sub_start_idx'] = None
    df[aug_method_name + '_sub_end_idx'] = None

    df[aug_method_name + '_obj_start_idx'] = None
    df[aug_method_name + '_obj_end_idx'] = None

    df[aug_method_name + '_subject_entity_dict'] = None
    df[aug_method_name + '_object_entity_dict'] = None

    for i in range(len(df)):
        if df[aug_method_name].iloc[i].find('@') < df[aug_method_name].iloc[i].find('#'):

            df[aug_method_name + '_sub_start_idx'].iloc[i] = df[aug_method_name].iloc[i].find('@')
            df[aug_method_name + '_sub_end_idx'].iloc[i] = df[aug_method_name + '_sub_start_idx'].iloc[i] + len(df['sub_word'].iloc[i])
            
            df[aug_method_name + '_obj_start_idx'].iloc[i] = df[aug_method_name].iloc[i].find('#')-3
            df[aug_method_name + '_obj_end_idx'].iloc[i] = df[aug_method_name + '_obj_start_idx'].iloc[i] + len(df['obj_word'].iloc[i])

        elif df[aug_method_name].iloc[i].find('@') > df[aug_method_name].iloc[i].find('#'):

            df[aug_method_name + '_obj_start_idx'].iloc[i] = df[aug_method_name].iloc[i].find('#')
            df[aug_method_name + '_obj_end_idx'].iloc[i] = df[aug_method_name + '_obj_start_idx'].iloc[i] + len(df['obj_word'].iloc[i])
            
            df[aug_method_name + '_sub_start_idx'].iloc[i] = df[aug_method_name].iloc[i].find('@')-3
            df[aug_method_name + '_sub_end_idx'].iloc[i] = df[aug_method_name + '_sub_start_idx'].iloc[i] + len(df['sub_word'].iloc[i]) 

        # making dictionary
        df[aug_method_name + '_subject_entity_dict'].iloc[i] = str({'word':df['sub_word'].iloc[i], 
                                                            'start_idx':df[aug_method_name + '_sub_start_idx'].iloc[i],
                                                            'end_idx':df[aug_method_name + '_sub_end_idx'].iloc[i],
                                                            'type':df['sub_type'].iloc[i]})
        
        df[aug_method_name + '_object_entity_dict'].iloc[i] = str({'word':df['obj_word'].iloc[i], 
                                                            'start_idx':df[aug_method_name + '_obj_start_idx'].iloc[i],
                                                            'end_idx':df[aug_method_name + '_obj_end_idx'].iloc[i],
                                                            'type':df['obj_type'].iloc[i]})

    # 엔티티 표시를 위해 붙여줬던 특수기호들을 지워줍니다.
    df[aug_method_name+'_clean'] = df[aug_method_name].apply(lambda x: x.replace('!@', ''))
    df[aug_method_name+'_clean'] = df[aug_method_name+'_clean'].apply(lambda x: x.replace('@', ''))
    df[aug_method_name+'_clean'] = df[aug_method_name+'_clean'].apply(lambda x: x.replace('!#', ''))
    df[aug_method_name+'_clean'] = df[aug_method_name+'_clean'].apply(lambda x: x.replace('#', ''))
    # df[aug_method_name+'_clean'] = df[aug_method_name+'_clean'].apply(lambda x: x.replace('  ', ' '))
    return df


# 1. 반말 -> 존댓말 변형
def informal_to_honorific(sentence, honorific_model):
    """반말 문장을 존댓말 문장으로 바꿔주는 함수
        
    Args: 
        sentence (str): 문장
        honorific_model: 반말 문장을 존댓말 문장(습니다체)으로 바꿔주는 함수 (reference: https://github.com/kosohae/AIpjt-1)
    
    Returns:
        result (str): 존댓말 변환이 적용된 문장
    """

    # kiwi 형태소 분석기가 외국어를 없애버리는 이슈가 있음.
    # subject, object entity의 선행하는 부분에 외국어가 있으면 인덱스를 맞추기 어려워짐.
    # 이를 막기 위해, !# 혹은 !@ 뒤의 부분만 존댓말 변환 수행. (어차피 종결어미만 바뀌니 괜찮다.)
    if sentence.find('!#') < sentence.find('!@'):
        split_idx = sentence.find('!@')
    elif sentence.find('!#') > sentence.find('!@'):
        split_idx = sentence.find('!#')
        
    honorific = honorific_model.changer(sentence[split_idx+1:])
    
    # 존댓말 변형 오류 교정
    decomposed = hgtk.text.decompose(honorific)
    # 하었습니다, 가었습니다 등 -> 핬습니다, 갔습니다
    sub = re.sub(r'ㅏᴥㅇㅓㅆ', r'ㅏㅆ', decomposed)
    honorific = hgtk.text.compose(sub)
    #핬습니다 -> 했습니다
    sub = re.sub('핬', '했', honorific)
    honorific = hgtk.text.compose(sub)
    # 막었습니다, 받었습니다 -> 막았습니다. 받았습니다
    decomposed = hgtk.text.decompose(honorific)
    honorific = re.sub(r'(ㅏ[ㄱ-ㅎ]ᴥ)ㅇㅓㅆ', r'\1았', decomposed)
    honorific = hgtk.text.compose(sub)
    
    # 부른다 -> 부릅니다., 치르다 -> 치릅니다., 올린다 -> 올립니다
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ][ㄱ-ㅎ]?ᴥ)([ㄱ-ㅎ]ᴥ?[ㅏ-ㅣ])ㄴ?ᴥㄷㅏᴥ', r'\1\2ㅂ니다', decomposed)
    honorific = hgtk.text.compose(sub)
    
    # 기타 오류 수정
    honorific = honorific.replace('하어', '해')
    honorific = honorific.replace('했다.', '했습니다.')
    honorific = honorific.replace('이다.', '입니다.')
    honorific = honorific.replace('었다.', '었습니다.')
    honorific = honorific.replace('있다.', '있습니다.')
    honorific = honorific.replace('갔다.', '갔습니다.')
    honorific = honorific.replace('입닙니다.', '입니다.')
    honorific = honorific.replace('습닙니다.', '습니다.')
    honorific = honorific.replace('()', '')
    honorific = honorific.replace('  ', ' ')
    
    # 습니다체로 변경이 안 된 경우 마지막 부분에 '-요' 첨가
    if '니다' not in honorific:
        honorific = honorific[:-1] + '요.'
    result = sentence[:split_idx+1] + honorific
    return result.strip()


## 부사 -> 사전 뜻풀이 (synonym replacement)

def adverb_detector(sentence):
    # Tokenizing    
    tokenized = mecab.pos(sentence)
    
    # 어절 정보도 활용
    eojeol_list = sentence.split()
    
    for i, token in enumerate(tokenized):
        if i ==0 or i >= len(tokenized)-1:
            continue
        
        # 너무 짧은 부사는 제외.
        if token[1] == 'MAG' and len(token[0]) > 1:
            for i, eojeol in enumerate(eojeol_list):
                if (token[0] in eojeol and '#' not in eojeol and '@' not in eojeol):
                    return token[0]
            

def get_synonym(word):
    """단어를 입력 받아 해당 단어의 뜻풀이를 반환합니다.
    어떤 단어와, 그 단어의 사전 뜻풀이는 의미가 동일합니다.
    부사의 경우 뜻풀이는 대부분 '부사구'로 되어 있습니다. 즉, 단어와 그 단어의 사전 뜻풀이가 문장에서 수행하는 역할이 동일합니다.
    '부사'를 '뜻풀이'로 대체한다면 의미를 보존하면서 형태만 달라지는 augmentation이 가능합니다.
    뜻풀이는 다음 사전 https://dic.daum.net/ 에서 가져옵니다.

    Args:
        word (str): 사전 뜻풀이로 교체할 단어

    Returns:
        meaning (str): word의 사전(다음사전) 뜻풀이
    """
    res = requests.get("https://dic.daum.net/search.do?q=" + word, timeout=5)
    time.sleep(random.uniform(2,4))
    soup = BeautifulSoup(res.content, "html.parser")
    try:
        # 첫 번째 뜻풀이.
        meaning = soup.find('span', class_='txt_search')
    except AttributeError:
        return word
    if meaning == None:
        return word
    
    # parsing 결과에서 한글만 추출
    meaning = re.findall('[가-힣]+', str(meaning))
    meaning = ' '.join(meaning)
    
    # 띄어쓰기 오류 교정 (위 에 -> 위에)
    meaning = spell_checker.check(meaning).as_dict()['checked'].strip()
    return meaning.strip()
        
def synonym_replacement(sentence, adverb_list):
    """문장 내 부사의 유무에 따라서 부사를 사전 뜻풀이로 교체하거나, 새로운 부사를 문장에 추가합니다. .
    Args:
        
    
    """
    # 문장 안에 부사가 존재한다면:
    adverb = adverb_detector(sentence)
    
    if adverb:
        synonym = get_synonym(adverb)
        return sentence.replace(adverb, synonym)
    else:
        # 360 개 부사 목록 중 택 1
        new_adverb = random.choice(adverb_list)
        
        # 일단 안전하게 문장 맨 앞에 삽입해주자.
        return new_adverb + ' ' + sentence



# 데이터 통합
def concat_dataset(original_df, aug_df, aug_method_name):
    """원본 학습 셋과 증강한 학습 셋을 이어붙인다.
    Args:
        original_df (dataframe): 원본 train set
        aug_df (dataframe): 원본 train set을 변형한 데이터셋
        aug_method_name (str): 증강 방법

    Returns:
        total_df (dataframe): 원본 train set과 변형한 데이터셋을 합한 데이터셋
    """
    aug_df_for_concat = aug_df[['id', aug_method_name + '_clean', aug_method_name + '_subject_entity_dict', aug_method_name + '_object_entity_dict', 'label', 'source']]

    aug_df_for_concat = aug_df_for_concat.rename(columns={
        aug_method_name + '_clean':'sentence', 
        aug_method_name + '_subject_entity_dict':'subject_entity', 
        aug_method_name + '_object_entity_dict':'object_entity'}).reset_index(drop=True)

    print('aug_df_for_cocat: ', aug_df_for_concat.columns)
    # 증강 방법 표시
    original_df['aug_method'] = 'original'
    aug_df_for_concat['aug_method'] = aug_method_name

    # concat
    total_df = pd.concat([original_df, aug_df_for_concat], axis=0).reset_index(drop=True)
    print('total_df: ', total_df.columns)
    return total_df


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
    total_df.reset_index(drop=True).to_csv('module_test_train_honorific_augmentation.csv')

