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

##TODO: 함수에 self. 추가
# 구동 확인
# augmentation.py에 코드가 어떤 순서로 들어갈지 고민.
# augmentation-swap (주체 엔티티 포함 어절, 객체 엔티티 포함 어절 swap. 동일 어절에 있을 경우 순서만 교체.

# general utils
# label filtering
def label_filtering(df, n):
    """label이 n개 이상인 것들 제외하고 augmentation 수행하기 위해, 해당 label들을 filtering
    Args:
        df (dataframe): dataframe
    Returns:
        filterd_df (dataframe): label filtering이 적용된 dataframe
    """
    label_dict = dict(df['label'].value_counts())
    # 개수가 n 개 미만인 label만 필터링
    delete_list = [k for k, v in label_dict.items() if int(v) > n]
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
        
        df['sub_extracted'].loc[i] = df['sentence'].loc[i][sub_start_idx:sub_end_idx]
        df['obj_extracted'].loc[i] = df['sentence'].loc[i][obj_start_idx:obj_end_idx]
        
        df['sentence_with_entity_marker'].loc[i] = list(df['sentence'].loc[i])
        
        # sub_entity, obj_entity 중 앞에 있는 엔티티에 먼저 마커를 붙여주면, 뒤에 있는 엔티티는 인덱스가 2만큼 밀려나는 것을 반영합니다.  
        
        if sub_start_idx < obj_start_idx:
        
            df['sentence_with_entity_marker'].loc[i].insert(sub_start_idx, '@')
            df['sentence_with_entity_marker'].loc[i].insert(sub_end_idx+1, '!@')

            df['sentence_with_entity_marker'].loc[i].insert(obj_start_idx+2, '#')
            df['sentence_with_entity_marker'].loc[i].insert(obj_end_idx+3, '!#')

        elif sub_start_idx > obj_start_idx:
            df['sentence_with_entity_marker'].loc[i].insert(obj_start_idx, '#')
            df['sentence_with_entity_marker'].loc[i].insert(obj_end_idx+1, '!#')
            
            df['sentence_with_entity_marker'].loc[i].insert(sub_start_idx+2, '@')
            df['sentence_with_entity_marker'].loc[i].insert(sub_end_idx+3, '!@')

        df['sentence_with_entity_marker'].loc[i] = ''.join(df['sentence_with_entity_marker'].loc[i]).strip()

    return df

# 데이터셋 통합
def concat_dataset(original_df, aug_df, aug_method_name):
    """원본 학습 셋과 증강한 학습 셋을 이어붙입니다.
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


class entityProcessing():
    def __init__(self):
        pass

    def extract_entity(self, sentence):
        sub_word = sentence[sentence.find('@')+1:sentence.find('!@')]
        obj_word = sentence[sentence.find('#')+1:sentence.find('!#')]

        return sub_word, obj_word

    # 원본 엔티티를 보존해야 할 때만 사용
    def entity_check(self, df, aug_method_name):
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
            if df['sub_word'].iloc[i] != df['sub_extracted_2'].iloc[i]:
                df[aug_method_name].iloc[i] = df[aug_method_name].iloc[i].replace('@'+df['sub_extracted_2'].iloc[i]+'!@', '@'+df['sub_word'].iloc[i]+'!@')

            if df['obj_word'].iloc[i] != df['obj_extracted_2'].iloc[i]:
                df[aug_method_name].iloc[i] = df[aug_method_name].iloc[i].replace('#'+df['obj_extracted_2'].iloc[i]+'!#', '#'+df['obj_word'].iloc[i]+'!#')

        return df

    # entity_re_indexing
    def entity_re_indexing(self, df, aug_method_name):
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
        for i in range(len(df)):
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

    # 원본 엔티티를 보존해야 할 때만 사용.
    def entity_check_after_cleaning(self, df, aug_method_name):
        for i in range(len(df.index)):
            if df['sub_word'].iloc[i] != df[aug_method_name+'_clean'].iloc[i][df[aug_method_name+'_sub_start_idx'].iloc[i]:df[aug_method_name+'_sub_end_idx'].iloc[i]] or \
                df['obj_word'].iloc[i] != df[aug_method_name+'_clean'].iloc[i][df[aug_method_name+'_obj_start_idx'].iloc[i]:df[aug_method_name+'_obj_end_idx'].iloc[i]]:
                print('error: ', i, df['sentence'].iloc[i], df[aug_method_name+'_clean'].iloc[i], df['sub_word'].iloc[i], df['obj_word'].iloc[i], 
                    df[aug_method_name+'_clean'].iloc[i][df[aug_method_name+'_sub_start_idx'].iloc[i]:df[aug_method_name+'_sub_end_idx'].iloc[i]],
                    df[aug_method_name+'_clean'].iloc[i][df[aug_method_name+'_obj_start_idx'].iloc[i]:df[aug_method_name+'_obj_end_idx'].iloc[i]])


# Augmentation
class augmentation():
    # 1. 반말 -> 존댓말 변형
    def informal_to_honorific(self, sentence, honorific_model):
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
    def _adverb_detector(self, sentence):
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
                

    def _get_synonym(self, word):
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
            
    def synonym_replacement(self, sentence, adverb_list):
        """문장 내 부사의 유무에 따라서 부사를 사전 뜻풀이로 교체하거나, 새로운 부사를 문장에 추가합니다. .
        Args:
            
        
        """
        # 문장 안에 부사가 존재한다면:
        adverb = self._adverb_detector(sentence)
        
        if adverb:
            synonym = self._get_synonym(adverb)
            return sentence.replace(adverb, synonym)
        else:
            # 360 개 부사 목록 중 택 1
            new_adverb = random.choice(adverb_list)
            
            # 일단 안전하게 문장 맨 앞에 삽입해주자.
            return new_adverb + ' ' + sentence


    # aeda
    def aeda(self, sentence):
        punc_list = list(".,;:?!")
        
        sentence = sentence.split()
        random_ratio = random.uniform(0.1, 0.3) # 범위는 ADEA 논문을 따름.
        n_ri = max(1, int(len(t) * random_ratio))
        
        for _ in range(n_ri):
            random_punc = random.choice(punc_list)
            random_idx = random.randint(0, len(sentence)-1)
            sentence.insert(random_idx, random_punc)
            
        return ' '.join(sentence).strip()


    # random_masking_replacement
    def random_masking_replacement(sentence, mask_token, unmasker):
        sentence = sentence.split()
        sub_start_idx = sentence.index(sub_start_marker)
        sub_end_idx = sentence.index(sub_end_marker)
        obj_start_idx = sentence.index(obj_start_marker)
        obj_end_idx = sentence.index(obj_end_marker)
        
        entity_indices = [sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx]
        
        random_idx = random.randint(0, len(sentence) - 4) # entity marker 4개.
        
        list_without_entity_marker = [(i, v) for i, v in enumerate(sentence) if i not in entity_indices]
        
        selected_token_idx, selected_token_value = list_without_entity_marker[random_idx-1]
        sentence[selected_token_idx] = mask_token
        
        # 복원된 토큰을 mask 인덱스와 교대함.
        unmask_result = unmasker(' '.join(sentence), skip_special_tokens=False)
        
        unmask_sentence = sentence
        if re.findall('[가-힣]', unmask_result[0]['token_str']):
            unmask_token = unmask_result[0]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        elif re.findall('[가-힣]', unmask_result[1]['token_str']):
            unmask_token = unmask_result[1]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        elif re.findall('[가-힣]', unmask_result[2]['token_str']):
            unmask_token = unmask_result[2]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        elif re.findall('[가-힣]', unmask_result[3]['token_str']):
            unmask_token = unmask_result[3]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        elif re.findall('[가-힣]', unmask_result[4]['token_str']):
            unmask_token = unmask_result[4]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        else:
            unmask_token = unmask_result[0]['token_str']
            unmask_sentence[selected_token_idx] = unmask_token
            
        unmask_sentence = ' '.join(unmask_sentence)
        unmask_sentence = unmask_sentence.replace('  ', ' ')

        return unmask_sentence.strip()


    # random masking insertion
    def random_masking_insertion(self, sentence, mask_token, unmasker):
        original_sentence = sentence
        sentence = sentence.split()
        # random_ratio = random.uniform(0.1, 0.3)
        # n_ri = max(1, int(len(sentence) * random_ratio))
        # 한 문장에 하나의 mask token만 있어야 제대로된 복원이 가능함.

        sub_eojeol_spt = re.findall('\w*\[S1\].+\[S2\]\w*', original_sentence)[0]
        obj_eojeol_spt = re.findall('\w*\[S3\].+\[S4\]\w*', original_sentence)[0]
        
        sub_eojeol = re.sub('\[S1\]|\[S2\]', '', sub_eojeol_spt).strip() 
        obj_eojeol = re.sub('\[S3\]|\[S4\]', '', obj_eojeol_spt).strip()
        # print(sub_eojeol, obj_eojeol)

        list_without_entity = sentence
        list_without_entity_for_idx = [(i, v) for i, v in enumerate(list_without_entity) if i < min(list_without_entity.index('[S1]'), list_without_entity.index('[S3]')) 
                                                                                                    or i > max(list_without_entity.index('[S2]'), list_without_entity.index('[S4]'))]
        
        random_idx = random.randint(0, len(sentence) - len(sub_eojeol_spt.split()) - len(obj_eojeol_spt.split()) - 1) # entity 어절을 빼야 함.
        selected_token_idx, selected_token_value = list_without_entity_for_idx[random_idx-1]
        sentence.insert(selected_token_idx, mask_token)

        # replacement
        unmask_result = unmasker(' '.join(sentence))

        if re.findall('[가-힣]', unmask_result[0]['token_str']):
            unmask_sentence = unmask_result[0]['sequence'].split()

        elif re.findall('[가-힣]', unmask_result[1]['token_str']):
            unmask_sentence = unmask_result[1]['sequence'].split()

        elif re.findall('[가-힣]', unmask_result[2]['token_str']):
            unmask_sentence = unmask_result[2]['sequence'].split()
        elif re.findall('[가-힣]', unmask_result[3]['token_str']):
            unmask_sentence = unmask_result[3]['sequence'].split()
        elif re.findall('[가-힣]', unmask_result[4]['token_str']):
            unmask_sentence = unmask_result[4]['sequence'].split()    
        else:
            unmask_sentence = unmask_result[0]['sequence'].split()

        # entity 원상복구
        unmask_sentence = ' '.join(unmask_sentence)
        unmask_sentence = re.sub(sub_eojeol, sub_eojeol_spt, unmask_sentence)
        unmask_sentence = re.sub(obj_eojeol, obj_eojeol_spt, unmask_sentence)

        return unmask_sentence