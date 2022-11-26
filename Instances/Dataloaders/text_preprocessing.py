import re


# 따옴표 처리
## """" 제거
def four_double_quotation_delete(sentence):
    sentence = re.sub('""""', "", sentence)
    return sentence


## "" -> "으로 변경
def double_quotation_to_quotation(sentence):
    sentence = re.sub(r'""', '"', sentence)
    return sentence


## 따옴표 계열 통일
def quotation_consist(sentence):
    sentence = re.sub(r"[‘’′＇`]", "'", sentence)
    sentence = re.sub(r"[“”]", '"', sentence)
    return sentence


# 괄호 처리
## 소괄호 처리
def parenthesis_consist(sentence):
    sentence = re.sub(r"（", "(", sentence)
    sentence = re.sub(r"）", ")", sentence)
    return sentence


## 브래킷 처리
def bracket_consist(sentence):
    sentence = re.sub(r"[«≪〈]", "《", sentence)
    sentence = re.sub(r"[»≫〉]", "》", sentence)
    return sentence


## 박스 처리
def box_consist(sentence):
    sentence = re.sub(r"[˹『｢]", "「", sentence)
    sentence = re.sub(r"[˼』｣]", "」", sentence)
    return sentence


# -처리
def bar_consist(sentence):
    sentence = re.sub(r"[–—―]", "-", sentence)
    return sentence


# cdot 처리
def cdot_consist(sentence):
    sentence = re.sub(r"[⋅･•・]", "·", sentence)
    return sentence


# 특수기호 제거
def symbol_delete(sentence):
    sentence = re.sub(r"[▲△▴▵□☎☏⁺∞Ⓐ®𑀫𑀕𑀥★☆♡♥※˘³𑀫𑀕𑀥]", "", sentence)
    return sentence


# 야구에서 사용되는 분수표현 대체
def frac_consist(sentence):
    sentence = re.sub(r"⅓", " 1/3", sentence)
    sentence = re.sub(r"⅔", " 2/3", sentence)
    return sentence


# 회사 표현 대체
def company_consist(sentence):
    sentence = re.sub(r"㈔", "(사)", sentence)
    sentence = re.sub(r"㈜", "(주)", sentence)
    return sentence


# 단위 기호 통일
def measure_consist(sentence):
    sentence = re.sub(r"㎏", "kg", sentence)
    sentence = re.sub(r"ℓ", "L", sentence)
    sentence = re.sub(r"㎖", "mL", sentence)
    sentence = re.sub(r"㎜", "mm", sentence)
    sentence = re.sub(r"㎞", "km", sentence)
    sentence = re.sub(r"㎡", "m²", sentence)
    sentence = re.sub(r"㎿", "MW", sentence)
    sentence = re.sub(r"ｍ", "m", sentence)
    sentence = re.sub(r"°", "도", sentence)
    sentence = re.sub(r"℃", "도", sentence)

    return sentence


# 로마 숫자 -> 숫자로 변경
def roma_to_num(sentence):
    roma_Alpha_Upper = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ"]
    roma_Alpha_Lower = ["ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ", "ⅵ", "ⅶ", "ⅷ", "ⅸ", "ⅹ", "ⅺ", "ⅻ"]
    for i in range(0, 12):
        sentence = re.sub(roma_Alpha_Upper[i], str(i + 1), sentence)
        sentence = re.sub(roma_Alpha_Lower[i], str(i + 1), sentence)
    return sentence


# 그외 유니코드 이슈 처리
def unicode_err_consist(sentence):
    sentence = re.sub(r"％", "%", sentence)
    sentence = re.sub(r"，", ",", sentence)
    sentence = re.sub(r"／", "/", sentence)
    sentence = re.sub(r"１", "1", sentence)
    sentence = re.sub(r"：", ":", sentence)
    sentence = re.sub(r"？", "?", sentence)
    sentence = re.sub(r"～", "~", sentence)

    return sentence


# 전체 텍스트 전처리
def text_preprocessing(sentence):
    # 따옴표 계열 처리 과정
    sentence = four_double_quotation_delete(sentence)
    sentence = double_quotation_to_quotation(sentence)
    sentence = quotation_consist(sentence)

    # 괄호 계열 처리
    sentence = parenthesis_consist(sentence)
    sentence = bracket_consist(sentence)
    sentence = box_consist(sentence)

    # bar 처리
    sentence = bar_consist(sentence)

    # cdot 처리
    sentence = cdot_consist(sentence)

    # 특수 기호 제거
    sentence = symbol_delete(sentence)

    # 분수 표현(야구) 통일
    sentence = frac_consist(sentence)

    # 회사표현 통일
    sentence = company_consist(sentence)

    # 단위표현 통일
    sentence = measure_consist(sentence)

    # 로마숫자 숫자로 변경
    sentence = roma_to_num(sentence)

    # 그외 유니코드 이슈 처리
    sentence = unicode_err_consist(sentence)

    # 빈괄호 제거
    sentence = re.sub(r"()", "", sentence)
    sentence = re.sub(r"《》", "", sentence)

    # 공백 두번 이상인 것 처리 및 앞 뒤 공백 제거
    sentence = re.sub(r" +", " ", sentence).strip()

    return sentence
