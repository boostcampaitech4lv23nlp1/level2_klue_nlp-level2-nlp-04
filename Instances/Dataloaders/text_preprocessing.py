import re


# 따옴표 처리
## """" 제거  ("""")이 하나인 경우 : 그 안의 문장이 생략되었다 보고 마스크 토큰 삽입
## 다수인 경우 전부 "로 치환
def four_double_quotation_delete(sentence, tokenizer):
    if len(re.findall('""""', sentence)) == 1:
        sentence = re.sub('""""', tokenizer.mask_token, sentence)
    else:
        sentence = re.sub('""""', '"', sentence)
    return sentence


## "" -> "으로 변경
def double_quotation_to_quotation(sentence):
    sentence = re.sub(r'""', '"', sentence)
    return sentence


## 따옴표 계열 통일
def quotation_consist(sentence):
    sentence = re.sub(r"[′＇`]", "'", sentence)
    # sentence = re.sub(r"[“”]", '"', sentence)
    return sentence


# 괄호 처리
## 소괄호 처리
def parenthesis_consist(sentence):
    sentence = re.sub(r"（", "(", sentence)
    sentence = re.sub(r"）", ")", sentence)
    return sentence


## 브래킷 처리
def bracket_consist(sentence):
    sentence = re.sub(r"[«≪]", "《", sentence)
    sentence = re.sub(r"[»≫]", "》", sentence)
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
    sentence = re.sub(r"[▲△▴▵□☎☏⁺∞Ⓐ®𑀫𑀕𑀥★☆♡♥※˘³𑀫𑀕𑀥↔]", " ", sentence)

    return sentence


# 야구에서 사용되는 분수표현 대체
def frac_consist(sentence):
    sentence = re.sub(r"⅓", " 1/3", sentence)
    sentence = re.sub(r"⅔", " 2/3", sentence)
    return sentence


# 숫자 구두점 제거  1,000 -> 1000
def num_punctuation_delete(sentence):
    m = re.search(r"[0-9]{1,3},[0-9]{3}(,[0-9]{3}){0,}", sentence)
    while m != None:
        origin = sentence[m.start() : m.end()]
        replace_num = re.sub(",", "", origin)
        sentence = sentence[: m.start()] + replace_num + sentence[m.end() :]
        m = re.search(r"[0-9]{1,3},[0-9]{3}(,[0-9]{3}){0,}", sentence)
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

    return sentence


# 로마 숫자 -> 알파벳으로 변경
def roma_to_num(sentence):
    # https://unicode-table.com/kr/sets/roman-numerals/
    sentence = re.sub(r"Ⅰ", "I", sentence)
    sentence = re.sub(r"Ⅱ", "II", sentence)
    sentence = re.sub(r"Ⅲ", "III", sentence)
    sentence = re.sub(r"Ⅳ", "IV", sentence)
    sentence = re.sub(r"Ⅴ", "V", sentence)
    sentence = re.sub(r"Ⅵ", "VI", sentence)
    sentence = re.sub(r"Ⅶ", "VII", sentence)
    sentence = re.sub(r"Ⅷ", "VIII", sentence)
    sentence = re.sub(r"Ⅸ", "IX", sentence)
    sentence = re.sub(r"Ⅹ", "X", sentence)
    sentence = re.sub(r"Ⅺ", "XI", sentence)
    sentence = re.sub(r"Ⅻ", "XII", sentence)
    sentence = re.sub(r"Ⅼ", "L", sentence)
    sentence = re.sub(r"Ⅽ", "C", sentence)
    sentence = re.sub(r"Ⅾ", "D", sentence)
    sentence = re.sub(r"Ⅿ", "M", sentence)

    sentence = re.sub(r"ⅰ", "i", sentence)
    sentence = re.sub(r"ⅱ", "ii", sentence)
    sentence = re.sub(r"ⅲ", "iii", sentence)
    sentence = re.sub(r"ⅳ", "iv", sentence)
    sentence = re.sub(r"ⅴ", "v", sentence)
    sentence = re.sub(r"ⅵ", "vi", sentence)
    sentence = re.sub(r"ⅶ", "vii", sentence)
    sentence = re.sub(r"ⅷ", "viii", sentence)
    sentence = re.sub(r"ⅸ", "ix", sentence)
    sentence = re.sub(r"ⅹ", "x", sentence)
    sentence = re.sub(r"ⅺ", "xi", sentence)
    sentence = re.sub(r"ⅻ", "xii", sentence)
    sentence = re.sub(r"ⅼ", "l", sentence)
    sentence = re.sub(r"ⅽ", "c", sentence)
    sentence = re.sub(r"ⅾ", "d", sentence)
    sentence = re.sub(r"ⅿ", "m", sentence)

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
def text_preprocessing(sentence, tokenizer):
    # 따옴표 계열 처리 과정
    sentence = four_double_quotation_delete(sentence, tokenizer)
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

    # 숫자 구두점 제거
    sentence = num_punctuation_delete(sentence)

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

    # 공백 두번 이상인 것 처리 및 앞 뒤 공백 제거
    sentence = re.sub(r"\s+", " ", sentence).strip()

    # 빈괄호 마스크 처리 (의미상 제거되면 안됨)
    mask_pattern = tokenizer.mask_token
    sentence = re.sub(r"\(\s?\)", mask_pattern, sentence)
    sentence = re.sub(r"《\s?》", mask_pattern, sentence)

    return sentence
