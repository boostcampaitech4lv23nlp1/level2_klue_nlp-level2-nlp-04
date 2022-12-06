# 프로젝트 개요

## RE Task

> 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제이다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있다.
> 

![image](https://user-images.githubusercontent.com/101449496/205580342-65dc8275-59d2-478e-afbf-875762ce463d.png)

## 평가 지표

### **Micro F1 score**

> micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여한다. 데이터 분포상 많은 부분을 차지하고 있는 **no_relation class는 제외하고 F1 score가 계산된다**.
> 

### **AUPRC**

> x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정한다. 주어진 클래스에 대한 AUPRC는 PR 곡선 아래에 대한 면적이며 면적이 넓을수록 성능이 좋다. 이를 평균내어 계산하며 imbalance한 데이터에 유용한 metric이다.
> 

## 데이터

- **데이터 셋 통계**
    - train.csv: 총 32470개
    - test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현)
- **데이터 구조**
    
    ![image](https://user-images.githubusercontent.com/101449496/205580403-79faf5dd-8451-403c-96c0-0d8797fc501c.png)
    
    - id : 문장에 대한 고유 id
    - sentence : 추론해야 할 entity 쌍이 있는 sentence
    - subject_entity : word, start_idx, end_idx, type 정보로 구성
    - object_entity : word, start_idx, end_idx, type 정보로 구성
    - label : label 정보, 총 30개 classes로 분류
    - sourec : 샘플의 출처이며 policy_briefing / wikipedia / wikitree로 구성
<br/><br/><br/>
# 팀 구성 및 역할
| 이름 | 역할 |
| :--- | :--- |
| 🌱 김해원 | • 리서치 (Entity Marker, R-BERT) <br/> • 모델링 (Entity Marker, R-BERT) <br/> • Input data 형태 실험 <br/> • Label smoothing &emsp; |
| 🌱 김혜빈 | • 코드 리뷰 <br/> • 데이터 불균형 문제 리서치 <br/> • 모델링 실험 <br/> • Focal Loss 적용 &emsp; |
| 🌱 박준형 | • EDA <br/> • 데이터 증강 <br/> • Under Sampling <br/> • LR scheduler &emsp; |
| 🌱 양봉석 | • EDA <br/> • 데이터 증강 <br/> • Under Sampling <br/> • LR scheduler &emsp; |
| 🌱 이예령 | • 리서치 (ERACL) <br/> • 모델링 (Binary Loss, Pooling Layer) <br/> • 에러 분석 구현 (Confusion Matrix) &emsp; |

<br/><br/><br/>
# 프로젝트 구조

![image](https://user-images.githubusercontent.com/101449496/205580497-3c218016-1f10-4c54-8662-e162f8e50ba5.png)

<br/><br/><br/>
# 실험내역

### Data

1. **Under Sampling**
2. **Data Augmentation**
3. **Text Preprocessing**
4. **Tokenizer**
5. **Input data 형태**

### **Model**

1. **Entity Marker**
2. **R-BERT**
3. **LSTM**
4. **Pooling**
5. **Segment Embedding with Roberta**

### Optimization

1. **Binary-Loss**
2. **Focal Loss**
3. **Label Smoothing**
4. **LR scheduler**

### Ensemble

1. **K Fold**
2. **Model Ensemble**

<br/><br/><br/>

# 실행코드

```python
# Train
python main.py -m t -c my_config 

# Inference
python main.py -m i -c my_config -s SavedModels/my_model.ckpt

# Analysis 
python main.py -m -a -c my_config -s SavedModels/my_model.ckpt
```
<br/><br/><br/>
# 팀 회고

### 다음 프로젝트에 해볼 시도

**좋았던 점은 그대로**

- 베이스라인 코드를 작업 단위로 모듈화하여 코드 확장성을 갖고  빠른 기간 내에 완성하여 다양한 실험을 하기 위한 발판을 마련해보자
- githubflow 등의 convention을 활용 및 Pull Request 작성 및 코드 리뷰를 하면서 프로젝트 진행해보자
- 활발한 의사소통 통해 서로 모르는 부분을 채워줄 수 있었으며 함께 성장하자

**보다 체계적인 프로세스**

- 프로젝트 시작 전 상세하게 역할 분담을 정하고,  다같이 Task에 대한 정의 및 관련 논문, Survey 등 사전 조사를 해오자
- 본격적인 리더보드 제출 시작 전에 다같이 대회 Task에 대한 토론하는 시간을 갖자
    - 프로젝트에 대한 전체적인 challenge 및 문제 정의                ex) UNK 토큰 문제, 데이터 불균형 문제
    - 리서치 할 논문에 대한 전체적인 탐색 및 계획 세우기
    
    ![image](https://user-images.githubusercontent.com/101449496/205580662-ea9e11a7-341d-471b-ab6f-f86ebfe7c71b.png)

    

**다양한 리서치 및 효율적인 실험**

- 최신 논문 및 구현 코드를 통해 프로젝트를 진행해보자
- 쉘 스크립트 및 wandb의 sweep 기능을 통해 서버의 유휴 시간을 최소화해보자
- `local_files_only=True` 를 활용하여 허깅페이스 모델 로드 시간을 단축해보자
- backbone 모델을 이번 처럼 많이 탐색하지 말고 좋았던 모델에 대해 깊게 파고 들면서 실험을 진행해보자
- Ensemble 실험 기간도 기간을 좀 더 길게 잡고 세밀하게 분석하여 해보자

**How 보다는 Why**

- pytorch lightning 및 hugging face 라이브러리 대신 pytorch를 통해 베이스 라인을 구축해보고 훈련의 전체적인 동작 및 forward, backward 연산 과정을 살펴보자
- 리서치를 통해 실험을 할 때 어떠한 방법을 적용했는보다 **왜** 이러한 방법을 시도해보았는지 생각해보자
- 실험 결과에 대해 단순히 score만 확인하는 것 뿐만 아니라 오답 데이터가 어떤 것이 나오고 그에 대한 데이터 분포가 **왜** 이렇게 나왔는지 분석해보자

**보다 적극적인 의사소통 및 실험 공유**

- 결과가 좋았던 모델을 로컬에 계속 남기는 대신 허깅페이스 레포지토리에 공유해보자
- 실험에 대한 리더보드 공유는 보다 정밀하게 구글의 스프레드시트를 활용해보자
- wandb에 실험마다 project 및 name을 명확히하여 결과를 한눈에 볼 수 있도록 하자
- 한 사람이 한 실험을 전적으로 맡아서 하는 것이 아니라 소통을 통해 대조 실험을 병렬적으로 진행하여 서버의 유휴시간을 최소화 해보자
<br/><br/><br/>
# 랩업리포트

[Wrap-up Report](https://leeyeryeong.notion.site/KLUE-REPORT-60b06143abfa42bf8aff579212e13682)



