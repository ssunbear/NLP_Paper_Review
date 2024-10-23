<aside>
✏️

`DPO` , `직접 선호 최적화` : 인간의 선호도에 기반해 간단한 분류 손실만을 사용하여 언어모델을 최적화 하는 새로운 방법  (허깅페이스 오픈 LLM 리더보드와 한국어 리더보드 상위에서도 좋은 성적을 거두고 있는 방법론)

</aside>

# 1. Introduction

## AI Alignment : Human 피드백을 학습할수 있는 방법론

- 인공지능이 인간의 가치와 목표에 일치하도록 하는 것을 목표로 함
- AI alignment 연구자는 이상 구체화와 실제 AI 행동의 일치를 목표로 설계 구체화를 활용
    - Ideal specification : 인간 관리자가 시스템이 하기 원하는 것
    - Design specification : 실제로 AI 시스템을 구축하는데 사용되는 청사진
    - Emergent behavior: AI가 실제로 하게되는 행동
- AI Alignment 방법
    - Scalable oversight
    - Training by debate
    - Reward modeling and iterated amplification : 인간의 피드백을 모사하는 모델로부터 보상을 받는 강화학습 시스템
    - Inferring human preferences from behavior : 인간의 행동을 기반으로 선호를 최대한 실현

## RLHF

- Reinforcement Learning from Human Feedback : 강화 학습 기반 인간 피드백
    - 사람의 피드백을 통해 학습한 리워드 모델을 이용하여 생성모델 답변을 긍정, 부정 피드백 정렬
    - OpenAI GPT-3.5, GPT-4, Anthropic의 Claude-2, Meta의 LLaMA-2-Chat도 이 방법으로 강화학습

# 2. Related Works

<aside>
✏️

> LLM의 학습 프로세스 전체에서 모델 프리트레이닝으로 거대 PLM모델을 만든 후 파인튜닝 그리고 모델 정렬과정까지의 기술
> 
</aside>

## SFT(Supervised Fine-tuning)

- Unsupervised Learning (Pre-training) 과정이 선행된 후 fine-tuning하는 과정
- 특정 도메인의 데이터 혹은 크라우드 소싱 등을 통해 구축한 양질의 (Prompt, Response) 데이터를 구축하여 fine-tuning 진행
- 입력 프롬프트에 대해 사람의 의도에 맞는 원하느 답변 형태(말투, 답변 내용, 지식 등)로 문장을 생성하는 강화학습

## Human Preference Alignment

- SFT 과정에서 미리 잘 정제된 양질의 데이터로 미세조정을 거쳐야 제대로 된 답변 생성이 가능하고 이후 필요한 정렬과정에서 reward 모델에 필요한 답변 데이터를 수집가능
- SFT 이후 모델  정렬하는 과정을 `Learning from human Feedback` 이나 `Human Preference Alignment` 라고 함
- 긍정, 부정이나 순위 형식으로 피드백을 주면서 사람의 선호도를 모델에 강화학습시킴

### RLHF (Rank Responses to Align Language Models with Human Feedback without tears)

- 가장 대표적인 강화학습 기반의 Alignment 방식으로 사람의 피드백을 가지고 보상을 계산하여 모델이 강화학습

### RRHF

- 리워드 모델이 매긴 점수와 각 답변의 문장 확률 정보를 기반으로 학습을 진행
- 리워드 모델의 점수를 기반으로한 Ranking Loss를 계산하여 Ranking Loss를 최소화 하는 방식으로 학습

### SLiC-HF

- 두 답변 후보를 동시에 입력으로 받아서 그 중 어떤 답변이 더 좋은 답변 인지를 계산

### DPO (Direct Preference Optimizaiton)

- 리워드 모델 선호도 학습용 데이터를 모델에 직접 사용하여 positive 답변에 대한 확률은 높아지도록, negative 답변에 대한 확률은 낮아지도록 학습

<aside>
✏️

스탠포드 연구진은 여러가지 지표 중 **RLHF와 DPO의 비교를** 통해 DPO 기술을 공개하였음

</aside>

## RLHF vs. DPO

![스크린샷 2024-10-24 오전 2.53.28.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/32bae4cb-b294-47cc-ab1d-8f916f14004a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_2.53.28.png)

- RLHF : human preference 데이터를 가지고 reward 모델 LLM 모델 먼저 지도 학습으로 만들어 둔 후에 해당 모델을 가지고서 파이 SFT 모델을 추가로 학습하여 Language 모델의 Polish를 최적화하는 방식으로 강화학습하는 전체 과정

→ 언어 생성과 대화능력 최대화 가능 BUT, 기본 polish language 모델의 중간 보상 모델 필요, 단계와 과정 복잡성으로 계산비용 증가 및 불안정 

- **DPO : reward 모델링 과정 생략, 선호 데이터를 직접 언어모델 최적화에 사용하는 방법 제시**

→ reward function과 Optimal polish간에 reward maximization 문제를 단순히 1회 classification 문제로 치환함으로써 여러가지 장점(안정적, 효율적, 보상모델의 fitting, LM의 샘플링과 광범위한 하이퍼 파라미터 튜닝 불필요 등)을 가져갈 수있다.

## DPO 학습 Pipeline

1. π 모델 (Base LM)을 SFT로 학습하는 과정 (π SFT)

: 관심 데이터셋에 대해서 지도학습 + 미세조정 → 선후 데이터를 사용하여 1단계 모델에 대한 선호 학습

→ 파이프라인에서 preference 데이터는 여전히 polish 교육시 필요

1. (π SFT)위에 Preference를 학습하는 과정 → (π DPO)
- 보상 모델 학습 과정이 없음. 학습할 때 리워드 모델을 사용하지 않고 레퍼런스 모델만 사용
- 파인튜닝시 LM에서 Instruction에 대한 Output Sampling을 하지 않음
- 광범위한 Hparams 튜닝과정이 없어, RLHF가 LR에 민감한 문제가 없어, 보다 안정적으로 학습이 가능

# 3. Methods

## Parameterization

![스크린샷 2024-10-24 오전 3.02.20.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/0c8e735f-63b1-4663-964c-3c2879eb0521/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.02.20.png)

![스크린샷 2024-10-24 오전 3.02.55.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/df5f5091-17b9-4c03-be78-2ecc41541961/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.02.55.png)

- RLHF
    - 보상 모델 학습 > PPO (proximal policy optimization)
    - 보상 모델 학습 : 선호 답변과 비선호 답변을 입력으로 받아서 비교하는 Ranking Loss를 목적함수로 하여 polish 모델에게 보상을 줄 보상모델 학습
    - PPO : policy 모델이 생성한 답에 대해 RM이 매긴 보상 스코어로, policy 모델을 강화학습, PPO의 목적함수에는 보상향 이외에도 KL 제약이 추가로 존재

<aside>
✏️

- KL인 Polish 모델과 레퍼런스 모델이 예측한 log likelihood가 점차적으로 증가하면서 학습이 안정적인 단계에까지 매우 느리게 도달하는 것이 관찰
</aside>

- DPO policy objective
    - DPO는 RLHF의 SFT와 RM + PPO 과정에서 RM + PPO과정을 단일한 MLE기반의 목적함수로 대체하여 한번에 RLHF를 학습
    - 모델이 생성한 답변 후보들을 reward 모델이 우열을 가려서 학습하지 않고 reward 모델의 학습에 사용하는 선호도데이터를 직접 학습에 사용

1. polish 모델의 reference 모델과 예측간 차이 최소화 단계를 진행 
2. DPO 손실함수의 optimal 솔루션식을 보상값에 대해 정리
3. polish 모델과 reference 모델 예측값 차이가 함께 항으로 묶임 → PPO를 최적화 하는 optimal reward에 해당하는 보상값이 도출

![스크린샷 2024-10-24 오전 3.07.40.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/d21ebadf-6ca0-4920-9f89-819107cca088/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.07.40.png)

![스크린샷 2024-10-24 오전 3.07.51.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/8386667d-7a1e-489c-a900-4cb3f0e8cae9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.07.51.png)

![스크린샷 2024-10-24 오전 3.08.04.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/57ee6578-092a-454d-ae75-911fccda18f3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_3.08.04.png)

DPO 목적함수의 전체 식 : 선호 답변과 비선호 답변을 비교하는 랭킹 Loss이면서 각 항이 polish 모델과 reference 모델의 예측을 비우려하는 LM이 단순기반 손실 함수가 되는 형태

- 선호 답변에 대한 Polish 모델과 reference 모델간의 예측한 차이가 증가
- 비선호 답변에 대한 Polish 모델과 reference 모델간의 예측한 차이가 작아지거나 degrade
- 선호답변과 비선호 답변에 대한 polish 모델의 예측간 차이가 증가

# 4. Experiments

- Task
    - 제어된 감정 생성
        - IMDb 데이터셋
        - x는 영화 리뷰의 앞부분, y는 긍정적인 감정의 영화 리뷰
            
            
    - 요약
        - Reddit TL;DR 요약 데이터셋
        - x는 Reddit의 게시물, y는 게시물의 주요 요점에 대한 요약
            
            
    - Single-turn 대화
        - Anthropic Helpful and Harmless 대화 데이터셋
        - x는 사람의 질문, y는 질문에 대해 매력적이고 유용한 응답
            
            

저자들은 평가에 대해 두 가지 다른 접근 방식을 사용하였다.

1. 제어된 감정 생성 task의 경우 reward의 경계와 레퍼런스 모델에 대한 KL divergence를 기준으로 각 알고리즘을 평가한다. 이 경계는 ground-truth reward function(감정 분류기)에 접근할 수 있기 때문에 계산 가능하다.
2. 요약과 single-turn 대화의 경우 ground-truth reward function이 알려져 있지 않다. 따라서 각각 요약 품질과 응답 유용성에 대한 사람의 평가의 대체재로 GPT-4를 사용하여 baseline에 대한 승률로 알고리즘을 평가한다. 요약의 경우 테스트셋의 레퍼런스 요약을 baseline으로 사용한다. 대화의 경우 테스트셋에서 선호하는 응답을 baseline으로 사용한다.

6B 이하의 모델로 best of N 방식으로 실험을 진행

(좌) IMDb 감정 생성 task에 대한 결과 - sampling temperature을 0~1사이로 변화하면서 요약을 실험한 task를 진행 

(우) TL:DR 요약 task에 대한 승률 - reward 와 convergence 평균 값을 점으로 표현 (알고리즘 별로 베타, 러닝레이트, 랜덤 시드를 다르게 10step 이내에서 22가지 다른 실험 진행)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/728c8c4c-68a9-4202-a4c3-53543c0e70f4/image.png)

DPO 가 PPO에 비해 더 낮은 곳에 앞쪽으로 모여서 KL위치

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/acde0876-3e00-4338-9328-0bd5a75fe87e/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/dc5054b2-61f1-4246-9dfd-8d9d7e330b35/image.png)

DPO 가 PPO에 비해 위쪽으로 더 높은 리워드

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5dec6805-3a18-4a48-bf92-2a90b112134d/ab4233da-83fc-4017-947f-50b0fc316b7d/image.png)

DPO가 best of 128제외하고 가장 좋다는 결과 → 미세 조정에서 상당히 빠르게 수렴 

→ DPO 방식이 수식을 단순화 함으로써 안정성을 추구하는데 성능까지 높게 나오는것을 강조

# 5. Conclusion

- DPO가 RLHF를 대체하는 방법으로써 언어모델을 인간의 선호에 맞추어 안전하고 통제가능한 모델로 학습하는 과정을 간소화하는 새로운 방법을 제시함
- 비교적 컴퓨팅 비용과 자원을 줄이는 동시에 성능을 높이고 안정적이고 구현이 쉽고 효율적인 장점
- 질 좋은 선호도 데이터를 사용하고 이전 단계에서 미세조정(SFT)에 유의하는 것이 중요

학습한 데이터 외에는 취약하다는 이야기가 존재 → 실제 서비스에 구현되기 위해서는 optimization, over optimization 이슈나 더 큰 모델에 대한 다양한 추가실험이 진행되면 좋을 것 같다..
