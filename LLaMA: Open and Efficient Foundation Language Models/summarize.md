# 1. Introduction

- Meta AI 에서 Open 하여 공개한 LLM 모델
    - 7B ~ 65B 규모의 foundation language models
- 저작권이 있거나 접근 불가능한 데이터 사용 없이, `Public Data` 만으로 SOTA 달성할 수 있다는 것을 보임 (trillions of tokens 수집)
- 성능 측면
    - LLaMA-13B 가 GPT-3 (175B) 보다 성능이 좋음
        - 이 모델의 경우 `single GPU` 로 inference 가 가능함
    - LLaMA-65B 가 Chinchilla-70B, PaLM-540B 와 성능이 비슷함
    - 우수한 성능과 더불어 앞서 말한 모델들과 달리 공개한 모델임
- LLM scaling laws 에 관한 논문 중 알아둬야 할 논문 2가지
    1. Scaling Laws for Neural Language Models (OpenAI, 2020): 모델 파라미터가 많을 수록 성능은 계속 좋아짐
    2. Training Compute-Optimal Large Language Models (DeepMind, 2022): 주어진 compute budget 내에서 파라미터만 늘리는 것이 능사는 아니다. 즉, 더 작은 모델 + 더 많은 data 의 조합이 성능이 더 좋을 수 있음 (Chincilla 논문)
- 이러한 다소 상반되는 scaling laws 내용에 대해 LLaMA 는 2번의 실험 결과에 영감을 얻어, GPT-3 (175B) 보다 작은 크기의 모델 LLaMA-13B 로 더 우수한 성능을 낼 수 있었음
- 제안하는 모델은 train 시 가장 빠른 것은 아니지만, inference의 속도에 중점을 둠
- 최신 모델들의 연구 결과를 반영하여 Transformer 요소를 수정

# 2. Approach

> GPT-3 방식 + Chinchilla scaling laws 에 영향을 받음
> 

## Pre-training Data

### 1. Data Crawling Table

- Total 1.5T tokens 이며, Wiki 와 Books 를 제외하고는 사실 상 학습하는 동안 토큰이 딱 한번씩 쓰이는 꼴임
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/d47e6eaf-cb3d-4e46-96c7-05274485a4e5/image.png)
    

### 2. Data Details [Total 1.5T Tokens]

- English CommonCrawl [67%]
    - `2017 ~ 2020` 까지의 데이터를 CCNet pipeline 에 따라 preprocessing
    - 별도 fastText linear classifier 를 둬서 non-English 데이터를 클렌징함
    - wikipedia 에서 references 로 사용되는 page 를 분류하는 모델도 별도로 둬서 여기에 분류되지 않는 page 들은 모두 버림
        - crawling 이다보니 퀄리티가 낮은 page 를 버리기 위함
- C4 [15%]
    - CCNet pipeline 으로 preprocessing 하여 사용 (quality filtering 을 위해 사용하는 방식으로, 대부분 heuristics 에 기반)
- Github [4.5%]
    - public GitHub datasets on Google BigQuery
        - Apache, BSD, MIT lincenses
- Wikipedia [4.5%]
    - June-August 2022 period version, covering 20 languages
        - `bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk (kr은 빠졌음)`
- Gutenberg and Books3 [4.5%]
- ArXiv [2.5%]
    - scientific data 로 사용, Latex files 형태로
- Stack Exchange [2%]high quality Q&A 데이터를 위해 넣음
    
    > [참고] Stack Exchange는 다양한 분야의 주제에 대한 질의 응답 웹사이트 네트워크로, 각 사이트는 특정 주제를 다루며 질문, 답변 및 사용자는 평판 수여 프로세스의 대상이 됩니다.
    > 

### 3. Tokenizer

- SentencePiece 에서 구현된 BPE algorithm 을 사용

## Architecture

- GPT-3 방식의 Transformer 구조이나, 아래 설명하는 대로 몇 가지 원래 구조와 다르게 사용한 것이 있음

### 1. Pre-normalization [GPT3]

- stability 를 위해 각 transformer sub-layer 의 input 에 normalization 적용 (https://arxiv.org/pdf/2002.04745.pdf)
- input 에 normalization 적용하여 gradient vanishing 문제 해결
- 빠른 수렴 속도, 일반화 성능
- 기존 layerNorm 사용하는 것 대신 RMSNorm 을 normalizing function 으로 사용 (https://arxiv.org/abs/1910.07467, NeurIPS 2019)
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/48d09b2f-2d8a-4675-a8a5-355ea5f5beeb/image.png)
    

### 2. SwiGLU activation function [PaLM]

- PaLM 에서 사용한 방법이며, GLU dimension 크기를 기존 4d 를 쓰는 것 대신 2/3 * 4d 를 사용하는 것으로 수정했다고 함
- 다양한 activation function 중 Swish function 을 사용하고, Gated Linear Units (GLU) 를 적용한 것이 [SwiGLU](https://velog.io/@wkshin89/MLDL-SwiGLU) 임
- ReLU 대신 [SwiGLU](https://velog.io/@wkshin89/MLDL-SwiGLU) 사용 (https://arxiv.org/abs/2002.05202, Google 2020)
    - 이는 performance 향상이 목적
    - 다양한 activation function 에 대해 FFN 에 GLU 를 적용하였을 때, 성능 향상이 있었음
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/a576e986-e3b3-474c-a3cb-d84557335e57/image.png)
        

### 3. Rotary Embeddings [GPTNeo]

- 아이디어 제안은 다른 논문에서 시작 되었으나 ([RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)), EleutherAI 에서 GPT-J 구현에 사용하여 유명해진 방법
    - RoPE 는 Relative PE 기반의 방법이고, Additive form 이 아닌 Multiplicative 기법 + Sinusoid 함수를 활용한 방법
    - Token embedding 을 complex form 으로 매핑을 하고, position 정보는 이 token embedding 을 rotation 시켜서 표현할 수 있음
- absolute positional embeddings 사용하는 것 대신 rotary positional embeddings (RoPE) 사용했다고 함
- each layer 마다 RoPE 적용
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/dd3d59bf-a2cb-4a37-aa11-9ca8ba626726/image.png)
    

## Optimizer

- AdamW 사용 (0.9/0.95)
- cosine learning rate schedule, final learning rate = 10% of max
- weight decay = 0.1, gradient clipping = 1.0
- 2000 warm-up steps
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/e9af2f7d-1ec5-48c3-a043-92a6234ca220/image.png)
    

## Efficient Implementation

- casual multi-head attention 계산 시, memory/runtime 감소를 위해 별도 구현된 library 를 사용함
    - xformer library 사용
    - backward 에서는 Flashattention 사용 (https://arxiv.org/abs/2205.14135, 2022)
        - attention weights 를 저장하지 않고, LM task 에서 masked 되는 부분은 key/query 를 계산하지 않게끔 함
- backward pass 동안 다시 계산되는 activations 양을 줄임
    - 즉, linear layers 의 output 과 같은 계산량이 많은 activations 를 미리 저장해 놓는 방식
    - pytorch autograd 말고 직접 backward function 을 구현하였음
- Reducing activation recomputation in large transformer models (https://arxiv.org/pdf/2205.05198.pdf) 논문에 소개된 것 처럼 model and sequence parallelism 을 적용하여 memory usage 감소
- activation 계산과 GPU 간의 communication 을 최대한 overlap 시켜서 최적화 하였음 (all_reduce 같은 연산)
- 65B 모델의 경우, 380 tokens/sec/GPU on `2048 A100 GPU with 80GB of RAM` 이고, 이를 환산하면 1.4T tokens 학습 시 `21일` 걸림
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/801efd95-03e4-4b99-8566-d8cb20e59beb/image.png)
    

## 3. Main Results

- Multi choice tasks 의 경우, context 를 주고 주어진 options 중 적절한 completion 을 고르는 방식임
    - the highest likelihood 를 선택 (given the provided context)
    - completion 에 대하여 characters 수로 normalized 된 likelihood 사용

### 1. Common Sense Reasoning

- Multiple choice tasks
- 8개의 benchmarks 결과 측정하였음
    - BoolQ, PIQA, SIQA, HellaSwag, WinoGrande ARC-e ARC-c, OBQA
    - https://zenodo.org/record/7413426#.ZBBJuOxByrM (**EleutherAI)** 에서 일반적으로 사용하는 evaluation 방식 사용
- 측정 방법
    - 주어진 context 에 대하여 가장 높은 likelihood 값을 갖는 보기를 choice
    - 데이터셋에 따라 2가지 방식으로 normalized 된 likelihood 사용
        1. 일반적인 Common Sence Reasoning Dataset: character 수에 의해 normalized 된 likelihood
        2. OpenBookQA, BoolQ: 위와 같은 normalize 가 적용이 안되고 아래와 같은 방식으로 normalize
            - `context` 가 given 일 때의 likelihood / `answer` 가 given 일 때의 likelihood
- 데이터 예시 (SIQA)
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/70d3f0a5-e4c9-4b8c-b2b9-7a3dc1655ae0/image.png)
    

### Zero-shot 결과

- 다른 LLM 과 비교하였을 때 model capacity 대비 우수한 성능을 보여줌
- `GPT-3 (175B)` 와 비교하였을 때는 약 10배 정도 작은 사이즈의 `LLaMA-13B` 모델이 대체로 우수한 성능을 보임
- `Chinchilla (70B)` 와 유사한 성능을 보였는데, `LLaMA-65B` 는 1) Public data 만을 사용, 2) Open Model 이기 때문에 더욱 의미가 있음
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/413f8062-4ac7-411b-8c8c-4c2a5dd43d90/image.png)
    

### 2. Closed-book Question Answering

> Open-book QA: passage, documents 등을 주고 주어진 question 에 대한 답을 찾는 QA, Search 과정 필요
> 
> 
> `Closed-book QA`: context 없이 곧바로 question 을 던지는 QA (참고: T5), 생성
> 
- 측정 방법
    - Exact match performance 사용
    - 생성 후에 answer 를 뽑아냄 (개행 문자, 마침표, 쉼표가 나오면 stop)
    - normalization 후에 answer 리스트 중 어느 하나와 매칭이 된다면 correct 로 간주
        - 보통 데이터셋의 answer 는 다양한 candidates 형태를 포함하게끔 되어 있음
        - normalization: lowercase, remove articles, punctuation, duplicate whitespace
            
            ![image](https://velog.velcdn.com/images/wkshin89/post/5679955e-641c-490d-b6ca-7632bb248538/image.png)
            
- 주목할만한 결과로, `LLaMA-13B` 의 결과가 경쟁 모델과 비교하였을 때 (GPT-3, Gopher, Chinchilla) competitive 함을 알 수 있음
    - `LLaMA-13B` 모델의 경우, single GPU 추론이 가능함

### Natural Questions 결과

- 데이터 예시 (Natural Questions)
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/e1685b9d-79ce-4b36-8ebb-8e0a6700eb9b/image.png)
    
- LLaMA-65B 모델은 SOTA 달성
- LLaMA-13B 모델은 Chinchilla 와 견줄 수 있는 성능이 나왔고, GPT-3 보다 성능이 좋음
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/b77997b4-1a38-4a94-8572-005bf3e42adf/image.png)
    

### TriviaQA 결과

> TriviaQA is a realistic text-based question answering dataset which includes 950K question-answer pairs from 662K documents collected from Wikipedia and the web.
> 
- LLaMA-65B 모델은 SOTA 달성
- LLaMA-13B 모델은 Chinchilla 와 견줄 수 있는 성능이 나옴
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/6f2eea43-b1ee-44de-88f0-4f8685a78f83/image.png)
    

### 3. Reading Comprehension

- RACE benchmark 에 대하여 평가 진행 (zero-shot)
    - meddle and high school Chinese students 대상으로 하는 독해 시험
- 데이터 예시 (RACE)
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/a2ab02a3-7297-4c29-acdd-78a6f7f82b48/image.png)
    
- `LLaMA-13B` 가 `GPT-3 (175B)` 보다 성능이 좋음
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/21b40be4-2042-45d2-8ee3-cd7a2747dd04/image.png)
    

### 4. Mathematical Reasoning

- MATH, GSM8K 데이터셋 사용
- `Minerva` 모델과의 비교 (maj1@k metric: k samples 에 대한 majority voting)
    - Minerva: finetuned PaLM model (38.5B tokens from ArXiv, Math Web Pages)
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/95584628-a4c3-4b55-8e23-52ed82935e77/image.png)
        

### 5. Code Generation

- description 으로부터 코드를 생성하는 능력을 검증
    - HumanEval (zero-shot), MBPP (3-shot) 데이터셋 사용
- pass@k metric: k samples 생성 후, 1개라도 해결하면 맞다고 판단하는 지표
- finetuned 되지 않은 모델 간의 비교이고, 모델 크기 대비 우수한 성능을 보여줌
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/9401c537-7541-4c6b-93df-9966946ed3be/image.png)
    

### 6. Massive Multitask Language Understanding

- MMLU benchmark 데이터셋 사용
    - humanities, STEM, social sciences
- 5-shot setting 으로 진행
- GPT-3 보다는 우수하지만, PaLM 이나 Chinchilla 보다는 떨어지는 성능을 보여줌
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/b1416420-0ed9-4418-930c-9e9026d3f41f/image.png)
    

## 4. Instruction Finetuning

- 3.6 Section 의 `MMLU task` 에 대하여, 기존 LLaMA 도 어느 정도 성능이 나오지만 적은 양의 fine-tuning 이 추가되었을 때 성능 향상을 보임
- Instruction model 학습을 위한 protocol 로는 Google 의 `Flan` 방식을 사용함 (https://arxiv.org/pdf/2210.11416.pdf, Google 2022)
    - 기존 Flan 의 scaling up
    - Add CoT in fine-tuning (for Reasoning Task)
- `LLaMA-I`: LLaMA-65B fine-tuned with the protocol and instruction dataset
- Instruction model 인 `OPT-IML`, `Flan-PaLM` 과 비교하여 성능이 우세
    - 그러나 이 task 에 대하여 당시 SOTA 는 OpenAI 의 `code-davinci-002` (GPT3.5 + Optimized for code-completion tasks) 의 77.4
    - Closed product model 의 성능은 강력함
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/db82ee40-1f44-4e79-bc0c-cc4ab3edb42c/image.png)
    

## 5. Bias, Toxicity and Misinformation

- LM community 에서 사용되는 몇 가지 benchmarks 가 있지만, 사실상 fully understand (검증) 하기에는 충분하지 않음

### 1. RealToxicityPrompts [Toxicity]

- toxic language (insults, hate speech or threats) 에 대한 benchmark dataset
    - 100k prompts -> PerspectiveAPI request 에 의해 자동으로 score 가 계산되는 방식
    - 0 (non-toxic) to 1 (toxic) score
- model size 에 따라 toxicity 가 증가하는 경우도 있고, 특히 Respectful 부분에서 score 가 Basic 보다도 증가하는 경우가 생김
    - Respectful: `Complete the following sentence in a polite, respectful, and unbiased manner` prompt 가 앞에 붙음
    - Basic: 위 prompt 가 붙지 않음
    - `Gopher (280B)` 에서도 이와 같은 현상이 발견되었는데, 명확한 이유를 찾지는 못함
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/335d394d-f0b2-4dcd-9363-32c24d1ce96f/image.png)
    

### 2. CrowS-Pairs [Bias]

- 9개의 카테고리에 대하여 bias 를 측정 (https://arxiv.org/pdf/2010.00133.pdf)
    - gender, religion, race/color, sexual orientation, age, nationality, disability, physical appearance, socioeconomic status
- Detail 내용
    - 각 카테고리 별 데이터는 stereotype, anti-stereotype 의 pair 로 구성됨
    - zero-shot 으로 두 문장의 perplexity 를 각각 계산하여 stereotype 의 선호도를 측정
    - stereotype 문장을 선호할 수록 더 bias 된 모델로 판정
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/128647a2-6495-4107-bb13-44412c9706a4/image.png)
        
- Bias 결과
    - 값이 높을 수록 bias 된 모델
    - Religion, Sexual orientation 등 몇 가지 항목에서 높은 bias 결과가 나왔는데, 이는 여러 필터링에도 불구하고도 남은 CommonCrawl 데이터에 의한 bias 라고 생각됨
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/026066ec-504d-4bc3-b8b2-59b824cd956c/image.png)
        

### 3. WinoGender [Bias-Gender]

- Bias 중 gender 카테고리 쪽을 더 조사하기 위해 사용한 benchmark dataset (co-reference resolution dataset)
    - 대명사 gender 에 의해 co-reference resolution 성능이 얼마나 영향을 받는 지 평가
    - [참고] WinoGender Schema
        - https://github.com/rudinger/winogender-schemas
        - https://aclanthology.org/N18-2002.pdf
        - Occupation, Participant, Pronoun 으로 구성
            - Pronoun 을 바꿔가면서 가리키는 mention 이 occupation 인 지, participant 인 지 확인
            - gender bias 가 심할 수록 특정 gender 일 때의 성능이 낮아지게 됨 (특정 gender 에서 주로 나타나는 occupation 에 대한 편향)
- model 에서 측정하는 방법: 직업과 관련된 사회적 bias 를 모델이 과연 잡아낼 수 있는 지 확인 (e.g. nurse)3가지 pronouns 를 사용하여 모델의 성능 평가her/her/she, his/him/he, their/them/someone 대명사에 대하여 각각 multiple choice scoring
    
    > [예시]
    > 
    > 
    > The nurse notified the patient that his shift would be ending in an hour. `his`refers to:
    > 
    > - p(nurse), p(patient) etc
- Results
    - her/her/she or his/him/he 의 경우, 문장의 해석보다 특정 직업의 majority gender 에 의해 co-reference resolution 을 하는 경향을 보임
    - there/them/someone 에서는 상당한 성능 향상을 보임
    - All 이 높을 수록 bias 전체적인 bias 가 적다는 의미
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/a03b7530-4939-4a26-9033-ba16049c6ca5/image.png)
        

### 4. TruthfulQA

- model 의 truthfulness 측정
    - 어떠한 claim 이 `true` (literal truth about the real world) 인 지 아닌 지를 구분할 수 있는 지 확인
    - `misinformation` or `false claims` 생성 여부를 평가
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/933c5a7f-a1a8-4fff-8e52-43284a0c43ca/image.png)
        
- [참고] `GPT-3`, `InstructGPT` 에서 사용한 prompt 방식
    
    ![image](https://velog.velcdn.com/images/wkshin89/post/adb0878d-74ab-43f0-befb-47ef22c1d56d/image.png)
    
- Results
    - GPT-3 보다 모든 지표에 대하여 스코어가 높게 나옴. 그러나, 옳은 대답을 하는 비율은 여전히 낮아서 이는 hallucination 가능성이 있음을 시사
    - 이후 RLHF 방식까지 더해진 `InstructGPT` 의 경우, GPT-3 보다 해당 benchmark 결과가 대략 2배 정도 좋았다고 함 (Table 결과에는 없음)
        
        ![image](https://velog.velcdn.com/images/wkshin89/post/d8c44f2b-2db6-465a-8e5a-dc6502403e79/image.png)
        

## 6. Conclusion

- LLM 이 점점 모델/데이터 측면에서 규모가 커짐에 따라, 두 가지 중요한 scaling laws 에 대한 흐름을 이해할 필요가 있음
- 이 중 Deepmind (Chinchilla 모델) 에서 발표한 Compute-Optimal 실험 결과에 기반하여 LLaMA 학습을 진행
    - Transformer 기반 LM 에서 given compute budget 하에 optimal model size 와 tokens 수와의 관계
    - 모델 크기만 키웠던 기존 LLM 들은 상당히 undertrained 되어 있음
- 그 결과 LLaMA-13B 는 작은 사이즈임에도 GPT-3 (175B) 보다 우수한 성능을, LLaMA-65B 는 closed-model 인 Chinchilla 와 유사하거나 뛰어난 성능을 보였음
- 그리고 모델을 open 하여 아직 정확히 설명되지 않는 다양한 LLM 현상들에 대하여 AI 연구자들이 연구할 수 있도록 그 기틀을 마련함
    - 이후에 LLaMA 를 기반으로 한 수 많은 개선된 LLM 들이 등장할 수 있었음
