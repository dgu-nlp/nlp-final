# nlp-final

> [!caution]
> runpod 플랫폼을 이용해 동작 환경을 구성했습니다.
> ⚠️ **H100NVL, networkstorage 80GB**를 사용했습니다.
>
> 코드 개발과 디버깅을 포함해 전체 **$45 정도** 사용했습니다.
>
> 실행 시간은, 기본 과제 모델이 있다면 **약 3시간** 소모됩니다.(datastore 구성을 위해 토크나이저를 저장하는 시간이 오래 걸립니다.)
> 
> 기본 과제 모델이 없는 경우, **10시간** 소모됩니다.(paraphrase 10epoch를 모두 수행할 경우)

## 결과
Urvashi Khandelwal et.al의 knn-lm논문을 이용하여 기본과제와 knn-gpt를 구현했습니다.

- 기본 과제
    - Language models are unsupervised multitask learners. OpenAI Blog를 참고하여 foward layer 등을 구현했습니다.

- knn을 이용한 패러프레이즈
    - knn을 이용한 패러프레이즈 탐지는 data 디렉토리의 **quora-test-student.csv**를 입력으로
**knn_para-test-output_default.csv** 파일을 생성합니다. 

- knn을 이용한 소넷생성
    - knn을 이용한 소넷생성 테스크는 data 디렉토리의 **sonnet_held_out_dev.txt**를 이용하여
**knn_generated_sonnet.txt**를 생성합니다.

>[!note]
>결과로, 기본적인 sonnet_generation.py는 **28.26**
>
>knn을 이용한 sonnet_generation은 **30.14**
>
>기본적인 paraphrase_detection.py는 **0.868**의 dev acc 결과를 얻었습니다.
>
>knn을 이용한 paraphrase_detection은 **0.8968**의 dev acc 결과를 얻었습니다.

## 환경 구성 및 실행

#### 1. miniconda install
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
```
source ~/.bashrc
```
#### 2. git clone
```
cd /workspace
git clone https://github.com/dgu-nlp/nlp-final
cd nlp-final
``` 

#### 3. conda env setting
>[!NOTE]
> knn-lm의 효율적인 검색을 위해 fassis 라이브러리를 추가했습니다.
>
> knn-lm의 datastore 구축을 위해 datasets 라이브러리를 추가했습니다.
>
> 기본구현에서는 사용하지 않았습니다. 
```
conda env create -f env.yml
conda activate nlp_final
```
#### 4. 기본 과제 실행
```
python classifier.py --use_gpu
python paraphrase_detection.py --use_gpu
python sonnet_generation.py --use_gpu
```
#### 5. knn-gpt 실행
```
#!/bin/bash
# 기본과제 실행 이후 또는 모델이 존재할경우.
# 데이터스토어 구축
python scripts/build_datastore.py --task all --use_gpu

# 패러프레이즈 탐지 (기본 데이터스토어 사용)
python scripts/run_knn_augmented.py --task paraphrase --use_gpu --k 8 --max_chunks 8

# 패러프레이즈 탐지 (WikiText 데이터스토어 사용)
python scripts/run_knn_augmented.py --task paraphrase --use_gpu --k 8 --max_chunks 8 --wikitext_version 2

# 패러프레이즈 탐지(dev acc 평가)
python eval_knn_para.py --k 8 --use_gpu  

# 소넷 생성 (기본 데이터스토어 사용)
python scripts/run_knn_augmented.py \
    --task sonnet \
    --use_gpu \
    --k 8 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation \
    --max_length 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample

# 소넷 생성 (WikiText 데이터스토어 사용)
python scripts/run_knn_augmented.py \
    --task sonnet \
    --use_gpu \
    --k 8 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation \
    --use_wikitext \
    --wikitext_version 2 \
    --max_length 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample
```


