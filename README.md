# nlp-final

## 환경구축
```
# 추가라이브러리 설치(knn 검색용)
pip install faiss-cpu

```
## 실행
```
#!/bin/bash

# 데이터스토어 구축
echo "데이터스토어 구축 중..."
python scripts/build_datastore.py --task all --use_gpu

# 패러프레이즈 탐지 (기본 데이터스토어 사용)
echo -e "\n패러프레이즈 탐지 실행 (기본 데이터스토어)..."
python scripts/run_knn_augmented.py \
    --task paraphrase \
    --use_gpu \
    --k 8 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation

# 패러프레이즈 탐지 (WikiText 데이터스토어 사용)
echo -e "\n패러프레이즈 탐지 실행 (WikiText 데이터스토어)..."
python scripts/run_knn_augmented.py \
    --task paraphrase \
    --use_gpu \
    --k 8 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation \
    --use_wikitext \
    --wikitext_version 2

# 소넷 생성 (기본 데이터스토어 사용)
echo -e "\n소넷 생성 실행 (기본 데이터스토어)..."
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
echo -e "\n소넷 생성 실행 (WikiText 데이터스토어)..."
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

# 실험용 추가 실행 (다른 k 값으로)
echo -e "\n패러프레이즈 탐지 실행 (k=16)..."
python scripts/run_knn_augmented.py \
    --task paraphrase \
    --use_gpu \
    --k 16 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation

echo -e "\n소넷 생성 실행 (k=16)..."
python scripts/run_knn_augmented.py \
    --task sonnet \
    --use_gpu \
    --k 16 \
    --lambda_knn 0.25 \
    --knn_temperature 10.0 \
    --use_quality_filter \
    --use_adaptive_interpolation \
    --max_length 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample 
```