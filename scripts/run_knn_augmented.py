#!/usr/bin/env python3
"""
k-NN 증강 모델 실행 스크립트

패러프레이즈 탐지와 소넷 생성을 위한 k-NN 증강 모델을 실행합니다.
데이터스토어를 로드하고 k-NN 검색을 활성화하여 모델을 실행합니다.
"""

import argparse
import os
import sys
import torch
import logging
from typing import Optional, Dict, Any
import csv
import json
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knn_gpt import DataStore
from models.knn_gpt2 import KNNAugmentedGPT2
from paraphrase_detection import ParaphraseGPT
from sonnet_generation import SonnetGPT
from datasets import load_paraphrase_data, ParaphraseDetectionDataset, ParaphraseDetectionTestDataset, SonnetsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 결과 저장 디렉토리 생성
os.makedirs('predictions', exist_ok=True)


def load_knn_model(task: str, args: argparse.Namespace) -> KNNAugmentedGPT2:
    """k-NN 증강 모델 로드"""
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    if task == 'paraphrase':
        # 패러프레이즈 탐지 모델 로드
        model_path = f'{args.epochs}-{args.lr}-paraphrase.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 오류: {model_path}")
            
        saved = torch.load(model_path, map_location='cpu')
        base_model = ParaphraseGPT(saved['args'])
        base_model.load_state_dict(saved['model'])
        base_model = base_model.to(device)
        
        # k-NN 증강 모델 생성
        knn_model = KNNAugmentedGPT2(
            base_model=base_model.gpt,
            k=args.k,
            lambda_knn=args.lambda_knn,
            knn_temperature=args.knn_temperature,
            use_quality_filter=args.use_quality_filter,
            use_adaptive_interpolation=args.use_adaptive_interpolation,
            vocab_size=base_model.gpt.word_embedding.num_embeddings
        )
        
    elif task == 'sonnet':
        # 소넷 생성 모델 로드
        model_path = f'{args.epochs-1}_{args.epochs}-{args.lr}-sonnet.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 오류: {model_path}")
            
        saved = torch.load(model_path, map_location='cpu')
        base_model = SonnetGPT(saved['args'])
        base_model.load_state_dict(saved['model'])
        base_model = base_model.to(device)
        
        # k-NN 증강 모델 생성
        knn_model = KNNAugmentedGPT2(
            base_model=base_model.gpt,
            k=args.k,
            lambda_knn=args.lambda_knn,
            knn_temperature=args.knn_temperature,
            use_quality_filter=args.use_quality_filter,
            use_adaptive_interpolation=args.use_adaptive_interpolation,
            vocab_size=base_model.gpt.word_embedding.num_embeddings
        )
        
    else:
        raise ValueError(f"지원되지 않는 작업: {task}")
        
    return knn_model, base_model


def load_datastore(task: str, args: argparse.Namespace) -> DataStore:
    """데이터스토어 로드"""
    # 디바이스 설정
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:0')  # 명시적으로 cuda:0 사용
        logger.info(f"GPU 디바이스 사용: {device}")
    else:
        device = torch.device('cpu')
        logger.info(f"CPU 디바이스 사용: {device}")
    
    if args.use_wikitext:
        # WikiText 데이터스토어 사용
        datastore_path = os.path.join('datastores', f"wikitext_{args.wikitext_version}_datastore.pt")
    else:
        # 작업별 데이터스토어 사용
        datastore_path = os.path.join('datastores', f"{task}_datastore.pt")
        
    if not os.path.exists(datastore_path):
        raise FileNotFoundError(f"데이터스토어 파일 오류: {datastore_path}")
        
    datastore = DataStore(hidden_size=768, device=device)
    
    try:
        logger.info(f"데이터스토어 로드 중: {datastore_path}")
        data = torch.load(datastore_path, map_location=device)
        
        # 데이터스토어 형식 확인
        if isinstance(data, dict):
            if 'keys' in data:
                # 기본 형식
                datastore.keys = data['keys'].to(device)
                datastore.values = data['values'].to(device)
                datastore.contexts = data.get('contexts', [])
                datastore.size = data['size']
                datastore.hidden_size = data['hidden_size']
                datastore.is_built = data.get('is_built', True)
            elif 'chunk_count' in data:
                # 청크로 나뉜 형식
                logger.info(f"청크 데이터스토어 감지: {data['chunk_count']} 청크")
                base_path = data['base_path']
                
                # 첫 번째 청크만 로드 (메모리 제한)
                chunk_path = f"{base_path}_chunk0.pt"
                if os.path.exists(chunk_path):
                    chunk_data = torch.load(chunk_path, map_location=device)
                    datastore.keys = chunk_data['keys'].to(device)
                    datastore.values = chunk_data['values'].to(device)
                    datastore.contexts = chunk_data.get('contexts', [])
                    datastore.size = chunk_data['size']
                    datastore.hidden_size = chunk_data['hidden_size']
                    datastore.is_built = True
                    
                    logger.info(f"첫 번째 청크 로드 완료: {chunk_path} (크기: {datastore.size})")
                else:
                    raise FileNotFoundError(f"청크 파일을 찾을 수 없음: {chunk_path}")
            else:
                raise ValueError(f"인식할 수 없는 데이터스토어 형식: {list(data.keys())}")
        else:
            raise TypeError(f"데이터스토어가 딕셔너리가 아님: {type(data)}")
        
        logger.info(f"데이터스토어 로드 완료. 크기: {datastore.size}")
    except Exception as e:
        logger.error(f"데이터스토어 로드 중 오류 발생: {e}")
        raise
        
    return datastore


def run_paraphrase_detection(args: argparse.Namespace):
    """k-NN 증강 패러프레이즈 탐지 실행"""
    # 모델 로드
    knn_model, base_model = load_knn_model('paraphrase', args)
    
    # 데이터스토어 로드 및 설정
    datastore = load_datastore('paraphrase', args)
    knn_model.set_datastore(datastore)
    
    # 테스트 데이터 로드
    try:
        logger.info("테스트 데이터 로드 중...")
        test_data = load_paraphrase_data('data/quora-test-student.csv', split='test')
        test_dataset = ParaphraseDetectionTestDataset(test_data, args)
        
        # batch_size가 없는 경우 기본값 설정
        batch_size = getattr(args, 'batch_size', 8)  # 기본값 8
        
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                    collate_fn=test_dataset.collate_fn)
        logger.info(f"테스트 데이터 로드 완료: {len(test_dataset)} 샘플")
    except Exception as e:
        logger.error(f"테스트 데이터 로드 중 오류 발생: {e}")
        raise
    
    # 평가
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    knn_model = knn_model.to(device)
    base_model = base_model.to(device)
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'base_model': {'accuracy': 0, 'predictions': []},
        'knn_model': {'accuracy': 0, 'predictions': []}
    }
    
    # k-NN 활성화/비활성화 비교
    for use_knn in [False, True]:
        if use_knn:
            logger.info("k-NN 증강 모드로 평가 중...")
            model = knn_model
            model_type = 'knn_model'
        else:
            logger.info("기본 모드로 평가 중...")
            model = base_model
            model_type = 'base_model'
            
        model.eval()
        correct = 0
        total = 0
        predictions = []
        sent_ids = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                b_ids, b_mask = batch['token_ids'], batch['attention_mask']
                b_sent_ids = batch['sent_ids']
                
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                
                try:
                    if use_knn:
                        outputs = model(b_ids, b_mask)
                        logits = outputs['logits']
                    else:
                        logits = model(b_ids, b_mask)
                    
                    preds = torch.argmax(logits, dim=1)
                    
                    # 예측 결과와 문장 ID 저장
                    predictions.extend(preds.cpu().numpy().tolist())
                    sent_ids.extend(b_sent_ids)
                except RuntimeError as e:
                    logger.error(f"예측 중 오류 발생: {e}")
                    if "Expected all tensors to be on the same device" in str(e):
                        logger.info(f"텐서 디바이스 불일치 감지. 현재 디바이스: {device}")
                        logger.info(f"모델 디바이스: {next(model.parameters()).device}")
                        if use_knn:
                            logger.info(f"데이터스토어 디바이스: {datastore.device}")
                            logger.info(f"데이터스토어 키 디바이스: {datastore.keys.device}")
                        raise
        
        # 테스트 데이터에는 레이블이 없으므로 정확도를 계산할 수 없음
        logger.info(f"{'k-NN 증강' if use_knn else '기본'} 모드 예측 완료: {len(predictions)} 샘플")
        
        # 결과 저장
        results[model_type]['accuracy'] = 0  # 테스트 데이터에는 레이블이 없음
        results[model_type]['predictions'] = list(zip(sent_ids, predictions))
    
    # 결과 파일로 저장
    datastore_type = 'wikitext' if args.use_wikitext else 'default'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 모델 결과 저장
    base_output_file = f'predictions/para-test-output.csv'
    with open(base_output_file, "w+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Predicted_Is_Paraphrase"])
        for sent_id, pred in results['base_model']['predictions']:
            writer.writerow([sent_id, pred])
    logger.info(f"기본 모델 결과를 {base_output_file}에 저장했습니다.")
    
    # kNN 모델 결과 저장
    knn_output_file = f'predictions/knn_para-test-output_{datastore_type}_k{args.k}.csv'
    with open(knn_output_file, "w+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Predicted_Is_Paraphrase"])
        for sent_id, pred in results['knn_model']['predictions']:
            writer.writerow([sent_id, pred])
    logger.info(f"k-NN 모델 결과를 {knn_output_file}에 저장했습니다.")
    
    # 성능 비교 결과 저장
    comparison_file = f'predictions/knn_para_comparison_{datastore_type}_k{args.k}.json'
    with open(comparison_file, "w+") as f:
        json.dump({
            'parameters': {
                'k': args.k,
                'lambda_knn': args.lambda_knn,
                'knn_temperature': args.knn_temperature,
                'use_quality_filter': args.use_quality_filter,
                'use_adaptive_interpolation': args.use_adaptive_interpolation,
                'datastore_type': datastore_type
            }
        }, f, indent=2)
    logger.info(f"파라미터 정보를 {comparison_file}에 저장했습니다.")


def run_sonnet_generation(args: argparse.Namespace):
    """k-NN 증강 소넷 생성 실행"""
    # 모델 로드
    knn_model, base_model = load_knn_model('sonnet', args)
    
    # 데이터스토어 로드 및 설정
    datastore = load_datastore('sonnet', args)
    knn_model.set_datastore(datastore)
    
    # 토크나이저
    tokenizer = base_model.tokenizer
    
    # 시작 프롬프트
    prompts = [
        "Shall I compare thee to a summer's day?",
        "My mistress' eyes are nothing like the sun",
        "When I do count the clock that tells the time"
    ]
    
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    knn_model = knn_model.to(device)
    base_model = base_model.to(device)
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'base_model': [],
        'knn_model': []
    }
    
    # k-NN 활성화/비활성화 비교
    for use_knn in [False, True]:
        if use_knn:
            logger.info("\nk-NN 증강 모드로 생성 중...")
            model = knn_model
            model_type = 'knn_model'
        else:
            logger.info("\n기본 모드로 생성 중...")
            model = base_model
            model_type = 'base_model'
            
        model.eval()
        
        for prompt in prompts:
            # 입력 토큰화
            inputs = tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # 텍스트 생성
            generated_ids, generated_text = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample
            )
            
            logger.info(f"\n프롬프트: {prompt}")
            logger.info(f"생성된 텍스트:\n{generated_text}")
            
            # 결과 저장
            results[model_type].append({
                'prompt': prompt,
                'generated_text': generated_text
            })
    
    # 결과 파일로 저장
    datastore_type = 'wikitext' if args.use_wikitext else 'default'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 기본 모델 결과 저장
    base_output_file = f'predictions/generated_sonnets.txt'
    with open(base_output_file, "w+") as f:
        f.write("--Generated Sonnets (Base Model)--\n\n")
        for i, result in enumerate(results['base_model']):
            f.write(f"\nPrompt {i+1}: {result['prompt']}\n")
            f.write(f"{result['generated_text']}\n\n")
    logger.info(f"기본 모델 생성 결과를 {base_output_file}에 저장했습니다.")
    
    # kNN 모델 결과 저장
    knn_output_file = f'predictions/knn_generated_sonnets_{datastore_type}_k{args.k}.txt'
    with open(knn_output_file, "w+") as f:
        f.write("--Generated Sonnets (k-NN Model)--\n\n")
        for i, result in enumerate(results['knn_model']):
            f.write(f"\nPrompt {i+1}: {result['prompt']}\n")
            f.write(f"{result['generated_text']}\n\n")
    logger.info(f"k-NN 모델 생성 결과를 {knn_output_file}에 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(description="k-NN 증강 모델 실행")
    
    # 기본 인수
    parser.add_argument("--task", type=str, required=True,
                        choices=['paraphrase', 'sonnet'],
                        help="실행할 작업")
    parser.add_argument("--use_gpu", action='store_true',
                        help="GPU 사용 여부")
    
    # k-NN 관련 인수
    parser.add_argument("--k", type=int, default=8,
                        help="검색할 이웃 수")
    parser.add_argument("--lambda_knn", type=float, default=0.25,
                        help="k-NN 예측 가중치")
    parser.add_argument("--knn_temperature", type=float, default=10.0,
                        help="k-NN 검색 온도")
    parser.add_argument("--use_quality_filter", action='store_true',
                        help="품질 필터링 사용")
    parser.add_argument("--use_adaptive_interpolation", action='store_true',
                        help="적응적 interpolation 사용")
    
    # 데이터스토어 관련 인수
    parser.add_argument("--use_wikitext", action='store_true',
                        help="WikiText 데이터스토어 사용")
    parser.add_argument("--wikitext_version", type=str, default='2',
                        choices=['2', '103'],
                        help="WikiText 버전")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="데이터 디렉토리")
    
    # 모델 관련 인수
    parser.add_argument("--epochs", type=int, default=10,
                        help="훈련 에포크 수")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="학습률")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="배치 크기")
    
    # 생성 관련 인수 (sonnet 작업에만 사용)
    parser.add_argument("--max_length", type=int, default=200,
                        help="최대 생성 길이")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="생성 온도")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="nucleus sampling 파라미터")
    parser.add_argument("--do_sample", action='store_true',
                        help="샘플링 사용")
    
    args = parser.parse_args()
    
    try:
        if args.task == 'paraphrase':
            run_paraphrase_detection(args)
        elif args.task == 'sonnet':
            run_sonnet_generation(args)
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 