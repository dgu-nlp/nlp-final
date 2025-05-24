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

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn_gpt import DataStore
from models.knn_gpt2 import KNNAugmentedGPT2
from paraphrase_detection import ParaphraseGPT
from sonnet_generation import SonnetGPT
from datasets import load_paraphrase_data, SonnetsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if args.use_wikitext:
        # WikiText 데이터스토어 사용
        datastore_path = os.path.join(args.data_dir, f"wikitext_{args.wikitext_version}_datastore.pt")
    else:
        # 작업별 데이터스토어 사용
        datastore_path = os.path.join(args.data_dir, f"{task}_datastore.pt")
        
    if not os.path.exists(datastore_path):
        raise FileNotFoundError(f"데이터스토어 파일 오류: {datastore_path}")
        
    datastore = DataStore(hidden_size=768)
    datastore.load(datastore_path)
    return datastore


def run_paraphrase_detection(args: argparse.Namespace):
    """k-NN 증강 패러프레이즈 탐지 실행"""
    # 모델 로드
    knn_model, base_model = load_knn_model('paraphrase', args)
    
    # 데이터스토어 로드 및 설정
    datastore = load_datastore('paraphrase', args)
    knn_model.set_datastore(datastore)
    
    # 테스트 데이터 로드
    test_data = load_paraphrase_data('data/quora-test.csv')
    
    # 평가
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    knn_model = knn_model.to(device)
    base_model = base_model.to(device)
    
    # k-NN 활성화/비활성화 비교
    for use_knn in [False, True]:
        if use_knn:
            logger.info("k-NN 증강 모드로 평가 중...")
            model = knn_model
        else:
            logger.info("기본 모드로 평가 중...")
            model = base_model
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_data:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
        accuracy = correct / total
        logger.info(f"{'k-NN 증강' if use_knn else '기본'} 모드 정확도: {accuracy:.4f}")


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
    
    # k-NN 활성화/비활성화 비교
    for use_knn in [False, True]:
        if use_knn:
            logger.info("\nk-NN 증강 모드로 생성 중...")
            model = knn_model
        else:
            logger.info("\n기본 모드로 생성 중...")
            model = base_model
            
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