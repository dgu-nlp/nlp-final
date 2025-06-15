#!/usr/bin/env python3
"""
데이터스토어 구축 스크립트

기존에 훈련된 GPT-2 모델을 사용하여 k-NN 검색을 위한 데이터스토어를 구축함.
이 스크립트는 모든 작업(분류, 패러프레이즈 탐지, 소넷 생성)과 WikiText에 대해 데이터스토어를 생성함.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import logging
import pickle

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn_gpt import DataStore
from models.gpt2 import GPT2Model
from classifier import GPT2SentimentClassifier, SentimentDataset, load_data
from datasets import ParaphraseDetectionDataset, SonnetsDataset, load_paraphrase_data
from paraphrase_detection import ParaphraseGPT, add_arguments as add_para_arguments
from sonnet_generation import SonnetGPT, add_arguments as add_sonnet_arguments
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_classifier_datastore(args):
    """감성 분류를 위한 데이터스토어 구축"""
   
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    if args.dataset == 'sst':
        train_data, num_labels = load_data('data/ids-sst-train.csv', 'train')
        model_path = 'sst-classifier.pt'
    elif args.dataset == 'cfimdb':
        train_data, num_labels = load_data('data/ids-cfimdb-train.csv', 'train')
        model_path = 'cfimdb-classifier.pt'
    else:
        raise ValueError(f"지원되지 않는 데이터셋: {args.dataset}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SentimentDataset(train_data, args)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False,  # 데이터스토어 구축 시에는 순서 유지
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )
    
    # 모델 로드
    if os.path.exists(model_path):
        saved = torch.load(model_path, map_location='cpu')
        config = saved['model_config']
        model = GPT2SentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        logger.info(f"모델을 {model_path}에서 로드했습니다.")
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 데이터스토어 구축
    datastore = DataStore(hidden_size=768, device=device)
    save_path = f"datastores/{args.dataset}_classifier_datastore.pt"
    
    datastore.build_from_model_and_data(
        model=model,
        dataloader=train_dataloader,
        save_path=save_path
    )
    
    logger.info(f"감성 분류용 데이터스토어 구축 완료: {save_path}")
    
    # 메모리 해제
    del model, datastore, train_dataset, train_dataloader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return save_path


def build_paraphrase_datastore(args):
    """패러프레이즈 탐지를 위한 데이터스토어 구축"""
    
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    train_data = load_paraphrase_data('data/quora-train.csv')
    train_dataset = ParaphraseDetectionDataset(train_data, args)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )
    
    # 모델 파라미터 설정
    args = add_para_arguments(args)
    
    # 모델 로드
    model_path = f'{args.epochs}-{args.lr}-paraphrase.pt'
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location='cpu')
            model = ParaphraseGPT(saved['args'])
            model.load_state_dict(saved['model'])
        except Exception as e:
            logger.warning(f"기존 방식으로 모델 로드 실패: {e}")
            # 대안적인 로드 방식 시도
            saved = torch.load(model_path, map_location='cpu', pickle_module=pickle)
            model = ParaphraseGPT(saved['args'])
            model.load_state_dict(saved['model'])
        model = model.to(device)
    else:
        raise FileNotFoundError(f"모델 파일 오류: {model_path}")
    
    # 데이터스토어 구축
    datastore = DataStore(hidden_size=args.d, device=device)
    save_path = f"datastores/paraphrase_datastore.pt"
    
    datastore.build_from_model_and_data(
        model=model,
        dataloader=train_dataloader,
        save_path=save_path
    )
    
    logger.info(f"패러프레이즈 탐지용 데이터스토어 구축 완료: {save_path}")
    
    # 메모리 해제
    del model, datastore, train_dataset, train_dataloader, saved
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return save_path


def build_sonnet_datastore(args):
    """소넷 생성을 위한 데이터스토어 구축"""
    
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    sonnet_dataset = SonnetsDataset('data/sonnets.txt')
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn
    )
    
    # 모델 파라미터 설정
    args = add_sonnet_arguments(args)
    
    # 모델 로드
    model_path = f'{args.epochs-1}_{args.epochs}-{args.lr}-sonnet.pt'
    if os.path.exists(model_path):
        try:
            saved = torch.load(model_path, map_location='cpu')
            model = SonnetGPT(saved['args'])
            model.load_state_dict(saved['model'])
        except Exception as e:
            logger.warning(f"기존 방식으로 모델 로드 실패: {e}")
            # 대안적인 로드 방식 시도
            saved = torch.load(model_path, map_location='cpu', pickle_module=pickle, weights_only=False)
            model = SonnetGPT(saved['args'])
            model.load_state_dict(saved['model'])
        model = model.to(device)
    else:
        raise FileNotFoundError(f"모델 파일 오류: {model_path}")
    
    # 데이터스토어 구축
    datastore = DataStore(hidden_size=args.d, device=device)
    save_path = f"datastores/sonnet_datastore.pt"
    
    datastore.build_from_model_and_data(
        model=model,
        dataloader=sonnet_dataloader,
        save_path=save_path
    )
    
    logger.info(f"소넷 생성용 데이터스토어 구축 완료: {save_path}")
    
    # 메모리 해제
    del model, datastore, sonnet_dataset, sonnet_dataloader, saved
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return save_path


def build_wikitext_datastore(args):
    """WikiText 데이터스토어 구축"""
    
    from scripts.prepare_wikitext import prepare_wikitext_datastore
    
    # Namespace 객체 생성하여 필요한 파라미터 전달
    wikitext_args = SimpleNamespace(
        version=args.version,
        data_dir=args.data_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )
    
    datastore_path = prepare_wikitext_datastore(wikitext_args)
    
    # 메모리 해제 (prepare_wikitext_datastore 내부에서 사용된 리소스는 해제할 수 없지만,
    # 시스템 메모리 정리를 위해 가비지 컬렉션 강제 실행)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return datastore_path


def main():
    parser = argparse.ArgumentParser(description="k-NN 검색을 위한 데이터스토어 구축")
    
    # 기본 인수
    parser.add_argument("--task", type=str, required=True,
                        choices=['classifier', 'paraphrase', 'sonnet', 'wikitext', 'all'])
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default='datastores',
                        help="데이터 파일이 저장된 디렉토리")
    
    # 분류기 관련 인수
    parser.add_argument("--dataset", type=str, default='sst',
                        choices=['sst', 'cfimdb'])
    
    # 패러프레이즈 관련 인수
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_size", type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    
    # WikiText 관련 인수        
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--version", type=str, default='2',
                        choices=['2', '103'])
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs("datastores", exist_ok=True)
    
    built_datastores = []
    
    if args.task == 'classifier' or args.task == 'all':
        # SST와 CFIMDB 모두 구축
        if args.task == 'all':
            for dataset in ['sst', 'cfimdb']:
                args.dataset = dataset
                try:
                    datastore_path = build_classifier_datastore(args)
                    built_datastores.append(datastore_path)
                except Exception as e:
                    logger.error(f"{dataset} 분류기 데이터스토어 구축 실패: {e}")
        else:
            try:
                datastore_path = build_classifier_datastore(args)
                built_datastores.append(datastore_path)
            except Exception as e:
                logger.error(f"분류기 데이터스토어 구축 실패: {e}")
    
    if args.task == 'paraphrase' or args.task == 'all':
        try:
            datastore_path = build_paraphrase_datastore(args)
            built_datastores.append(datastore_path)
        except Exception as e:
            logger.error(f"패러프레이즈 데이터스토어 구축 실패: {e}")
    
    if args.task == 'sonnet' or args.task == 'all':
        try:
            datastore_path = build_sonnet_datastore(args)
            built_datastores.append(datastore_path)
        except Exception as e:
            logger.error(f"소넷 데이터스토어 구축 실패: {e}")
    
    if args.task == 'wikitext' or args.task == 'all':
        try:
            datastore_path = build_wikitext_datastore(args)
            built_datastores.append(datastore_path)
        except Exception as e:
            logger.error(f"WikiText 데이터스토어 구축 실패: {e}")
    
    # 결과 요약
    logger.info("\n=== 데이터스토어 구축 완료 ===")
    for path in built_datastores:
        logger.info(f"✓ {path}")
    
    if not built_datastores:
        logger.warning("구축된 데이터스토어가 없습니다.")
    else:
        logger.info(f"총 {len(built_datastores)}개의 데이터스토어가 구축되었습니다.")


if __name__ == "__main__":
    main() 