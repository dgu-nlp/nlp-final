#!/usr/bin/env python3
"""
WikiText 데이터셋 준비 스크립트

WikiText-2 또는 WikiText-103 데이터셋을 다운로드하고 전처리하여
k-NN 검색을 위한 대규모 데이터스토어 구축에 사용할 수 있도록 함.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import logging
import requests
import tarfile
from tqdm import tqdm
import numpy as np

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiTextDataset(Dataset):
    """WikiText 데이터셋 로더"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: GPT2Tokenizer,
                 max_length: int = 512,
                 split: str = 'train'):
        """
        Args:
            data_path: WikiText 데이터 파일 경로
            tokenizer: GPT-2 토크나이저
            max_length: 최대 시퀀스 길이
            split: 데이터셋 분할 ('train', 'validation', 'test')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 문서 분할 (빈 줄로 구분)
        self.documents = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
        
        # 토큰화
        self.tokenized_docs = []
        for doc in tqdm(self.documents, desc=f"Tokenizing {split} documents"):
            tokens = self.tokenizer.encode(
                doc,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )[0]
            self.tokenized_docs.append(tokens)
            
        logger.info(f"로드된 문서 수: {len(self.documents)}")
        logger.info(f"토큰화된 문서 수: {len(self.tokenized_docs)}")
        
    def __len__(self):
        return len(self.tokenized_docs)
        
    def __getitem__(self, idx):
        tokens = self.tokenized_docs[idx]
        attention_mask = torch.ones_like(tokens)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        
    def collate_fn(self, batch):
        """배치 생성 함수"""
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # 패딩
        max_len = max(len(ids) for ids in input_ids)
        input_ids = [torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in input_ids]
        attention_mask = [torch.cat([mask, torch.zeros(max_len - len(mask), dtype=torch.long)]) for mask in attention_mask]
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask)
        }


def download_wikitext(version: str = '2', target_dir: str = 'data'):
    """
    WikiText 데이터셋 다운로드
    
    Args:
        version: WikiText 버전 ('2' 또는 '103')
        target_dir: 저장할 디렉토리
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # 다운로드 URL
    url = f"https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-{version}-raw-v1.zip"
    target_path = os.path.join(target_dir, f"wikitext-{version}-raw-v1.zip")
    
    # 다운로드
    if not os.path.exists(target_path):
        logger.info(f"WikiText-{version} 다운로드 중...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f, tqdm(
            desc=f"Downloading WikiText-{version}",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    else:
        logger.info(f"WikiText-{version}가 이미 존재합니다: {target_path}")
    
    # 압축 해제
    import zipfile
    with zipfile.ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    logger.info(f"WikiText-{version} 다운로드 및 압축 해제 완료")
    
    return os.path.join(target_dir, f"wikitext-{version}-raw")


def prepare_wikitext_datastore(args):
    """WikiText 데이터스토어 구축"""
    logger.info("WikiText 데이터스토어 구축을 시작합니다...")
    
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 데이터 다운로드
    data_dir = download_wikitext(args.version, args.data_dir)
    
    # 토크나이저 초기화
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 데이터셋 생성
    train_dataset = WikiTextDataset(
        os.path.join(data_dir, f'wiki.train.raw'),
        tokenizer,
        max_length=args.max_length,
        split='train'
    )
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 데이터스토어 구축 시에는 순서 유지
        collate_fn=train_dataset.collate_fn
    )
    
    # 기본 GPT-2 모델 로드
    from models.gpt2 import GPT2Model
    model = GPT2Model.from_pretrained('gpt2')
    model = model.to(device)
    
    # 데이터스토어 구축
    from knn_gpt import DataStore
    datastore = DataStore(hidden_size=768, device=device)
    save_path = os.path.join(args.data_dir, f"wikitext_{args.version}_datastore.pt")
    
    datastore.build_from_model_and_data(
        model=model,
        dataloader=train_dataloader,
        save_path=save_path
    )
    
    logger.info(f"WikiText 데이터스토어 구축 완료: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="WikiText 데이터셋 준비 및 데이터스토어 구축")
    
    parser.add_argument("--version", type=str, default='2',
                        choices=['2', '103'])
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="배치 크기")
    parser.add_argument("--use_gpu", action='store_true',
                        help="GPU 사용 여부")
    
    args = parser.parse_args()
    
    try:
        datastore_path = prepare_wikitext_datastore(args)
        logger.info(f"wikitext 데이터스토어 구축: {datastore_path}")
    except Exception as e:
        logger.error(f"데이터스토어 구축 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 