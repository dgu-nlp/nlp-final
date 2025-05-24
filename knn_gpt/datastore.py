"""
DataStore: k-NN 검색을 위한 데이터스토어 구축 및 관리

Khandelwal et al. (2019)의 방법을 따라 훈련 데이터의 hidden states를 key로,
다음 토큰을 value로 하는 데이터스토어를 구축
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStore:
    """k-NN 검색을 위한 데이터스토어"""
    
    def __init__(self, hidden_size: int, device: torch.device = None):
        self.hidden_size = hidden_size
        self.device = device or torch.device('cpu')
        
        # 데이터스토어 저장소
        self.keys = []  # List[torch.Tensor] - hidden states
        self.values = []  # List[int] - next token ids
        self.contexts = []  # List[str] - 선택적으로 컨텍스트 저장
        
        # 메타데이터
        self.size = 0
        self.is_built = False
        
    def build_from_model_and_data(self, model, dataloader, save_path: str = None):
        """
        모델과 데이터로더를 사용하여 데이터스토어 구축
        
        Args:
            model: GPT-2 모델
            dataloader: 훈련 데이터 로더
            save_path: 데이터스토어 저장 경로
        """
        logger.info("데이터스토어 구축 시작")
        
        model.eval()
        self.keys = []
        self.values = []
        self.contexts = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Building datastore")):
                if 'token_ids' in batch:
                    input_ids = batch['token_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                else:
                    # sonnet generation의 경우
                    input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else batch['token_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                
                # 모델 forward pass
                outputs = model.gpt(input_ids, attention_mask)
                hidden_states = outputs['last_hidden_state']  # [batch_size, seq_len, hidden_size]
                
                # 각 위치에서 다음 토큰 예측을 위한 데이터 수집
                batch_size, seq_len, _ = hidden_states.shape
                
                for i in range(batch_size):
                    for t in range(seq_len - 1):  # 마지막 토큰은 제외 (다음 토큰이 없음)
                        if attention_mask[i, t] == 1:  # padding이 아닌 토큰만
                            # Key: 현재 위치의 hidden state
                            key = hidden_states[i, t].cpu()
                            # Value: 다음 토큰
                            value = input_ids[i, t + 1].cpu().item()
                            
                            self.keys.append(key)
                            self.values.append(value)
                            
                            # 선택적으로 컨텍스트 저장 (디버깅용)
                            if hasattr(model, 'tokenizer'):
                                context = model.tokenizer.decode(input_ids[i, :t+1].cpu().tolist())
                                self.contexts.append(context)
                
                if batch_idx % 100 == 0:
                    logger.info(f"처리된 배치: {batch_idx}, 현재 데이터스토어 크기: {len(self.keys)}")
        
        # 텐서로 변환
        self.keys = torch.stack(self.keys)  # [num_examples, hidden_size]
        self.values = torch.tensor(self.values, dtype=torch.long)  # [num_examples]
        self.size = len(self.keys)
        self.is_built = True
        
        logger.info(f"데이터스토어 구축 완료. 총 크기: {self.size}")
        
        # 저장
        if save_path:
            self.save(save_path)
            
    def save(self, path: str):
        """데이터스토어를 디스크에 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'keys': self.keys,
            'values': self.values,
            'contexts': self.contexts,
            'size': self.size,
            'hidden_size': self.hidden_size,
            'is_built': self.is_built
        }
        
        torch.save(data, path)
        logger.info(f"{path} 저장완료")
        
    def load(self, path: str):
        """디스크에서 데이터스토어 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f" {path}")
            
        data = torch.load(path, map_location='cpu')
        
        self.keys = data['keys']
        self.values = data['values']
        self.contexts = data.get('contexts', [])
        self.size = data['size']
        self.hidden_size = data['hidden_size']
        self.is_built = data['is_built']
        
        logger.info(f"{path} 로드. 크기: {self.size}")
        
    def get_subset(self, indices: torch.Tensor):
        """데이터스토어의 부분집합 반환"""
        return self.keys[indices], self.values[indices]
        
    def __len__(self):
        return self.size
        
    def __repr__(self):
        return f"DataStore(size={self.size}, hidden_size={self.hidden_size}, built={self.is_built})" 