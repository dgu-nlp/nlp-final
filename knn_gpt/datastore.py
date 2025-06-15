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
        
    def build_from_model_and_data(self, model, dataloader, save_path: str = None, chunk_size: int = 1000000):
        """
        모델과 데이터로더를 사용하여 데이터스토어 구축
        
        Args:
            model: GPT-2 모델
            dataloader: 훈련 데이터 로더
            save_path: 데이터스토어 저장 경로
            chunk_size: 메모리에 한 번에 저장할 최대 키-값 쌍 수 (메모리 관리용)
        """
        logger.info("데이터스토어 구축 시작")
        
        model.eval()
        self.keys = []
        self.values = []
        self.contexts = []
        
        # 청크 관리
        current_chunk = 0
        total_size = 0
        
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
                
                # 메모리 관리: 일정 크기마다 중간 저장 및 메모리 정리
                if len(self.keys) >= chunk_size:
                    # 현재까지의 데이터를 텐서로 변환
                    keys_tensor = torch.stack(self.keys)
                    values_tensor = torch.tensor(self.values, dtype=torch.long)
                    
                    # 중간 저장
                    if save_path:
                        chunk_path = f"{os.path.splitext(save_path)[0]}_chunk{current_chunk}.pt"
                        chunk_data = {
                            'keys': keys_tensor,
                            'values': values_tensor,
                            'contexts': self.contexts,
                            'size': len(keys_tensor),
                            'hidden_size': self.hidden_size
                        }
                        self._save_chunk(chunk_data, chunk_path)
                        logger.info(f"중간 데이터 저장 완료: {chunk_path} (크기: {len(keys_tensor)})")
                    
                    # 누적 크기 업데이트
                    total_size += len(self.keys)
                    
                    # 메모리 정리
                    del self.keys, self.values
                    if 'keys_tensor' in locals():
                        del keys_tensor, values_tensor
                    import gc
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # 새 청크 준비
                    self.keys = []
                    self.values = []
                    self.contexts = []
                    current_chunk += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"처리된 배치: {batch_idx}, 현재 데이터스토어 크기: {len(self.keys) + total_size}")
        
        # 마지막 데이터 처리
        if self.keys:
            # 텐서로 변환
            keys_tensor = torch.stack(self.keys)
            values_tensor = torch.tensor(self.values, dtype=torch.long)
            
            # 마지막 청크 저장
            if current_chunk > 0 and save_path:
                chunk_path = f"{os.path.splitext(save_path)[0]}_chunk{current_chunk}.pt"
                chunk_data = {
                    'keys': keys_tensor,
                    'values': values_tensor,
                    'contexts': self.contexts,
                    'size': len(keys_tensor),
                    'hidden_size': self.hidden_size
                }
                self._save_chunk(chunk_data, chunk_path)
                logger.info(f"마지막 청크 저장 완료: {chunk_path} (크기: {len(keys_tensor)})")
            
            # 최종 크기 업데이트
            total_size += len(self.keys)
            
            # 최종 데이터스토어 설정
            self.keys = keys_tensor
            self.values = values_tensor
        
        self.size = total_size
        self.is_built = True
        
        logger.info(f"데이터스토어 구축 완료. 총 크기: {self.size}")
        
        # 최종 저장
        if save_path:
            if current_chunk > 0:
                # 여러 청크로 나뉘어 저장된 경우, 메타데이터만 저장
                metadata = {
                    'total_size': self.size,
                    'hidden_size': self.hidden_size,
                    'is_built': self.is_built,
                    'chunk_count': current_chunk + 1,
                    'base_path': os.path.splitext(save_path)[0]
                }
                try:
                    torch.save(metadata, save_path)
                    logger.info(f"데이터스토어 메타데이터 저장 완료: {save_path}")
                except Exception as e:
                    logger.error(f"메타데이터 저장 실패: {e}")
            else:
                # 단일 파일로 저장
                self.save(save_path)

    def _save_chunk(self, data, path):
        """청크 데이터 저장 헬퍼 함수"""
        try:
            # 기본 저장 방식 시도
            torch.save(data, path)
        except Exception as e:
            logger.warning(f"기본 저장 방식 실패: {e}")
            
            # 대안적인 저장 방식 시도
            import pickle
            try:
                torch.save(data, path, pickle_module=pickle)
                logger.info(f"{path} 저장완료 (pickle 모듈 사용)")
            except Exception as e2:
                logger.error(f"모든 저장 방식 실패: {e2}")
                
                # 최후의 방법: NumPy 배열로 변환하여 저장
                try:
                    np_data = {
                        'keys': data['keys'].cpu().numpy(),
                        'values': data['values'].cpu().numpy(),
                        'contexts': data['contexts'],
                        'size': data['size'],
                        'hidden_size': data['hidden_size']
                    }
                    np.save(path + '.npy', np_data)
                    logger.info(f"{path}.npy 저장완료 (NumPy 배열 사용)")
                except Exception as e3:
                    logger.critical(f"모든 저장 방식 실패. 데이터스토어를 저장할 수 없습니다: {e3}")
                    raise
        
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