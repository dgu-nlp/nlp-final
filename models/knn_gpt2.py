"""
k-NN 증강 GPT-2 기본 모델

Khandelwal et al. (2019)의 k-NN Language Models를 GPT-2에 적용한 구현입니다.
기존 GPT-2 모델에 k-NN 검색 메커니즘을 추가하여 성능을 향상
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
import logging

from .gpt2 import GPT2Model
from knn_gpt import DataStore, KNNRetriever, InterpolationModule, QualityFilter

logger = logging.getLogger(__name__)


class KNNAugmentedGPT2(nn.Module):
    """k-NN 검색으로 증강된 GPT-2 모델"""
    
    def __init__(self, 
                 base_model: GPT2Model,
                 datastore: Optional[DataStore] = None,
                 k: int = 8,
                 lambda_knn: float = 0.25,
                 knn_temperature: float = 10.0,
                 use_quality_filter: bool = True,
                 use_adaptive_interpolation: bool = False,
                 vocab_size: int = 50257,
                 device: Optional[torch.device] = None):
        """
        Args:
            base_model: 기본 GPT-2 모델
            datastore: k-NN 검색을 위한 데이터스토어
            k: 검색할 이웃 수
            lambda_knn: k-NN 예측에 대한 가중치
            knn_temperature: k-NN 검색에서 사용할 온도
            use_quality_filter: 품질 필터링 사용 여부
            use_adaptive_interpolation: 적응적 interpolation 사용 여부
            vocab_size: 어휘 크기
            device: 디바이스
        """
        super().__init__()
        
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.k = k
        self.use_quality_filter = use_quality_filter
        self.use_adaptive_interpolation = use_adaptive_interpolation
        
        # k-NN 관련 컴포넌트들
        self.datastore = datastore
        self.knn_retriever = None
        self.interpolation_module = InterpolationModule(
            lambda_knn=lambda_knn, 
            adaptive_lambda=use_adaptive_interpolation
        )
        
        if use_quality_filter:
            self.quality_filter = QualityFilter()
        else:
            self.quality_filter = None
            
        # 데이터스토어가 있으면 retriever 초기화
        if datastore is not None:
            self.initialize_retriever(knn_temperature)
            
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_retriever(self, temperature: float = 10.0):
        """k-NN 검색기 초기화"""
        if self.datastore is None or not self.datastore.is_built:
            raise ValueError("데이터스토어가 구축되지 않았습니다.")
            
        self.knn_retriever = KNNRetriever(
            datastore=self.datastore,
            k=self.k,
            temperature=temperature,
            use_faiss=True,
            device=self.device
        )
        logger.info("k-NN 검색기가 초기화되었습니다.")
        
    def set_datastore(self, datastore):
        """데이터스토어 설정"""
        self.datastore = datastore
        
        # 디바이스 확인 및 조정
        if hasattr(self, 'device') and hasattr(datastore, 'device') and datastore.device != self.device:
            logger.warning(f"데이터스토어 디바이스({datastore.device})와 모델 디바이스({self.device})가 다릅니다.")
            logger.info(f"데이터스토어를 {self.device}로 이동합니다.")
            datastore.device = self.device
            if hasattr(datastore, 'keys') and isinstance(datastore.keys, torch.Tensor):
                datastore.keys = datastore.keys.to(self.device)
            if hasattr(datastore, 'values') and isinstance(datastore.values, torch.Tensor):
                datastore.values = datastore.values.to(self.device)
        
        # k-NN 검색기 초기화
        self.initialize_retriever()
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                return_knn_info: bool = False) -> Dict[str, torch.Tensor]:
        """
        k-NN 증강 forward pass
        
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            return_knn_info: k-NN 관련 정보 반환 여부
            
        Returns:
            출력 딕셔너리 (logits, base_logits, knn_logits 등)
        """
        # 기본 모델 forward pass
        base_outputs = self.base_model(input_ids, attention_mask)
        base_logits = self.base_model.hidden_state_to_token(base_outputs['last_token'])
        
        # k-NN 검색이 활성화되지 않았으면 기본 출력만 반환
        if self.knn_retriever is None:
            return {
                'logits': base_logits,
                'base_logits': base_logits,
                'last_hidden_state': base_outputs['last_hidden_state'],
                'last_token': base_outputs['last_token']
            }
            
        # k-NN 검색 수행
        query_hidden = base_outputs['last_token']  # [batch_size, hidden_size]
        
        # 배치 처리
        if query_hidden.dim() == 1:
            query_hidden = query_hidden.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = query_hidden.size(0)
        knn_logits = torch.zeros(batch_size, self.vocab_size, device=query_hidden.device)
        knn_info = []
        
        for i in range(batch_size):
            # 개별 쿼리에 대해 k-NN 검색
            distances, indices = self.knn_retriever.search(query_hidden[i])
            
            # 이웃들의 정보 가져오기
            neighbor_keys = self.datastore.keys[indices]
            neighbor_tokens = self.datastore.values[indices]
            
            # 품질 필터링 적용 (선택적)
            if self.quality_filter is not None:
                neighbor_keys, neighbor_tokens, distances, filter_info = self.quality_filter.filter_neighbors(
                    query_hidden[i], neighbor_keys, neighbor_tokens, distances
                )
                if return_knn_info:
                    knn_info.append({
                        'distances': distances.tolist(),
                        'tokens': neighbor_tokens.tolist(),
                        'filter_info': filter_info
                    })
            else:
                if return_knn_info:
                    knn_info.append({
                        'distances': distances.tolist(),
                        'tokens': neighbor_tokens.tolist()
                    })
            
            # k-NN 로짓 계산
            if len(neighbor_tokens) > 0:
                # 거리를 유사도로 변환
                similarities = -distances / self.knn_retriever.temperature
                weights = F.softmax(similarities, dim=0)
                
                # 토큰 분포 계산
                token_logits = torch.zeros(self.vocab_size, device=query_hidden.device)
                for j, token_id in enumerate(neighbor_tokens):
                    token_logits[token_id] += weights[j]
                    
                knn_logits[i] = torch.log(token_logits + 1e-10)
            else:
                # 이웃이 없으면 균등 분포
                knn_logits[i] = torch.zeros(self.vocab_size, device=query_hidden.device)
        
        # Interpolation 수행
        if self.use_adaptive_interpolation:
            # 적응적 interpolation (엔트로피 기반)
            interpolated_logits, adaptive_lambdas = self.interpolation_module.entropy_based_interpolation(
                base_logits, knn_logits
            )
        else:
            # 고정 가중치 interpolation
            interpolated_logits = self.interpolation_module.interpolate_logits(
                base_logits, knn_logits
            )
            adaptive_lambdas = None
            
        if squeeze_output:
            interpolated_logits = interpolated_logits.squeeze(0)
            base_logits = base_logits.squeeze(0)
            knn_logits = knn_logits.squeeze(0)
            
        result = {
            'logits': interpolated_logits,
            'base_logits': base_logits,
            'knn_logits': knn_logits,
            'last_hidden_state': base_outputs['last_hidden_state'],
            'last_token': base_outputs['last_token']
        }
        
        if return_knn_info:
            result['knn_info'] = knn_info
            
        if adaptive_lambdas is not None:
            result['adaptive_lambdas'] = adaptive_lambdas
            
        return result
        
    def generate(self, 
                 input_ids: torch.Tensor, 
                 attention_mask: torch.Tensor,
                 max_length: int = 50,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 do_sample: bool = True) -> Tuple[torch.Tensor, str]:
        """
        k-NN 증강 텍스트 생성
        
        Args:
            input_ids: 시작 토큰들
            attention_mask: 어텐션 마스크
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            do_sample: 샘플링 여부
            
        Returns:
            생성된 토큰들과 디코딩된 텍스트
        """
        self.eval()
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated_ids, current_attention_mask)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if hasattr(self.base_model, 'tokenizer'):
                    if next_token.item() == self.base_model.tokenizer.eos_token_id:
                        break
                
                # Append generated token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.size(0), 1), 
                              device=current_attention_mask.device, 
                              dtype=current_attention_mask.dtype)
                ], dim=-1)
        
        # Decode generated text
        if hasattr(self.base_model, 'tokenizer'):
            generated_text = self.base_model.tokenizer.decode(
                generated_ids[0].cpu().tolist(), 
                skip_special_tokens=True
            )
        else:
            generated_text = ""
            
        return generated_ids, generated_text
        
    def get_knn_statistics(self, 
                          input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor) -> Dict:
        """k-NN 관련 통계 정보 반환 (분석용)"""
        if self.knn_retriever is None:
            return {'error': 'k-NN retriever not initialized'}
            
        outputs = self.forward(input_ids, attention_mask, return_knn_info=True)
        
        # Interpolation 정보
        interp_info = self.interpolation_module.get_interpolation_info(
            outputs['base_logits'], outputs['knn_logits']
        )
        
        return {
            'knn_info': outputs.get('knn_info', []),
            'interpolation_info': interp_info,
            'adaptive_lambdas': outputs.get('adaptive_lambdas', None),
            'datastore_size': len(self.datastore) if self.datastore else 0,
            'k': self.k
        }
        
    def update_knn_parameters(self, 
                             k: Optional[int] = None,
                             lambda_knn: Optional[float] = None,
                             knn_temperature: Optional[float] = None):
        """k-NN 파라미터 업데이트"""
        if k is not None:
            self.k = k
            if self.knn_retriever is not None:
                self.knn_retriever.update_parameters(k=k)
                
        if lambda_knn is not None:
            self.interpolation_module.update_parameters(lambda_knn=lambda_knn)
            
        if knn_temperature is not None and self.knn_retriever is not None:
            self.knn_retriever.update_parameters(temperature=knn_temperature)
            
    def enable_knn(self):
        """k-NN 검색 활성화"""
        if self.datastore is None:
            raise ValueError("데이터스토어가 설정되지 않았습니다.")
        if self.knn_retriever is None:
            self.initialize_retriever()
            
    def disable_knn(self):
        """k-NN 검색 비활성화 (기본 모델만 사용)"""
        self.knn_retriever = None
        
    def __repr__(self):
        knn_status = "enabled" if self.knn_retriever is not None else "disabled"
        return (f"KNNAugmentedGPT2(base_model={type(self.base_model).__name__}, "
                f"k={self.k}, knn={knn_status}, "
                f"datastore_size={len(self.datastore) if self.datastore else 0})") 