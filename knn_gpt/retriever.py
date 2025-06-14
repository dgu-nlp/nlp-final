"""
KNNRetriever: k-NN 검색 엔진

주어진 쿼리 hidden state에 대해 데이터스토어에서 가장 가까운 k개의 이웃을 찾고,
이를 바탕으로 다음 토큰 분포를 생성
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import faiss  # 빠른 유사도 검색을 위한 라이브러리

logger = logging.getLogger(__name__)


class KNNRetriever:
    """k-NN 검색을 수행하는 클래스"""
    
    def __init__(self, 
                 datastore,
                 k: int = 8,
                 temperature: float = 10.0,
                 use_faiss: bool = True,
                 faiss_index_type: str = 'l2'):
        """
        Args:
            datastore: DataStore 객체
            k: 검색할 이웃 수
            temperature: 거리 기반 가중치 계산을 위한 온도 파라미터
            use_faiss: FAISS 사용 여부 (대용량 데이터에서 빠른 검색)
            faiss_index_type: FAISS 인덱스 유형 ('l2' 또는 'inner_product')
        """
        self.datastore = datastore
        self.k = k
        self.temperature = temperature
        self.use_faiss = use_faiss
        self.faiss_index_type = faiss_index_type
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        self.gpu_resources = None
        self.using_gpu = False
        
        if use_faiss and datastore.is_built:
            self._build_faiss_index()
            
    def _build_faiss_index(self):
        """FAISS 인덱스 구축 (GPU 우선, CPU fallback)"""
        if not self.datastore.is_built:
            raise ValueError("데이터스토어가 구축되지 않았습니다.")
            
        logger.info("FAISS 인덱스를 구축하고 있습니다...")
        
        # 키들을 numpy 배열로 변환
        keys_np = self.datastore.keys.numpy().astype('float32')
        
        # CPU 인덱스 생성
        if self.faiss_index_type == 'l2':
            cpu_index = faiss.IndexFlatL2(self.datastore.hidden_size)
        elif self.faiss_index_type == 'inner_product':
            cpu_index = faiss.IndexFlatIP(self.datastore.hidden_size)
            # inner product의 경우 정규화 필요
            faiss.normalize_L2(keys_np)
        else:
            raise ValueError(f"지원되지 않는 FAISS 인덱스 유형: {self.faiss_index_type}")
            
        # GPU 사용 시도 (GPU가 있으면 자동으로 사용)
        if torch.cuda.is_available():
            try:
                # GPU 리소스 설정
                self.gpu_resources = faiss.StandardGpuResources()
                
                # GPU 인덱스 생성
                self.faiss_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, 
                    0,  # GPU 0 사용
                    cpu_index
                )
                self.using_gpu = True
                logger.info("GPU 인덱스 생성 완료 (GPU 0)")
                
            except Exception as e:
                logger.warning(f"GPU 인덱스 생성 실패, CPU로 fallback: {e}")
                self.faiss_index = cpu_index
                self.using_gpu = False
        else:
            # GPU가 없으면 CPU 인덱스 사용
            self.faiss_index = cpu_index
            self.using_gpu = False
            logger.info("CPU 인덱스 생성 완료 (GPU 사용 불가)")
            
        # 인덱스에 키 추가
        self.faiss_index.add(keys_np)
        logger.info(f"FAISS 인덱스 구축 완료. 총 {self.faiss_index.ntotal}개 벡터")
        
    def search(self, query_hidden: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k-NN 검색 수행
        
        Args:
            query_hidden: 쿼리 hidden state [hidden_size] 또는 [batch_size, hidden_size]
            k: 검색할 이웃 수 (None이면 self.k 사용)
            
        Returns:
            distances: k개 이웃까지의 거리 [k] 또는 [batch_size, k]
            indices: k개 이웃의 인덱스 [k] 또는 [batch_size, k]
        """
        if k is None:
            k = self.k
            
        if not self.datastore.is_built:
            raise ValueError("데이터스토어가 구축되지 않았습니다.")
            
        # 배치 차원 처리
        if query_hidden.dim() == 1:
            query_hidden = query_hidden.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = query_hidden.size(0)
        
        if self.use_faiss and self.faiss_index is not None:
            # FAISS를 사용한 빠른 검색
            query_np = query_hidden.detach().cpu().numpy().astype('float32')
            
            if self.faiss_index_type == 'inner_product':
                faiss.normalize_L2(query_np)
                
            distances, indices = self.faiss_index.search(query_np, k)
            distances = torch.from_numpy(distances)
            indices = torch.from_numpy(indices)
        else:
            # PyTorch를 사용한 직접 검색
            distances = torch.cdist(query_hidden, self.datastore.keys)  # [batch_size, datastore_size]
            distances, indices = torch.topk(distances, k, largest=False, dim=1)
            
        if squeeze_output:
            distances = distances.squeeze(0)
            indices = indices.squeeze(0)
            
        return distances, indices
        
    def get_knn_logits(self, 
                      query_hidden: torch.Tensor, 
                      vocab_size: int,
                      k: Optional[int] = None) -> torch.Tensor:
        """
        k-NN 검색 결과를 바탕으로 다음 토큰에 대한 로짓 계산
        
        Args:
            query_hidden: 쿼리 hidden state
            vocab_size: 어휘 크기
            k: 검색할 이웃 수
            
        Returns:
            knn_logits: k-NN 기반 로짓 [vocab_size] 또는 [batch_size, vocab_size]
        """
        # k-NN 검색 수행
        distances, indices = self.search(query_hidden, k)
        
        # 배치 차원 처리
        if distances.dim() == 1:
            distances = distances.unsqueeze(0)
            indices = indices.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = distances.size(0)
        k = distances.size(1)
        
        # 거리를 유사도로 변환 (작은 거리 = 높은 유사도)
        similarities = -distances / self.temperature
        weights = F.softmax(similarities, dim=1)  # [batch_size, k]
        
        # 각 배치에 대해 토큰 분포 계산
        knn_logits = torch.zeros(batch_size, vocab_size)
        
        for b in range(batch_size):
            # 해당 이웃들의 다음 토큰들
            neighbor_tokens = self.datastore.values[indices[b]]  # [k]
            
            # 각 토큰에 대해 가중치 합계 계산
            for i, token_id in enumerate(neighbor_tokens):
                knn_logits[b, token_id] += weights[b, i]
                
        # 로그 확률로 변환
        knn_logits = torch.log(knn_logits + 1e-10)  # 수치 안정성을 위한 작은 값 추가
        
        if squeeze_output:
            knn_logits = knn_logits.squeeze(0)
            
        return knn_logits
        
    def get_knn_prob_distribution(self, 
                                 query_hidden: torch.Tensor, 
                                 vocab_size: int,
                                 k: Optional[int] = None) -> torch.Tensor:
        """
        k-NN 검색 결과를 바탕으로 다음 토큰에 대한 확률 분포 계산
        
        Args:
            query_hidden: 쿼리 hidden state
            vocab_size: 어휘 크기
            k: 검색할 이웃 수
            
        Returns:
            knn_probs: k-NN 기반 확률 분포 [vocab_size] 또는 [batch_size, vocab_size]
        """
        knn_logits = self.get_knn_logits(query_hidden, vocab_size, k)
        return F.softmax(knn_logits, dim=-1)
        
    def get_neighbors_info(self, query_hidden: torch.Tensor, k: Optional[int] = None) -> Dict:
        """
        k-NN 검색 결과에 대한 상세 정보 반환 (디버깅 및 분석용)
        
        Args:
            query_hidden: 쿼리 hidden state
            k: 검색할 이웃 수
            
        Returns:
            정보 딕셔너리 (distances, indices, tokens, contexts 등)
        """
        distances, indices = self.search(query_hidden, k)
        
        # 단일 쿼리만 처리 (배치 처리 시 첫 번째만)
        if distances.dim() > 1:
            distances = distances[0]
            indices = indices[0]
            
        neighbor_tokens = self.datastore.values[indices]
        neighbor_contexts = []
        
        if self.datastore.contexts:
            for idx in indices:
                if idx < len(self.datastore.contexts):
                    neighbor_contexts.append(self.datastore.contexts[idx])
                else:
                    neighbor_contexts.append("")
                    
        return {
            'distances': distances.tolist(),
            'indices': indices.tolist(),
            'tokens': neighbor_tokens.tolist(),
            'contexts': neighbor_contexts
        }
        
    def update_parameters(self, k: Optional[int] = None, temperature: Optional[float] = None):
        """검색 파라미터 업데이트"""
        if k is not None:
            self.k = k
        if temperature is not None:
            self.temperature = temperature
            
    def cleanup_gpu_resources(self):
        """GPU 리소스 정리"""
        if self.gpu_resources is not None:
            del self.gpu_resources
            self.gpu_resources = None
            self.using_gpu = False
            logger.info("GPU 리소스가 정리되었습니다.")
            
    def get_device_info(self) -> Dict:
        """현재 사용 중인 디바이스 정보 반환"""
        info = {
            "using_gpu": self.using_gpu,
            "cuda_available": torch.cuda.is_available(),
            "faiss_available": self.use_faiss and self.faiss_index is not None
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["current_gpu"] = torch.cuda.current_device()
            
        return info
            
    def __repr__(self):
        device_info = "GPU" if self.using_gpu else "CPU"
        return f"KNNRetriever(k={self.k}, temperature={self.temperature}, use_faiss={self.use_faiss}, device={device_info})"
        
    def __del__(self):
        """소멸자에서 GPU 리소스 정리"""
        self.cleanup_gpu_resources() 