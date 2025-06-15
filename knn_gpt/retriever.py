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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNRetriever:
    """k-NN 검색을 수행하는 클래스"""
    
    def __init__(
        self,
        datastore,
        k: int = 8,
        temperature: float = 10.0,
        use_faiss: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            datastore: DataStore 객체
            k: 검색할 이웃 수
            temperature: 거리 기반 가중치 계산을 위한 온도 파라미터
            use_faiss: FAISS 사용 여부 (대용량 데이터에서 빠른 검색)
            device: 사용할 디바이스 (None이면 자동으로 선택)
        """
        self.datastore = datastore
        self.k = min(k, len(datastore))
        self.temperature = temperature
        self.use_faiss = use_faiss
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        self.gpu_resources = None
        self.using_gpu = False
        
        if use_faiss and datastore.is_built:
            self._build_faiss_index()
            
    def _build_faiss_index(self):
        """FAISS 인덱스 구축"""
        if not self.datastore.is_built:
            raise ValueError("데이터스토어가 구축되지 않았습니다.")
            
        logger.info("FAISS 인덱스를 구축하고 있습니다...")
        
        # 데이터스토어 키 가져오기
        keys = self.datastore.keys
        
        # 디바이스 확인
        if hasattr(keys, 'device'):
            keys_device = keys.device
            if keys_device != self.device:
                logger.warning(f"키 디바이스({keys_device})와 검색기 디바이스({self.device})가 다릅니다.")
                logger.info(f"키를 {self.device}로 이동합니다.")
                keys = keys.to(self.device)
        
        # 차원 확인
        d = keys.shape[1]
        
        # GPU 사용 가능 여부 확인
        use_gpu = self.device.type == 'cuda' and torch.cuda.is_available()
        
        if use_gpu:
            # GPU 인덱스 생성
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = self.device.index if hasattr(self.device, 'index') else 0
            self.faiss_index = faiss.GpuIndexFlatL2(res, d, config)
            self.using_gpu = True
            logger.info(f"GPU 인덱스 생성 완료 (GPU {config.device})")
        else:
            # CPU 인덱스 생성
            self.faiss_index = faiss.IndexFlatL2(d)
            self.using_gpu = False
            logger.info("CPU 인덱스 생성 완료")
        
        # 인덱스에 키 추가
        self.faiss_index.add(keys.cpu().numpy())
        logger.info(f"FAISS 인덱스 구축 완료. 총 {len(keys)}개 벡터")
        
        # 리소스 정리
        self.gpu_resources = res if use_gpu else None
        
    def search(self, query_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        쿼리 벡터에 가장 가까운 k개의 이웃 검색
        
        Args:
            query_vectors: 쿼리 벡터 [batch_size, hidden_size]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (거리, 인덱스) 튜플
                - 거리: [batch_size, k]
                - 인덱스: [batch_size, k]
        """
        # 디바이스 확인 및 조정
        if query_vectors.device != self.device:
            query_vectors = query_vectors.to(self.device)
        
        # 실제 사용할 k 결정
        k = min(self.k, len(self.datastore))
        
        if self.use_faiss:
            # FAISS 검색
            query_np = query_vectors.cpu().numpy().astype('float32')
            distances, indices = self.faiss_index.search(query_np, k)
            
            # 텐서로 변환하고 디바이스로 이동
            distances = torch.from_numpy(distances).to(self.device)
            indices = torch.from_numpy(indices).to(self.device)
        else:
            # 직접 거리 계산
            keys = self.datastore.keys.to(self.device)
            
            # L2 거리 계산
            # [batch_size, 1, hidden_size] - [1, num_keys, hidden_size] = [batch_size, num_keys, hidden_size]
            query_expanded = query_vectors.unsqueeze(1)
            keys_expanded = keys.unsqueeze(0)
            
            # L2 거리 계산
            distances = torch.norm(query_expanded - keys_expanded, dim=2, p=2)
            
            # 상위 k개 가져오기
            distances, indices = torch.topk(distances, k=k, dim=1, largest=False)
        
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
        distances, indices = self.search(query_hidden)
        
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
        distances, indices = self.search(query_hidden)
        
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
        """소멸자: GPU 리소스 정리"""
        if hasattr(self, 'gpu_resources') and self.gpu_resources is not None:
            logger.info("GPU 리소스가 정리되었습니다.")
            del self.gpu_resources
        
    def get_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        주어진 인덱스에 해당하는 값 가져오기
        
        Args:
            indices: 인덱스 [batch_size, k]
            
        Returns:
            torch.Tensor: 값 [batch_size, k]
        """
        # 데이터스토어 값 가져오기
        values = self.datastore.values
        
        # 디바이스 확인 및 조정
        if indices.device != values.device:
            indices = indices.to(values.device)
        
        # 인덱스에 해당하는 값 가져오기
        batch_size, k = indices.shape
        gathered_values = torch.gather(values.unsqueeze(0).expand(batch_size, -1), 1, indices)
        
        return gathered_values 