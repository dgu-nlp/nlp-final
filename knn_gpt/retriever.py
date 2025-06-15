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
import time

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
        
        try:
            # NumPy 배열로 변환 (학습 및 인덱스 구축에 사용)
            keys_np = keys.cpu().numpy().astype('float32')
            
            # 데이터 크기에 따라 적절한 인덱스 선택
            data_size = len(keys)
            
            # GPU 사용 가능 여부 확인
            use_gpu = self.device.type == 'cuda' and torch.cuda.is_available()
            
            if use_gpu:
                # GPU 정보 출력
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                logger.info(f"사용 가능한 GPU 수: {gpu_count}, 현재 디바이스: {current_device}")
                logger.info(f"현재 GPU: {torch.cuda.get_device_name(current_device)}")
                
                # GPU 리소스 설정
                res = faiss.StandardGpuResources()
            
            # 인덱스 유형 결정 및 생성
            if data_size > 1000000:
                # 대용량 데이터에는 IVF 인덱스 사용
                nlist = min(4096, data_size // 100)  # 클러스터 수
                logger.info(f"대용량 데이터 감지: IVF 인덱스 사용 (nlist={nlist})")
                
                # CPU에서 인덱스 생성
                quantizer = faiss.IndexFlatL2(d)
                cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                cpu_index.nprobe = min(256, nlist // 4)  # 검색 시 탐색할 클러스터 수
                
                # 학습 데이터 샘플링 (메모리 절약)
                logger.info("IVF 인덱스 학습 중...")
                train_start = time.time()
                max_train_points = min(1000000, len(keys_np))
                if len(keys_np) > max_train_points:
                    indices = np.random.choice(len(keys_np), max_train_points, replace=False)
                    train_data = keys_np[indices]
                else:
                    train_data = keys_np
                
                # CPU에서 학습 수행
                cpu_index.train(train_data)
                logger.info(f"IVF 인덱스 학습 완료. 소요 시간: {time.time() - train_start:.1f}초")
                
                # GPU로 이동 (학습된 상태 유지)
                if use_gpu:
                    logger.info(f"학습된 IVF 인덱스를 GPU {current_device}로 이동합니다.")
                    self.faiss_index = faiss.index_cpu_to_gpu(res, current_device, cpu_index)
                    self.gpu_resources = res
                    self.using_gpu = True
                else:
                    self.faiss_index = cpu_index
                    self.using_gpu = False
            else:
                # 소규모 데이터에는 Flat 인덱스 사용
                logger.info("소규모 데이터: Flat 인덱스 사용")
                if use_gpu:
                    # GPU Flat 인덱스 생성
                    config = faiss.GpuIndexFlatConfig()
                    config.device = current_device
                    config.useFloat16 = False  # 정확도를 위해 float32 사용
                    
                    logger.info(f"GPU {current_device}에 Flat 인덱스를 생성합니다.")
                    self.faiss_index = faiss.GpuIndexFlatL2(res, d, config)
                    self.gpu_resources = res
                    self.using_gpu = True
                else:
                    # CPU Flat 인덱스 생성
                    self.faiss_index = faiss.IndexFlatL2(d)
                    self.using_gpu = False
            
            # 인덱스에 키 추가
            logger.info("데이터를 FAISS 인덱스에 추가하는 중...")
            
            # 메모리 효율성을 위해 배치로 추가
            batch_size = 10000  # 배치 크기 설정
            num_batches = (len(keys_np) + batch_size - 1) // batch_size  # 올림 나눗셈
            
            logger.info(f"총 {num_batches}개 배치로 {len(keys_np)}개 벡터를 추가합니다. (배치 크기: {batch_size})")
            
            # 시작 시간 기록
            start_time = time.time()
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(keys_np))
                batch = keys_np[start_idx:end_idx]
                
                # 배치 추가
                self.faiss_index.add(batch)
                
                # 진행 상황 보고 (10% 단위)
                if (i + 1) % max(1, num_batches // 10) == 0 or i == num_batches - 1:
                    progress = (i + 1) / num_batches * 100
                    elapsed = time.time() - start_time
                    logger.info(f"인덱스 구축 진행률: {progress:.1f}% ({i+1}/{num_batches} 배치, 경과 시간: {elapsed:.1f}초)")
                
            # 총 소요 시간
            total_time = time.time() - start_time
            logger.info(f"FAISS 인덱스 구축 완료. 총 {len(keys)}개 벡터, 소요 시간: {total_time:.1f}초")
            
            # 인덱스 검증
            logger.info("인덱스 검증 중...")
            
            # 인덱스 크기 확인
            if hasattr(self.faiss_index, 'ntotal'):
                index_size = self.faiss_index.ntotal
                logger.info(f"인덱스에 추가된 총 벡터 수: {index_size}")
                
                if index_size != len(keys_np):
                    logger.warning(f"인덱스 크기 불일치: 예상 {len(keys_np)}, 실제 {index_size}")
            
            # 테스트 쿼리로 검증
            test_indices = np.random.choice(len(keys_np), min(5, len(keys_np)), replace=False)
            for idx in test_indices:
                test_query = keys_np[idx:idx+1]
                distances, indices = self.faiss_index.search(test_query, 1)
                
                # 자기 자신이 가장 가까운 이웃이어야 함
                if indices[0][0] == idx:
                    logger.info(f"검증 성공: 인덱스 {idx}의 가장 가까운 이웃은 자기 자신입니다.")
                else:
                    logger.warning(f"검증 실패: 인덱스 {idx}의 가장 가까운 이웃은 {indices[0][0]}입니다.")
            
        except Exception as e:
            logger.error(f"FAISS 인덱스 구축 중 오류 발생: {e}")
            self.faiss_index = None
            self.gpu_resources = None
            self.using_gpu = False
            logger.warning("FAISS 인덱스 구축에 실패했습니다. 직접 거리 계산을 사용합니다.")
        
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
        
        # 차원 확인 및 처리
        original_shape = query_vectors.shape
        if len(original_shape) == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = original_shape
            # 마지막 시퀀스 위치의 벡터만 사용
            query_vectors = query_vectors[:, -1, :]
            logger.info(f"3D 텐서 감지: 마지막 시퀀스 위치의 벡터만 사용합니다. 새 형태: {query_vectors.shape}")
        elif len(original_shape) == 2:  # [batch_size, hidden_size] 또는 [seq_len, hidden_size]
            # 이미 올바른 형태일 수 있음
            if original_shape[0] == 1 and original_shape[1] == self.datastore.hidden_size:
                # 단일 샘플, 올바른 형태
                pass
            elif original_shape[1] != self.datastore.hidden_size:
                # 첫 번째 차원이 배치가 아닌 시퀀스 길이인 경우
                # 마지막 시퀀스 위치의 벡터만 사용
                query_vectors = query_vectors[-1:, :]
                logger.info(f"시퀀스 텐서 감지: 마지막 위치의 벡터만 사용합니다. 새 형태: {query_vectors.shape}")
        elif len(original_shape) == 1:  # [hidden_size]
            # 단일 벡터를 [1, hidden_size] 형태로 변환
            query_vectors = query_vectors.unsqueeze(0)
            logger.info(f"1D 텐서를 2D로 변환: {query_vectors.shape}")
            
        # 최종 형태 확인
        if query_vectors.shape[1] != self.datastore.hidden_size:
            logger.warning(f"쿼리 벡터 차원({query_vectors.shape[1]})이 데이터스토어 hidden_size({self.datastore.hidden_size})와 일치하지 않습니다.")
            # 차원이 일치하지 않으면 전치 시도
            if query_vectors.shape[0] == self.datastore.hidden_size:
                query_vectors = query_vectors.transpose(0, 1)
                logger.info(f"쿼리 벡터를 전치했습니다. 새 형태: {query_vectors.shape}")
        
        # 배치 크기 확인
        batch_size = query_vectors.shape[0]
        
        # FAISS 검색 또는 직접 계산
        if self.faiss_index is not None and self.use_faiss:
            try:
                # 큰 배치 크기를 처리하기 위해 분할 처리
                max_batch_size = 32  # FAISS 검색에 적합한 최대 배치 크기
                
                if batch_size > max_batch_size:
                    logger.info(f"큰 배치 크기 감지: {batch_size}. {max_batch_size}씩 분할 처리합니다.")
                    
                    all_distances = []
                    all_indices = []
                    
                    # 배치 분할 처리
                    for i in range(0, batch_size, max_batch_size):
                        end_idx = min(i + max_batch_size, batch_size)
                        batch_queries = query_vectors[i:end_idx]
                        
                        # 현재 배치에 대한 FAISS 검색
                        batch_queries_np = batch_queries.cpu().numpy().astype('float32')
                        batch_distances, batch_indices = self.faiss_index.search(batch_queries_np, k)
                        
                        # 결과 수집
                        all_distances.append(torch.from_numpy(batch_distances))
                        all_indices.append(torch.from_numpy(batch_indices))
                    
                    # 결과 결합
                    distances = torch.cat(all_distances, dim=0).to(self.device)
                    indices = torch.cat(all_indices, dim=0).to(self.device)
                    
                else:
                    # 작은 배치는 한 번에 처리
                    query_np = query_vectors.cpu().numpy().astype('float32')
                    distances, indices = self.faiss_index.search(query_np, k)
                    
                    # 텐서로 변환하고 디바이스로 이동
                    distances = torch.from_numpy(distances).to(self.device)
                    indices = torch.from_numpy(indices).to(self.device)
                
                # 결과 검증
                if distances.shape[0] != query_vectors.shape[0]:
                    logger.warning(f"FAISS 검색 결과 형태 불일치: 쿼리 배치 크기 {query_vectors.shape[0]}, 결과 크기 {distances.shape[0]}")
                    # 직접 계산으로 대체
                    return self._direct_search(query_vectors, k)
                
                # 검색 결과 검증
                valid_indices = (indices >= 0) & (indices < len(self.datastore))
                if not valid_indices.all():
                    invalid_count = (~valid_indices).sum().item()
                    logger.warning(f"유효하지 않은 인덱스 발견: {invalid_count}개. 직접 계산으로 대체합니다.")
                    return self._direct_search(query_vectors, k)
                
                return distances, indices
                
            except Exception as e:
                logger.warning(f"FAISS 검색 실패: {e}. 직접 거리 계산으로 대체합니다.")
                # FAISS 검색 실패 시 직접 계산으로 대체
                return self._direct_search(query_vectors, k)
        else:
            # 직접 거리 계산
            logger.info("FAISS 인덱스가 없거나 비활성화되어 있습니다. 직접 거리 계산을 사용합니다.")
            return self._direct_search(query_vectors, k)
        
    def _direct_search(self, query_vectors: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """직접 거리 계산을 통한 검색"""
        try:
            # 데이터스토어 키를 동일한 디바이스로 이동
            keys = self.datastore.keys.to(query_vectors.device)
            
            # 배치 처리를 위한 차원 확인
            if query_vectors.dim() == 1:
                query_vectors = query_vectors.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
                
            batch_size = query_vectors.size(0)
            distances_list = []
            indices_list = []
            
            # 차원 확인 및 로깅
            logger.info(f"쿼리 벡터 크기: {query_vectors.shape}, 키 크기: {keys.shape}")
            
            # 메모리 효율성을 위해 배치 단위로 처리
            for i in range(batch_size):
                # 현재 쿼리 벡터
                query = query_vectors[i].unsqueeze(0)  # [1, hidden_size]
                
                # 차원 확인
                if query.shape[1] != keys.shape[1]:
                    logger.warning(f"쿼리 차원({query.shape[1]})과 키 차원({keys.shape[1]})이 일치하지 않습니다.")
                    # 차원이 일치하지 않으면 패딩 또는 잘라내기 시도
                    if query.shape[1] < keys.shape[1]:
                        # 패딩
                        padding = torch.zeros(1, keys.shape[1] - query.shape[1], device=query.device)
                        query = torch.cat([query, padding], dim=1)
                        logger.info(f"쿼리 벡터를 패딩했습니다. 새 형태: {query.shape}")
                    else:
                        # 잘라내기
                        query = query[:, :keys.shape[1]]
                        logger.info(f"쿼리 벡터를 잘라냈습니다. 새 형태: {query.shape}")
                
                # 모든 키와의 유클리드 거리 계산 (직접 계산)
                # L2 거리: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
                query_norm = torch.sum(query ** 2, dim=1).view(-1, 1)  # [1, 1]
                keys_norm = torch.sum(keys ** 2, dim=1).view(1, -1)    # [1, num_keys]
                distances = query_norm + keys_norm
                distances = distances - 2 * torch.mm(query, keys.t())  # [1, num_keys]
                distances = torch.sqrt(torch.clamp(distances, min=1e-8))  # 수치 안정성을 위한 클램핑
                distances = distances.squeeze(0)  # [num_keys]
                
                # 상위 k개 가져오기
                topk_distances, topk_indices = torch.topk(distances, k=min(k, len(distances)), largest=False)
                
                distances_list.append(topk_distances)
                indices_list.append(topk_indices)
                
            # 결과 결합
            distances = torch.stack(distances_list)
            indices = torch.stack(indices_list)
            
            return distances, indices
            
        except Exception as e:
            logger.error(f"직접 거리 계산 중 오류 발생: {e}")
            # 최후의 방법: 무작위 이웃 반환
            batch_size = 1 if query_vectors.dim() == 1 else query_vectors.size(0)
            random_distances = torch.ones(batch_size, k, device=query_vectors.device)
            random_indices = torch.randint(0, len(self.datastore), (batch_size, k), device=query_vectors.device)
            return random_distances, random_indices
        
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