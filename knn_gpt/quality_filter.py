"""
QualityFilter: 검색된 컨텍스트 품질 평가 및 필터링

Yoran et al. (2024)의 "Making Retrieval-Augmented Language Models Robust to Irrelevant Context"
방법을 참고하여 검색된 이웃들의 관련성을 평가하고 품질이 낮은 컨텍스트를 필터링.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """검색된 컨텍스트의 품질을 평가하고 필터링하는 클래스"""
    
    def __init__(self, 
                 relevance_threshold: float = 0.7,
                 diversity_threshold: float = 0.3,
                 confidence_threshold: float = 0.8,
                 use_semantic_filtering: bool = True,
                 use_diversity_filtering: bool = True):
        """
        Args:
            relevance_threshold: 관련성 임계값 (이하면 필터링)
            diversity_threshold: 다양성 임계값 (이상이면 유지)
            confidence_threshold: 신뢰도 임계값
            use_semantic_filtering: 의미적 필터링 사용 여부
            use_diversity_filtering: 다양성 기반 필터링 사용 여부
        """
        self.relevance_threshold = relevance_threshold
        self.diversity_threshold = diversity_threshold
        self.confidence_threshold = confidence_threshold
        self.use_semantic_filtering = use_semantic_filtering
        self.use_diversity_filtering = use_diversity_filtering
        
    def compute_relevance_scores(self, 
                                query_hidden: torch.Tensor, 
                                neighbor_hiddens: torch.Tensor,
                                distances: torch.Tensor) -> torch.Tensor:
        """
        쿼리와 이웃들 간의 관련성 점수 계산
        
        Args:
            query_hidden: 쿼리 hidden state [hidden_size]
            neighbor_hiddens: 이웃들의 hidden states [k, hidden_size]
            distances: 이웃들과의 거리 [k]
            
        Returns:
            relevance_scores: 관련성 점수 [k]
        """
        # 코사인 유사도 계산
        cosine_similarities = F.cosine_similarity(
            query_hidden.unsqueeze(0), neighbor_hiddens, dim=1
        )
        
        # 거리 기반 유사도 (작은 거리 = 높은 유사도)
        distance_similarities = torch.exp(-distances / distances.mean())
        
        # 두 유사도를 결합
        relevance_scores = 0.7 * cosine_similarities + 0.3 * distance_similarities
        
        return relevance_scores
        
    def compute_diversity_scores(self, neighbor_hiddens: torch.Tensor) -> torch.Tensor:
        """
        이웃들 간의 다양성 점수 계산
        
        Args:
            neighbor_hiddens: 이웃들의 hidden states [k, hidden_size]
            
        Returns:
            diversity_scores: 각 이웃의 다양성 점수 [k]
        """
        k = neighbor_hiddens.size(0)
        
        if k == 1:
            return torch.ones(1)
            
        # 모든 쌍 간의 코사인 유사도 계산
        pairwise_similarities = torch.zeros(k, k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    pairwise_similarities[i, j] = F.cosine_similarity(
                        neighbor_hiddens[i].unsqueeze(0), 
                        neighbor_hiddens[j].unsqueeze(0)
                    )
                    
        # 각 이웃의 다양성 = 1 - 다른 이웃들과의 평균 유사도
        diversity_scores = 1 - pairwise_similarities.mean(dim=1)
        
        return diversity_scores
        
    def semantic_filtering(self, 
                          query_hidden: torch.Tensor,
                          neighbor_hiddens: torch.Tensor,
                          neighbor_tokens: torch.Tensor,
                          distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        의미적 관련성에 기반한 필터링
        
        Args:
            query_hidden: 쿼리 hidden state
            neighbor_hiddens: 이웃들의 hidden states [k, hidden_size]
            neighbor_tokens: 이웃들의 토큰 [k]
            distances: 거리 [k]
            
        Returns:
            filtered_hiddens, filtered_tokens, filtered_distances
        """
        relevance_scores = self.compute_relevance_scores(
            query_hidden, neighbor_hiddens, distances
        )
        
        # 관련성 임계값 이상인 이웃들만 유지
        relevant_mask = relevance_scores >= self.relevance_threshold
        
        if not relevant_mask.any():
            # 모든 이웃이 필터링되면 가장 관련성 높은 하나만 유지
            best_idx = relevance_scores.argmax()
            relevant_mask = torch.zeros_like(relevant_mask, dtype=torch.bool)
            relevant_mask[best_idx] = True
            
        filtered_hiddens = neighbor_hiddens[relevant_mask]
        filtered_tokens = neighbor_tokens[relevant_mask]
        filtered_distances = distances[relevant_mask]
        
        logger.debug(f"의미적 필터링: {len(neighbor_hiddens)} -> {len(filtered_hiddens)}")
        
        return filtered_hiddens, filtered_tokens, filtered_distances
        
    def diversity_filtering(self, 
                           neighbor_hiddens: torch.Tensor,
                           neighbor_tokens: torch.Tensor,
                           distances: torch.Tensor,
                           max_neighbors: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        다양성 기반 필터링으로 유사한 이웃들 제거
        
        Args:
            neighbor_hiddens: 이웃들의 hidden states
            neighbor_tokens: 이웃들의 토큰
            distances: 거리
            max_neighbors: 최대 유지할 이웃 수
            
        Returns:
            filtered_hiddens, filtered_tokens, filtered_distances
        """
        k = neighbor_hiddens.size(0)
        
        if k <= max_neighbors:
            return neighbor_hiddens, neighbor_tokens, distances
            
        # 다양성 점수 계산
        diversity_scores = self.compute_diversity_scores(neighbor_hiddens)
        
        # 거리와 다양성을 결합한 점수
        combined_scores = diversity_scores - 0.3 * (distances / distances.max())
        
        # 상위 max_neighbors개 선택
        _, selected_indices = torch.topk(combined_scores, max_neighbors)
        
        filtered_hiddens = neighbor_hiddens[selected_indices]
        filtered_tokens = neighbor_tokens[selected_indices]
        filtered_distances = distances[selected_indices]
        
        logger.debug(f"다양성 필터링: {k} -> {len(filtered_hiddens)}")
        
        return filtered_hiddens, filtered_tokens, filtered_distances
        
    def confidence_filtering(self, 
                            neighbor_hiddens: torch.Tensor,
                            neighbor_tokens: torch.Tensor,
                            distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        신뢰도 기반 필터링
        
        Args:
            neighbor_hiddens: 이웃들의 hidden states
            neighbor_tokens: 이웃들의 토큰
            distances: 거리
            
        Returns:
            filtered_hiddens, filtered_tokens, filtered_distances
        """
        # 거리 기반 신뢰도 계산
        confidences = torch.exp(-distances / distances.mean())
        
        # 신뢰도 임계값 이상인 이웃들만 유지
        confident_mask = confidences >= self.confidence_threshold
        
        if not confident_mask.any():
            # 모든 이웃이 필터링되면 가장 신뢰도 높은 하나만 유지
            best_idx = confidences.argmax()
            confident_mask = torch.zeros_like(confident_mask, dtype=torch.bool)
            confident_mask[best_idx] = True
            
        filtered_hiddens = neighbor_hiddens[confident_mask]
        filtered_tokens = neighbor_tokens[confident_mask]
        filtered_distances = distances[confident_mask]
        
        logger.debug(f"신뢰도 필터링: {len(neighbor_hiddens)} -> {len(filtered_hiddens)}")
        
        return filtered_hiddens, filtered_tokens, filtered_distances
        
    def adaptive_filtering(self, 
                          query_hidden: torch.Tensor,
                          neighbor_hiddens: torch.Tensor,
                          neighbor_tokens: torch.Tensor,
                          distances: torch.Tensor,
                          context_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        컨텍스트에 따른 적응적 필터링
        
        Args:
            query_hidden: 쿼리 hidden state
            neighbor_hiddens: 이웃들의 hidden states
            neighbor_tokens: 이웃들의 토큰
            distances: 거리
            context_length: 현재 컨텍스트 길이 (긴 컨텍스트에서는 더 엄격한 필터링)
            
        Returns:
            filtered_hiddens, filtered_tokens, filtered_distances
        """
        # 컨텍스트 길이에 따라 임계값 조정
        if context_length is not None:
            # 긴 컨텍스트에서는 더 엄격한 필터링
            adaptive_threshold = self.relevance_threshold + 0.1 * min(context_length / 100, 0.2)
        else:
            adaptive_threshold = self.relevance_threshold
            
        # 관련성 점수 계산
        relevance_scores = self.compute_relevance_scores(
            query_hidden, neighbor_hiddens, distances
        )
        
        # 적응적 임계값 적용
        relevant_mask = relevance_scores >= adaptive_threshold
        
        if not relevant_mask.any():
            best_idx = relevance_scores.argmax()
            relevant_mask = torch.zeros_like(relevant_mask, dtype=torch.bool)
            relevant_mask[best_idx] = True
            
        filtered_hiddens = neighbor_hiddens[relevant_mask]
        filtered_tokens = neighbor_tokens[relevant_mask]
        filtered_distances = distances[relevant_mask]
        
        return filtered_hiddens, filtered_tokens, filtered_distances
        
    def filter_neighbors(self, 
                        query_hidden: torch.Tensor,
                        neighbor_hiddens: torch.Tensor,
                        neighbor_tokens: torch.Tensor,
                        distances: torch.Tensor,
                        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        종합적인 이웃 필터링 수행
        
        Args:
            query_hidden: 쿼리 hidden state
            neighbor_hiddens: 이웃들의 hidden states
            neighbor_tokens: 이웃들의 토큰
            distances: 거리
            
        Returns:
            filtered_hiddens, filtered_tokens, filtered_distances, filter_info
        """
        original_count = len(neighbor_hiddens)
        filter_info = {'original_count': original_count}
        
        current_hiddens = neighbor_hiddens
        current_tokens = neighbor_tokens
        current_distances = distances
        
        # 1. 의미적 필터링
        if self.use_semantic_filtering:
            current_hiddens, current_tokens, current_distances = self.semantic_filtering(
                query_hidden, current_hiddens, current_tokens, current_distances
            )
            filter_info['after_semantic'] = len(current_hiddens)
            
        # 2. 신뢰도 필터링
        current_hiddens, current_tokens, current_distances = self.confidence_filtering(
            current_hiddens, current_tokens, current_distances
        )
        filter_info['after_confidence'] = len(current_hiddens)
        
        # 3. 다양성 필터링
        if self.use_diversity_filtering:
            current_hiddens, current_tokens, current_distances = self.diversity_filtering(
                current_hiddens, current_tokens, current_distances
            )
            filter_info['after_diversity'] = len(current_hiddens)
            
        filter_info['final_count'] = len(current_hiddens)
        filter_info['filter_ratio'] = filter_info['final_count'] / original_count
        
        logger.info(f"필터링 완료: {original_count} -> {filter_info['final_count']} "
                   f"(유지율: {filter_info['filter_ratio']:.2f})")
        
        return current_hiddens, current_tokens, current_distances, filter_info
        
    def get_filtering_statistics(self, 
                               query_hidden: torch.Tensor,
                               neighbor_hiddens: torch.Tensor,
                               distances: torch.Tensor) -> Dict:
        """
        필터링 통계 정보 반환 (분석용)
        
        Args:
            query_hidden: 쿼리 hidden state
            neighbor_hiddens: 이웃들의 hidden states
            distances: 거리
            
        Returns:
            통계 정보 딕셔너리
        """
        relevance_scores = self.compute_relevance_scores(
            query_hidden, neighbor_hiddens, distances
        )
        diversity_scores = self.compute_diversity_scores(neighbor_hiddens)
        
        return {
            'relevance_scores': relevance_scores.tolist(),
            'diversity_scores': diversity_scores.tolist(),
            'mean_relevance': relevance_scores.mean().item(),
            'mean_diversity': diversity_scores.mean().item(),
            'min_distance': distances.min().item(),
            'max_distance': distances.max().item(),
            'mean_distance': distances.mean().item(),
            'relevant_count': (relevance_scores >= self.relevance_threshold).sum().item(),
            'diverse_count': (diversity_scores >= self.diversity_threshold).sum().item()
        }
        
    def update_thresholds(self, 
                         relevance_threshold: Optional[float] = None,
                         diversity_threshold: Optional[float] = None,
                         confidence_threshold: Optional[float] = None):
        """임계값 업데이트"""
        if relevance_threshold is not None:
            self.relevance_threshold = relevance_threshold
        if diversity_threshold is not None:
            self.diversity_threshold = diversity_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            
    def __repr__(self):
        return (f"QualityFilter(rel_thresh={self.relevance_threshold}, "
                f"div_thresh={self.diversity_threshold}, "
                f"conf_thresh={self.confidence_threshold})") 