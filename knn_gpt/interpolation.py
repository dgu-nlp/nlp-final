"""
InterpolationModule: 모델 예측과 k-NN 예측 결합

Khandelwal et al. (2019)의 방법을 따라 GPT-2 모델의 예측과 k-NN 검색 결과를
interpolation하여 최종 예측을 생성.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class InterpolationModule:
    """모델 예측과 k-NN 예측을 결합하는 모듈"""
    
    def __init__(self, 
                 lambda_knn: float = 0.25,
                 adaptive_lambda: bool = False,
                 temperature: float = 1.0):
        """
        Args:
            lambda_knn: k-NN 예측에 대한 가중치 (0 = 모델만 사용, 1 = k-NN만 사용)
            adaptive_lambda: 적응적 lambda 사용 여부
            temperature: 최종 softmax에 적용할 온도
        """
        self.lambda_knn = lambda_knn
        self.adaptive_lambda = adaptive_lambda
        self.temperature = temperature
        
    def interpolate_logits(self, 
                          model_logits: torch.Tensor, 
                          knn_logits: torch.Tensor,
                          lambda_knn: Optional[float] = None) -> torch.Tensor:
        """
        모델 로짓과 k-NN 로짓을 interpolation
        
        Args:
            model_logits: 모델의 로짓 [batch_size, vocab_size] 또는 [vocab_size]
            knn_logits: k-NN의 로짓 [batch_size, vocab_size] 또는 [vocab_size]
            lambda_knn: k-NN 가중치 (None이면 self.lambda_knn 사용)
            
        Returns:
            interpolated_logits: 결합된 로짓
        """
        if lambda_knn is None:
            lambda_knn = self.lambda_knn
            
        # 로짓을 확률로 변환
        model_probs = F.softmax(model_logits / self.temperature, dim=-1)
        knn_probs = F.softmax(knn_logits / self.temperature, dim=-1)
        
        # 확률 interpolation
        interpolated_probs = (1 - lambda_knn) * model_probs + lambda_knn * knn_probs
        
        # 다시 로짓으로 변환
        interpolated_logits = torch.log(interpolated_probs + 1e-10)
        
        return interpolated_logits
        
    def interpolate_probs(self, 
                         model_probs: torch.Tensor, 
                         knn_probs: torch.Tensor,
                         lambda_knn: Optional[float] = None) -> torch.Tensor:
        """
        모델 확률과 k-NN 확률을 직접 interpolation
        
        Args:
            model_probs: 모델의 확률 분포
            knn_probs: k-NN의 확률 분포
            lambda_knn: k-NN 가중치
            
        Returns:
            interpolated_probs: 결합된 확률 분포
        """
        if lambda_knn is None:
            lambda_knn = self.lambda_knn
            
        return (1 - lambda_knn) * model_probs + lambda_knn * knn_probs
        
    def adaptive_interpolation(self, 
                             model_logits: torch.Tensor, 
                             knn_logits: torch.Tensor,
                             knn_distances: torch.Tensor,
                             confidence_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k-NN 검색 결과의 신뢰도에 따른 적응적 interpolation
        
        Args:
            model_logits: 모델의 로짓
            knn_logits: k-NN의 로짓
            knn_distances: k-NN 검색에서의 거리들
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            interpolated_logits: 결합된 로짓
            adaptive_lambdas: 각 위치에서 사용된 lambda 값들
        """
        # k-NN 신뢰도 계산 (작은 거리 = 높은 신뢰도)
        if knn_distances.dim() == 1:
            knn_distances = knn_distances.unsqueeze(0)
            
        # 평균 거리를 신뢰도로 변환
        avg_distances = knn_distances.mean(dim=-1)  # [batch_size]
        confidences = torch.exp(-avg_distances)  # 거리가 작을수록 높은 신뢰도
        
        # 적응적 lambda 계산
        adaptive_lambdas = torch.where(
            confidences > confidence_threshold,
            torch.full_like(confidences, self.lambda_knn * 1.5),  # 신뢰도 높으면 k-NN 가중치 증가
            torch.full_like(confidences, self.lambda_knn * 0.5)   # 신뢰도 낮으면 k-NN 가중치 감소
        )
        
        # 각 샘플에 대해 개별적으로 interpolation
        if model_logits.dim() == 1:
            model_logits = model_logits.unsqueeze(0)
            knn_logits = knn_logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = model_logits.size(0)
        interpolated_logits = torch.zeros_like(model_logits)
        
        for i in range(batch_size):
            interpolated_logits[i] = self.interpolate_logits(
                model_logits[i], 
                knn_logits[i], 
                adaptive_lambdas[i].item()
            )
            
        if squeeze_output:
            interpolated_logits = interpolated_logits.squeeze(0)
            adaptive_lambdas = adaptive_lambdas.squeeze(0)
            
        return interpolated_logits, adaptive_lambdas
        
    def entropy_based_interpolation(self, 
                                   model_logits: torch.Tensor, 
                                   knn_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        엔트로피 기반 적응적 interpolation
        모델 예측의 불확실성이 높을 때 k-NN에 더 의존
        
        Args:
            model_logits: 모델의 로짓
            knn_logits: k-NN의 로짓
            
        Returns:
            interpolated_logits: 결합된 로짓
            adaptive_lambdas: 적응적 lambda 값들
        """
        # 모델 예측의 엔트로피 계산
        model_probs = F.softmax(model_logits, dim=-1)
        model_entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-10), dim=-1)
        
        # 엔트로피를 정규화하여 lambda로 사용
        # 높은 엔트로피(불확실성) = 높은 lambda (k-NN에 더 의존)
        max_entropy = torch.log(torch.tensor(model_probs.size(-1), dtype=torch.float))
        normalized_entropy = model_entropy / max_entropy
        adaptive_lambdas = self.lambda_knn * (1 + normalized_entropy)
        
        # Clamp to valid range
        adaptive_lambdas = torch.clamp(adaptive_lambdas, 0, 1)
        
        # Interpolation 수행
        if model_logits.dim() == 1:
            interpolated_logits = self.interpolate_logits(
                model_logits, knn_logits, adaptive_lambdas.item()
            )
        else:
            batch_size = model_logits.size(0)
            interpolated_logits = torch.zeros_like(model_logits)
            
            for i in range(batch_size):
                interpolated_logits[i] = self.interpolate_logits(
                    model_logits[i], 
                    knn_logits[i], 
                    adaptive_lambdas[i].item()
                )
                
        return interpolated_logits, adaptive_lambdas
        
    def context_aware_interpolation(self, 
                                   model_logits: torch.Tensor, 
                                   knn_logits: torch.Tensor,
                                   context_similarity: torch.Tensor) -> torch.Tensor:
        """
        컨텍스트 유사도에 기반한 interpolation
        검색된 컨텍스트가 현재 컨텍스트와 유사할 때 k-NN에 더 의존
        
        Args:
            model_logits: 모델의 로짓
            knn_logits: k-NN의 로짓
            context_similarity: 컨텍스트 유사도 점수 [0, 1]
            
        Returns:
            interpolated_logits: 결합된 로짓
        """
        # 유사도에 기반하여 lambda 조정
        adaptive_lambda = self.lambda_knn * context_similarity
        
        return self.interpolate_logits(model_logits, knn_logits, adaptive_lambda.item())
        
    def get_interpolation_info(self, 
                              model_logits: torch.Tensor, 
                              knn_logits: torch.Tensor) -> Dict:
        """
        Interpolation에 대한 상세 정보 반환 (분석용)
        
        Args:
            model_logits: 모델의 로짓
            knn_logits: k-NN의 로짓
            
        Returns:
            정보 딕셔너리
        """
        model_probs = F.softmax(model_logits, dim=-1)
        knn_probs = F.softmax(knn_logits, dim=-1)
        
        # 예측 차이 계산
        kl_divergence = F.kl_div(
            F.log_softmax(model_logits, dim=-1), 
            knn_probs, 
            reduction='none'
        ).sum(dim=-1)
        
        # 각 예측의 엔트로피
        model_entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-10), dim=-1)
        knn_entropy = -torch.sum(knn_probs * torch.log(knn_probs + 1e-10), dim=-1)
        
        return {
            'lambda_knn': self.lambda_knn,
            'kl_divergence': kl_divergence.item() if kl_divergence.dim() == 0 else kl_divergence.tolist(),
            'model_entropy': model_entropy.item() if model_entropy.dim() == 0 else model_entropy.tolist(),
            'knn_entropy': knn_entropy.item() if knn_entropy.dim() == 0 else knn_entropy.tolist(),
            'model_top_prob': model_probs.max(dim=-1)[0].item() if model_probs.dim() == 1 else model_probs.max(dim=-1)[0].tolist(),
            'knn_top_prob': knn_probs.max(dim=-1)[0].item() if knn_probs.dim() == 1 else knn_probs.max(dim=-1)[0].tolist()
        }
        
    def update_parameters(self, 
                         lambda_knn: Optional[float] = None, 
                         temperature: Optional[float] = None):
        """파라미터 업데이트"""
        if lambda_knn is not None:
            self.lambda_knn = lambda_knn
        if temperature is not None:
            self.temperature = temperature
            
    def __repr__(self):
        return f"InterpolationModule(lambda_knn={self.lambda_knn}, adaptive={self.adaptive_lambda}, temp={self.temperature})" 