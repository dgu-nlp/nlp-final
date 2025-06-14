"""
k-NN 증강 감성 분류기

기존 GPT2SentimentClassifier에 k-NN 검색 메커니즘을 추가하여
분류 성능을 향상.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

from .knn_gpt2 import KNNAugmentedGPT2
from knn_gpt import DataStore

logger = logging.getLogger(__name__)


class KNNSentimentClassifier(nn.Module):
    """k-NN 증강 감성 분류기"""
    
    def __init__(self, 
                 base_classifier,
                 datastore: Optional[DataStore] = None,
                 k: int = 8,
                 lambda_knn: float = 0.25,
                 knn_temperature: float = 10.0,
                 use_quality_filter: bool = True):
        """
        Args:
            base_classifier: 기존 GPT2SentimentClassifier
            datastore: k-NN 검색용 데이터스토어
            k: 검색할 이웃 수
            lambda_knn: k-NN 예측 가중치
            knn_temperature: k-NN 검색 온도
            use_quality_filter: 품질 필터링 사용 여부
        """
        super().__init__()
        
        self.base_classifier = base_classifier
        self.num_labels = base_classifier.num_labels
        
        # k-NN 증강 GPT-2 모델 생성
        self.knn_gpt = KNNAugmentedGPT2(
            base_model=base_classifier.gpt,
            datastore=datastore,
            k=k,
            lambda_knn=lambda_knn,
            knn_temperature=knn_temperature,
            use_quality_filter=use_quality_filter,
            vocab_size=base_classifier.gpt.word_embedding.num_embeddings
        )
        
        # 분류 헤드는 기존 것 사용
        self.classifier = base_classifier.classifier
        self.dropout = base_classifier.droupout if hasattr(base_classifier, 'droupout') else nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask, use_knn=True):
        """
        forward pass with optional k-NN augmentation
        
        Args:
            input_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크  
            use_knn: k-NN 증강 사용 여부
        """
        if use_knn and self.knn_gpt.knn_retriever is not None:
            # k-NN 증강 사용
            outputs = self.knn_gpt(input_ids, attention_mask)
            last_hidden_states = outputs['last_token']
        else:
            # 기존 모델만 사용
            outputs = self.base_classifier.gpt(input_ids, attention_mask)
            last_hidden_states = outputs['last_token']
        
        # 분류
        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.classifier(last_hidden_states)
        
        return logits
        
    def set_datastore(self, datastore: DataStore):
        """데이터스토어 설정"""
        self.knn_gpt.set_datastore(datastore)
        
    def enable_knn(self):
        """k-NN 검색 활성화"""
        self.knn_gpt.enable_knn()
        
    def disable_knn(self):
        """k-NN 검색 비활성화"""
        self.knn_gpt.disable_knn()
        
    def update_knn_parameters(self, **kwargs):
        """k-NN 파라미터 업데이트"""
        self.knn_gpt.update_knn_parameters(**kwargs) 