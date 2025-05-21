from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: forward() 함수를 위한 이 helper 메서드를 구현하시오:
      - 이 함수는 multi-head attention layer와 feed forward layer 이후에 적용된다.
      - GPT-2 layer는 각 sublayer의 변환된 출력에 드롭아웃을 적용한 후, 이를 sublayer 입력에 더한다. 
        이 함수에서는 Layer Normalization을 적용하지 않는다.
    """
    ### 완성시켜야 할 빈 코드 블록
    raise NotImplementedError


  def forward(self, hidden_states, attention_mask):
    """
    TODO: forward pass의 구현. 고려해야 할 주요 사항은 다음과 같다:
      - Multi-head Attention layer(CausalSelfAttention): mask된 입력을 기반으로 self-attention을 계산한다.
      - Layer Normalization: Attention layer와 Feed-forward layer 이전에 적용된다.
      - Dropout, Residual Connection, Layer Normalization를 적용하시오(self.add() 메서드를 사용)
      - Feed-Forward layer: hidden states를 추가로 refine하기 위해 변환을 적용한다.
    """

    ### 완성시켜야 할 빈 코드 블록
    raise NotImplementedError
