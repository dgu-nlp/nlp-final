from typing import Optional

import torch
import torch.nn as nn


class KNNParaphraseClassifier(nn.Module):
    """Lightweight wrapper that adapts ``KNNAugmentedGPT2`` so it can be evaluated with
    ``evaluation.model_eval_paraphrase``.

    The evaluation helpers expect ``model(input_ids, attention_mask)`` to return a
    *tensor* of shape ``[batch, vocab_size]`` (token-level logits).  ``KNNAugmentedGPT2``
    instead returns a dict that contains those logits under the key ``'logits'``.

    This wrapper simply forwards the call and extracts that field, so existing
    evaluation utilities can be reused without modification.
    """

    def __init__(self, knn_gpt_model: nn.Module):
        """Args
        -----
        knn_gpt_model: An instance of :class:`models.knn_gpt2.KNNAugmentedGPT2` that has
            already been initialised (and, if needed, had a datastore attached).
        """
        super().__init__()
        self.knn_gpt_model = knn_gpt_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.knn_gpt_model(input_ids, attention_mask)
        # ``KNNAugmentedGPT2`` returns a dict.  We only need the token-level logits.
        if isinstance(outputs, dict):
            return outputs["logits"]
        # Fallback: if someone passes the underlying GPT2Model directly
        return outputs

    # Convenience proxy methods ------------------------------------------------
    def set_datastore(self, *args, **kwargs):
        if hasattr(self.knn_gpt_model, "set_datastore"):
            return self.knn_gpt_model.set_datastore(*args, **kwargs)

    def update_knn_parameters(self, *args, **kwargs):
        if hasattr(self.knn_gpt_model, "update_knn_parameters"):
            return self.knn_gpt_model.update_knn_parameters(*args, **kwargs)

    def enable_knn(self):
        if hasattr(self.knn_gpt_model, "enable_knn"):
            return self.knn_gpt_model.enable_knn()

    def disable_knn(self):
        if hasattr(self.knn_gpt_model, "disable_knn"):
            return self.knn_gpt_model.disable_knn() 