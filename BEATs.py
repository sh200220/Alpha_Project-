# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi
import sys
sys.path.append('/content/drive/MyDrive/unilm/beats/BEATs')
from backbone import (
    TransformerEncoder,
)

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = 1.0  # ratio for layer-wise gradient decay
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = True  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = 0.0  # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.05  # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.1  # dropout to apply to the input (after feat extr)

        # positional embeddings
        self.conv_pos: int = 128  # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = True  # apply relative position embedding
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = 800  # maximum distance for relative position embedding
        self.gru_rel_pos: bool = True  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    def __init__(
            self,
            cfg: BEATsConfig,
    ) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size,
                                         bias=cfg.conv_bias)

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
            self,
            features: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
            self,
            source: torch.Tensor,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        do_preprocess: bool = True,
        skip_predictor: bool = False,  # Predictor 호출 여부를 제어하는 매개변수 추가
):
      #print(f"[DEBUG] Initial Source Shape: {source.shape}")
      if do_preprocess:
          # Preprocess
          fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)
          #print(f"[DEBUG] Fbank Shape After Preprocess: {fbank.shape}")

          if padding_mask is not None:
              padding_mask = self.forward_padding_mask(fbank, padding_mask)
              #print(f"[DEBUG] Padding Mask Shape After forward_padding_mask: {padding_mask.shape}")

          fbank = fbank.unsqueeze(1)
          #print(f"[DEBUG] Fbank Shape After Unsqueeze: {fbank.shape}")
      else:
          fbank = source
          #print(f"[DEBUG] Fbank Shape (No Preprocessing): {fbank.shape}")

      # Patch Embedding
      features = self.patch_embedding(fbank)
      #print(f"[DEBUG] Features Shape After Patch Embedding: {features.shape}")

      features = features.reshape(features.shape[0], features.shape[1], -1)
      #print(f"[DEBUG] Features Shape After Reshape: {features.shape}")

      features = features.transpose(1, 2)
      #print(f"[DEBUG] Features Shape After Transpose: {features.shape}")

      features = self.layer_norm(features)
      #print(f"[DEBUG] Features Shape After LayerNorm: {features.shape}")

      if padding_mask is not None:
          padding_mask = self.forward_padding_mask(features, padding_mask)
          #print(f"[DEBUG] Padding Mask Shape After Forward Padding Mask (Second Time): {padding_mask.shape}")

      if self.post_extract_proj is not None:
          features = self.post_extract_proj(features)
          #print(f"[DEBUG] Features Shape After Post Extract Proj: {features.shape}")

      # Dropout
      x = self.dropout_input(features)
      #print(f"[DEBUG] Features Shape After Dropout Input: {x.shape}")

      # Encoder
      x, layer_results = self.encoder(x, padding_mask=padding_mask)
      #print(f"[DEBUG] Encoder Output Shape: {x.shape}")

      # Predictor 호출 여부 제어
      if not skip_predictor and self.predictor is not None:
          x = self.predictor_dropout(x)
          #print(f"[DEBUG] Features Shape After Predictor Dropout: {x.shape}")

          logits = self.predictor(x)
          #print(f"[DEBUG] Logits Shape From Predictor: {logits.shape}")

          if padding_mask is not None and padding_mask.any():
              logits[padding_mask] = 0
              logits = logits.sum(dim=1)
              logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
          else:
              logits = logits.mean(dim=1)
          print(f"[DEBUG] Logits Shape After Mean Reduction: {logits.shape}")

          lprobs = torch.sigmoid(logits)
          print(f"[DEBUG] Logits Shape After Sigmoid: {lprobs.shape}")

          return lprobs, padding_mask
      else:
          # Predictor를 건너뛸 경우, Encoder 출력만 반환
          return x, padding_mask

      # Predictor
      if self.predictor is not None:
          x = self.predictor_dropout(x)
          #print(f"[DEBUG] Features Shape After Predictor Dropout: {x.shape}")

          logits = self.predictor(x)
          #print(f"[DEBUG] Logits Shape From Predictor: {logits.shape}")

          if padding_mask is not None and padding_mask.any():
              logits[padding_mask] = 0
              logits = logits.sum(dim=1)
              logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
              #print(f"[DEBUG] Logits Shape After Mask Handling: {logits.shape}")
          else:
              logits = logits.mean(dim=1)
              #print(f"[DEBUG] Logits Shape After Mean Reduction: {logits.shape}")

          lprobs = torch.sigmoid(logits)
          #print(f"[DEBUG] Logits Shape After Sigmoid: {lprobs.shape}")

          return lprobs, padding_mask
      else:
          #print(f"[DEBUG] Returning Encoder Output Without Predictor")
          return x, padding_mask