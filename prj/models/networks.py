"""
Neural Network Architectures Module

This module contains the core neural network models for RL-based text generation:
- SimpleTokenPolicy: Transformer-based policy network (actor)
- SimpleValueNetwork: Transformer-based value network (critic)
- PositionalEncoding: Sinusoidal positional embeddings for transformers

How it works:
- Both policy and value networks use transformer encoders to process token sequences
- Policy network outputs logits over vocabulary (next token distribution)
- Value network outputs scalar value estimate (expected future return)
- Positional encoding adds position information to token embeddings

Implementation approach:
- Uses PyTorch transformer layers for efficiency
- Token embeddings + positional encoding -> Transformer -> Task-specific head
- Policy uses last token's representation for next-token prediction
- Value network uses mean pooling across sequence for state value
- Relatively small models (256-dim, 2-4 layers) for faster training
"""

import torch
import torch.nn as nn
import math


class SimpleTokenPolicy(nn.Module):
    """
    Transformer-based policy network for autoregressive token generation

    Architecture: Embedding -> Positional Encoding -> Transformer -> Linear Head
    Outputs probability distribution over vocabulary for next token.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 2, nhead: int = 4):
        """
        Initialize policy network

        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimensionality of model embeddings
            num_layers: Number of transformer encoder layers
            nhead: Number of attention heads
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding layer: vocab_size -> d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding to add position information
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard practice: FFN is 4x model dim
            batch_first=True  # Input shape: [batch, seq, features]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: d_model -> vocab_size
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, token_ids):
        """
        Forward pass to compute next-token logits

        Args:
            token_ids: Token IDs tensor of shape [batch_size, seq_len]

        Returns:
            logits: Unnormalized log probabilities of shape [batch_size, vocab_size]
        """
        # Embed tokens and scale by sqrt(d_model) (standard transformer practice)
        x = self.embedding(token_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Process through transformer layers
        x = self.transformer(x)

        # Take last token's representation for next-token prediction
        # This is autoregressive: predict next token based on all previous tokens
        x = x[:, -1, :]

        # Project to vocabulary size to get logits
        logits = self.output_layer(x)

        return logits

    def get_action_distribution(self, token_ids):
        """
        Get categorical probability distribution over next token

        Used for sampling actions during RL training.

        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]

        Returns:
            Categorical distribution over vocabulary
        """
        logits = self.forward(token_ids)
        return torch.distributions.Categorical(logits=logits)


class SimpleValueNetwork(nn.Module):
    """
    Transformer-based value network to estimate V(state)

    Architecture: Embedding -> Positional Encoding -> Transformer -> Mean Pool -> Linear
    Outputs scalar value estimate representing expected cumulative future reward.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 2, nhead: int = 4):
        """
        Initialize value network

        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimensionality of model embeddings
            num_layers: Number of transformer encoder layers
            nhead: Number of attention heads
        """
        super().__init__()

        self.d_model = d_model

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        # Transformer encoder stack (same architecture as policy)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Value head: projects to single scalar
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, token_ids):
        """
        Forward pass to compute state value estimate

        Args:
            token_ids: Token IDs tensor of shape [batch_size, seq_len]

        Returns:
            value: Scalar value estimate of shape [batch_size, 1]
        """
        # Embed tokens with scaling
        x = self.embedding(token_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Process through transformer
        x = self.transformer(x)

        # Mean pooling over sequence dimension to get fixed-size representation
        # This aggregates information from all tokens
        x = x.mean(dim=1)

        # Project to scalar value estimate
        value = self.value_head(x)

        return value


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers

    Adds position information to token embeddings using sine and cosine functions
    of different frequencies. This allows the model to understand token order.

    Based on "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize positional encoding

        Args:
            d_model: Dimensionality of model embeddings
            max_len: Maximum sequence length to support
        """
        super().__init__()

        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len).unsqueeze(1)

        # Create division term for scaling: exp(log(10000) * -2i/d_model)
        # This creates different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding (broadcasting over batch dimension)
        return x + self.pe[:x.size(1)]