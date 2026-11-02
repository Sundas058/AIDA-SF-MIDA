from __future__ import absolute_import

from torch import nn
import torch
import torch.nn.functional as F


class ViTBlock(nn.Module):
    """
    Vision Transformer (ViT) encoder block.
    
    A lightweight transformer block that can be stacked on top of CNN feature maps
    to capture long-range dependencies and improve feature representation.
    """
    
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension (should match CNN feature channels)
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embed_dim
            dropout: Dropout probability
        """
        super(ViTBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # Use (seq_len, batch, embed_dim) format
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass through ViT block.
        
        Args:
            x: [seq_len, batch, embed_dim] input tokens
        
        Returns:
            out: [seq_len, batch, embed_dim] output tokens
        """
        # Self-attention with residual connection
        # Input format: [seq_len, batch, embed_dim]
        attn_out, _ = self.attn(x, x, x)  # Self-attention: Q=K=V=x
        x = self.norm1(x + attn_out)
        
        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class ViTStack(nn.Module):
    """
    Stack of ViT blocks for processing CNN feature maps.
    
    Takes CNN feature maps, flattens spatial dimensions into tokens,
    applies transformer blocks, and optionally pools the output.
    """
    
    def __init__(self, embed_dim, num_blocks=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension (should match CNN feature channels)
            num_blocks: Number of ViT blocks to stack
            num_heads: Number of attention heads per block
            mlp_ratio: Ratio of MLP hidden dimension to embed_dim
            dropout: Dropout probability
        """
        super(ViTStack, self).__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        
        # Stack of ViT blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Forward pass through ViT stack.
        
        Args:
            x: [B, C, H, W] CNN feature maps
        
        Returns:
            out: [B, C, H, W] transformed feature maps (same spatial dimensions)
        """
        B, C, H, W = x.size()
        
        # Flatten spatial dimensions into tokens: [B, C, H, W] -> [H*W, B, C]
        # This treats each spatial location as a token
        x = x.view(B, C, H * W)  # [B, C, H*W]
        x = x.permute(2, 0, 1)   # [H*W, B, C] (seq_len, batch, embed_dim)
        
        # Apply stack of ViT blocks
        for block in self.blocks:
            x = block(x)
        
        # Reshape back to spatial format: [H*W, B, C] -> [B, C, H, W]
        x = x.permute(1, 2, 0)   # [B, C, H*W]
        x = x.view(B, C, H, W)   # [B, C, H, W]
        
        return x


