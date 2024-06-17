import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The fundamental convolutional blocks' determinations

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, num_groups=8):
        super(ConvBlock, self).__init__()
        
        self.cnn = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # According to authors, they change the normalization technique from
        # weight to group.
        self.norm = nn.GroupNorm(num_groups, out_size)
        self.act = nn.SiLU()
    
    def forward(self, input):
        
        output = self.cnn(input)
        output = self.norm(output)
        output = self.act(output)
        
        return output

class ResNetBlock(nn.Module):
    def __init__(self, in_size, out_size, time_embbeding=None, num_groups=8):
        super(ResNetBlock, self).__init__()
        
        self.time_embedding_projitile = (
            # nn.SiLU a.k.a. swish function
            nn.Sequential(nn.SiLU(), nn.Linear(time_embbeding, out_size))
            if time_embbeding else None
        ) 
        
        self.cnn1 = ConvBlock(in_size, out_size, num_groups=num_groups)
        self.cnn2 = ConvBlock(out_size, out_size, num_groups=num_groups)
        self.residual_conv = nn.Conv2d(in_size, out_size, 1) if in_size != out_size else nn.Identity()
        
    def forward(self, input, time_embedding=None):
        tensor = input
        h = self.cnn1(tensor)
        
        time_emb = self.time_embedding_projitile(time_embedding)
        time_emb = time_emb[:, :, None, None]
        output = time_emb + h
        
        output = self.cnn2(output)
        
        return output + self.residual_conv(tensor)

class SelfAttention(nn.Module):
    # The class aims to construct the self attention linear layer block from "Attention Is All You Need"
    # with most of the implementations are built by 
    # https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py
    def __init__(self, n_heads, in_size, n_groups=32, embedding_dim=256):
        super(SelfAttention, self).__init__()
        
        self.num_heads = n_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // n_heads
        self.d_values = embedding_dim // n_heads

        self.Q = nn.Linear(in_size, embedding_dim)
        self.K = nn.Linear(in_size, embedding_dim)
        self.V = nn.Linear(in_size, embedding_dim)

        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=n_groups)
        
    def split_features_for_heads(self, tensor):
        # We receive Q, K and V at shape [batch, h*w, embedding_dim].
        # This method splits embedding_dim into 'num_heads' features so that
        # each channel becomes of size embedding_dim / num_heads.
        # Output shape becomes [batch, num_heads, h*w, embedding_dim/num_heads],
        # where 'embedding_dim/num_heads' is equal to d_k = d_k = d_v = sizes for
        # K, Q and V respectively, according to paper.
        _, _, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1)
        heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
        
        return heads_splitted_tensor

    def forward(self, input):
        x = input
        batch, features, h, w = x.shape
        # Do reshape and transpose input tensor since we want to process depth feature maps, not spatial maps
        x = x.view(batch, features, h * w).transpose(1, 2)

        # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
        Q = self.Q(x)  # [b, in_channels, embedding_dim]
        K = self.K(x)       # [b, in_channels, embedding_dim]
        V = self.V(x)   # [b, in_channels, embedding_dim]

        # Split Q, K, V between attention heads to process them simultaneously
        Q = self.split_features_for_heads(Q)
        K = self.split_features_for_heads(K)
        V = self.split_features_for_heads(V)

        # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
        # Each SDPA block yields tensor of size d_v = embedding_dim/num_heads.
        scale = self.d_keys ** (-0.5)
        attention_scores = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, V)

        # Permute computed attention scores such that
        # [batch, num_heads, h*w, embedding_dim] --> [batch, h*w, num_heads, d_v]
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # Concatenate scores per head into one tensor so that
        # [batch, h*w, num_heads, d_v] --> [batch, h*w, num_heads*d_v]
        concatenated_heads_attention_scores = attention_scores.view(batch, h * w, self.d_model)

        # Perform linear projection and reshape tensor such that
        # [batch, h*w, d_model] --> [batch, d_model, h*w] -> [batch, d_model, h, w]
        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(batch, self.d_model, h, w)

        # Residual connection + norm
        x = self.norm(linear_projection + input)
        
        return x

# The following convolutional blocks' determinations are those usually been used to built the hourglass network

class PositionalEmbedding(nn.Module):
    """The sinusoidal embedding in Section 3.5 of "Attention Is All You Need", but differs slightly.
    The original implementation is in TensorFlow and can be found in the `tensor2tensor` GitHub repository. 
    This is a similar implementation in PyTorch.
    """
    def __init__(self, dim, max_timesteps=1000):
        super(PositionalEmbedding, self).__init__()
        
        self.dim = dim
        self.max_timesteps = max_timesteps
    
    def forward(self, input):
        device = input.device
        half_dim = self.dim // 2
        emb = math.log(self.max_timesteps) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = input[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class DownSample(nn.Module):
    def __init__(self, in_size, out_size, n_layers, time_emb_size, n_groups):
        super(DownSample, self).__init__()
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_size if i == 0 else out_size, out_size, time_emb_size, n_groups) 
            for i in range(n_layers)
        ])
        
        self.cnn = nn.Conv2d(out_size, out_size, 3, stride=2, padding=1)

    def forward(self, input, time_emb_size):
        for resnet_block in self.resnet_blocks:
            input = resnet_block(input, time_emb_size)
        output = self.cnn(input)
        
        return output
    
class UpSample(nn.Module):
    def __init__(self, in_size, out_size, n_layers, time_emb_size, n_groups, scale_factor=2.0):
        super(UpSample, self).__init__()
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_size if i == 0 else out_size, out_size, time_emb_size, n_groups) 
            for i in range(n_layers)
        ])
        
        self.scale = scale_factor
        self.cnn = nn.Conv2d(out_size, out_size, 3, padding=1)

    def forward(self, input, time_emb_size):
        for resnet_block in self.resnet_blocks:
            input = resnet_block(input, time_emb_size)
        # align_corners=True for potential convertibility to ONNX
        output = F.interpolate(input, scale_factor=self.scale, mode="bilinear", align_corners=True)
        output = self.cnn(output)
        
        return output

class DownSampleAttention(nn.Module):
    def __init__(self, in_size, out_size, n_layers, time_emb_size, n_groups, num_att_heads):
        """
        DownSampleAttention consists of ResNet blocks with Self-Attention blocks in-between
        """
        super(DownSampleAttention, self).__init__()
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_size if i == 0 else out_size, out_size, time_emb_size, n_groups) 
            for i in range(n_layers)
        ])
        
        self.attention_blocks = nn.ModuleList([
            SelfAttention(num_att_heads, out_size, n_groups, out_size)
            for i in range(n_layers)
        ])
        
        self.cnn = nn.Conv2d(out_size, out_size, 3, stride=2, padding=1)
    
    def forward(self, input, time_emb_size):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            input = resnet_block(input, time_emb_size)
            input = attention_block(input)
            
        output = self.cnn(input)
        
        return output
    
class MiddleSampleAttension(nn.Module):
    def __init__(self, in_size, out_size, n_layers, time_emb_size, n_groups, num_att_heads):
        """
        MiddleSampleAttension consists of ResNet blocks with Self-Attention blocks in-between.
        This class is made for the bottleneck of the nueral network.
        """
        super(MiddleSampleAttension, self).__init__()
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_size if i == 0 else out_size, out_size, time_emb_size, n_groups) 
            for i in range(n_layers)
        ])
        
        self.attention_blocks = nn.ModuleList([
            SelfAttention(num_att_heads, out_size, n_groups, out_size)
            for i in range(n_layers)
        ])
    
    def forward(self, input, time_emb_size):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            input = resnet_block(input, time_emb_size)
            input = attention_block(input)
        
        return input

class UpSampleAttention(nn.Module):
    def __init__(self, in_size, out_size, n_layers, time_emb_size, n_groups, num_att_heads, scale_factor=2.0):
        """
        UpSampleAttention consists of ResNet blocks with Self-Attention blocks in-between
        """
        super(UpSampleAttention, self).__init__()
        
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(in_size if i == 0 else out_size, out_size, time_emb_size, n_groups) 
            for i in range(n_layers)
        ])
        
        self.attention_blocks = nn.ModuleList([
            SelfAttention(num_att_heads, out_size, n_groups, out_size)
            for i in range(n_layers)
        ])
        
        self.scale = scale_factor
        self.cnn = nn.Conv2d(out_size, out_size, 3, padding=1)
    
    def forward(self, input, time_emb_size):
        for resnet_block, attention_block in zip(self.resnet_blocks, self.attention_blocks):
            input = resnet_block(input, time_emb_size)
            input = attention_block(input)
        # align_corners=True for potential convertibility to ONNX
        output = F.interpolate(input, scale_factor=self.scale, mode="bilinear", align_corners=True)
        output = self.cnn(output)
        
        return output