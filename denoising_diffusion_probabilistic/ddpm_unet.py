import torch
import torch.nn as nn

from denoising_diffusion_probabilistic.layer import DownSample, \
    DownSampleAttention, \
    MiddleSampleAttension, \
    UpSampleAttention, \
    PositionalEmbedding, \
    UpSample

class DDPMUnet(nn.Module):
    def __init__(self, image_size=256, input_channels=3, max_timesteps=1000):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.positional_encoding = nn.Sequential(
            PositionalEmbedding(dim=128, max_timesteps=max_timesteps),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.downsample_blocks = nn.ModuleList([
            DownSample(in_size=128, out_size=128, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            DownSample(in_size=128, out_size=128, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            DownSample(in_size=128, out_size=256, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            DownSampleAttention(in_size=256, out_size=256, n_layers=2, num_att_heads=4, n_groups=32, time_emb_size=128 * 4),
            DownSample(in_size=256, out_size=512, n_layers=2, n_groups=32, time_emb_size=128 * 4)
        ])

        self.bottleneck = MiddleSampleAttension(in_size=512, out_size=512, n_layers=2, num_att_heads=4, n_groups=32, time_emb_size=128*4)                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            UpSample(in_size=512 + 512, out_size=512, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            UpSampleAttention(in_size=512 + 256, out_size=256, n_layers=2, num_att_heads=4, n_groups=32, time_emb_size=128 * 4),
            UpSample(in_size=256 + 256, out_size=256, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            UpSample(in_size=256 + 128, out_size=128, n_layers=2, n_groups=32, time_emb_size=128 * 4),
            UpSample(in_size=128 + 128, out_size=128, n_layers=2, n_groups=32, time_emb_size=128 * 4)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)

        initial_x = self.initial_conv(input_tensor)

        states_for_skip_connections = [initial_x]

        x = initial_x
        for i, block in enumerate(self.downsample_blocks):
            x = block(x, time_encoded)
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))

        x = self.bottleneck(x, time_encoded)

        for i, (block, skip) in enumerate(zip(self.upsample_blocks, states_for_skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)

        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        # Get initial shape [3, 256, 256] with convolutions
        out = self.output_conv(x)

        return out