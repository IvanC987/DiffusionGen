import torch
from torch import nn
from diffusers import AutoencoderKL


class TimeEncoding(nn.Module):
    def __init__(self, T: int, t_embd: int):
        """
        Using the original positional encoding method from Transformers paper

        :param T: Maximum number of timesteps
        :param t_embd: Embedding dimension for timesteps
        """

        super().__init__()
        te = torch.zeros((T, t_embd))  # Init empty tensor
        t = torch.arange(T).unsqueeze(1)  # shape=(T, 1)
        div_term = 10_000 ** (torch.arange(0, t_embd, 2, dtype=torch.float32) / t_embd)  # shape=(t_embd//2)

        te[:, 0::2] = torch.sin(t / div_term)
        te[:, 1::2] = torch.cos(t / div_term)

        self.register_buffer("te", te)

    def forward(self, t: int):
        return self.te[t]


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, n_heads):
        """
        Given input of shape (b, hw, c) it applies the attention mechanism and returns output of same shape

        :param channels: Number of channels in input tensor
        :param n_heads:  Number of attention heads ot use
        """

        super().__init__()
        assert channels % n_heads == 0, f"Number of channels, {channels=}, should be divisible by num heads, {n_heads=}!"
        self.channels = channels
        self.n_heads = n_heads
        self.h_dim = channels // n_heads

        self.qkv = nn.Linear(channels, 3 * channels)
        self.w = nn.Linear(channels, channels)



    def forward(self, x):
        # x.shape = (b, hw, c)
        b, hw, c = x.shape

        # Same shape for q k v
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (b, hw, c) -> (b, hw, n_heads, h_dim) -> (b, n_heads, hw, h_dim)
        q = q.reshape(b, hw, self.n_heads, self.h_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, self.h_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, self.h_dim).permute(0, 2, 1, 3)

        # (b, n_heads, hw, h_dim) @ (b, n_heads, h_dim, hw) -> (b, n_heads, hw, hw)
        h = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)  # Normalize by sqrt(h_dim)

        # No causal mask is needed here since we want all pixels to attend to each other
        h = nn.functional.softmax(h, dim=-1)

        # Test out later on
        # h = h.permute(0, 1, 3, 2)  # Original LDM code swapped k's hw and q's hw

        # (b, n_heads, hw, hw) @ (b, n_heads, hw, h_dim) -> (b, n_heads, hw, h_dim)
        h = h @ v

        # (b, n_heads, hw, h_dim) -> (b, hw, n_heads, h_dim) -> (b, hw, c)
        h = h.permute(0, 2, 1, 3).reshape(b, hw, c)

        return self.w(h)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, n_heads, n_embd):
        """
        Similar to SelfAttention, but this uses input latent repr as query matrix and prompt embedding via CLIP as kv matrices
        Input/Output shape are the same, (b, hw, c)

        :param channels: Number of channels from input
        :param n_heads: Number of heads to use for CA
        :param n_embd: Embedding dimension of the prompt tensor (Fixed at 512 for CLIP)
        """

        super().__init__()
        assert channels % n_heads == 0, f"Number of channels, {channels=}, should be divisible by num heads, {n_heads=}!"
        self.channels = channels
        self.n_heads = n_heads
        self.h_dim = channels // n_heads

        # Consider q and kv comes from different sources, it's probably better to have 3 class variables
        # self.q shape remains the same
        # self.k/v shape (b, 77, 512) -> (b, 77, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(n_embd, channels)
        self.v = nn.Linear(n_embd, channels)
        self.w = nn.Linear(channels, channels)


    def forward(self, x, p):
        # x.shape = (b, hw, c)
        # p.shape = (b, 77, 512)

        b, hw, c, = x.shape
        q = self.q(x)  # (b, hw, c)
        k = self.k(p)  # (b, 77, c)
        v = self.v(p)  # (b, 77, c)

        # (b, hw, c) -> (b, hw, n_heads, h_dim) -> (b, n_heads, hw, h_dim)
        q = q.reshape(b, hw, self.n_heads, self.h_dim).permute(0, 2, 1, 3)

        # (b, 77, c) -> (b, 77, n_heads, h_dim) -> (b, n_heads, 77, h_dim)
        k = k.reshape(b, 77, self.n_heads, self.h_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, 77, self.n_heads, self.h_dim).permute(0, 2, 1, 3)

        # (b, n_heads, hw, h_dim) @ (b, n_heads, h_dim, 77) -> (b, n_heads, hw, 77)
        h = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)  # Normalize by sqrt(h_dim)

        # No causal mask is needed here since we want all pixels to attend to each other
        h = nn.functional.softmax(h, dim=-1)

        # Test out later on
        # h = h.permute(0, 1, 3, 2)  # Original LDM code swapped k's hw and q's hw

        # (b, n_heads, hw, 77) @ (b, n_heads, 77, h_dim) -> (b, n_heads, hw, h_dim)
        h = h @ v

        # (b, n_heads, hw, h_dim) -> (b, hw, n_heads, h_dim) -> (b, hw, c)
        h = h.permute(0, 2, 1, 3).reshape(b, hw, c)

        # Final shape = (b, hw, c), same as input
        return self.w(h)


class UnetAttentionBlock(nn.Module):
    """
    This serves as a single attention block by combining self attention, cross attention, gn/ln layers and geglu
    Felt like the one shown by Umar Jamil works very well, so I just borrowed this block with minor changes
    """

    def __init__(self, channels: int, n_groups: int, n_heads: int, n_embd: int):
        """
        Combines the entire attention portion into a single block for modularity.
        Uses MHSA, MHCA, Conv layers, Residual connection, GeGLU and such.

        :param channels: Number of channels of input tensor
        :param n_groups: Number of normalization groups for groupnorm
        :param n_heads: Number of attention heads
        :param n_embd: Number of embedding dimension (Fixed at 512 for CLIP)
        """

        super().__init__()
        self.groupnorm = nn.GroupNorm(n_groups, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.self_attention = SelfAttentionBlock(channels, n_heads)
        self.ln2 = nn.LayerNorm(channels)
        self.cross_attention = CrossAttentionBlock(channels, n_heads, n_embd)
        self.ln3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, p):
        # x: (b, c, h, w)
        # p: (b, 77, 512)

        # (b, c, h, w)
        residue_long = x

        # Shape stays the same, (b, c, h, w)
        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        # (b, c, h, w) -> (b, c, hw)
        x = x.view(b, c, h * w)
        # (b, c, hw) -> (b, hw, c)
        x = x.permute(0, 2, 1)

        # (b, hw, c)
        residue_short = x

        x = self.ln1(x)
        x = self.self_attention(x)

        # ----------------------------
        # shape stays the same throughout, (b, hw, c)
        x += residue_short
        residue_short = x

        x = self.ln2(x)
        x = self.cross_attention(x, p)

        x += residue_short
        residue_short = x

        x = self.ln3(x)
        # ----------------------------

        # "GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10"
        # After linear_geglu_1, x.shape -> (b, hw, 8 * c) then split into two tensors of shape (b, hw, 4 * c)
        # This section is applying the gating mechanism/FFWD
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * nn.functional.gelu(gate)

        # (b, hw, 4 * c) -> (b, hw, c)
        x = self.linear_geglu_2(x)
        x += residue_short

        # (b, hw, c) -> (b, c, hw) -> (b, c, h, w)
        x = x.permute(0, 2, 1)
        x = x.view(b, c, h, w)

        # Finally, add residual connection and return (b, c, h, w)
        return self.conv_output(x) + residue_long


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_groups: int,
                 t_embd: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float):
        """
        Encoder portion of the UNet

        :param in_channels: Channels of input image
        :param out_channels: Desired channels for output image
        :param n_groups: Number of norm groups for groupnorm
        :param t_embd: Embedding dimension for timesteps
        :param n_embd: Embedding dimension for prompts tensors (Fixed at 512 for CLIP)
        :param n_heads: Number of attention heads
        :param n_layers: Number of sequential layers for this block
        :param dropout: Dropout prob
        """

        super().__init__()

        # Each Encoder block is composed of n-layers, dictated by the n_layers parameter
        # Overall structure of each layer is
        # (GroupNorm -> SiLU -> Conv2d -> Dropout2d -> t_embd -> GroupNorm -> SiLU -> Conv2d -> Dropout2d, -> attention block) repeat n_layers times
        # There's an intermediate conv block for initial resconnection then at the very end of the block, downsample.
        self.n_layers = n_layers

        self.resnet1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for i in range(n_layers)
        ])

        self.t_embd_linear = nn.Linear(t_embd, out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.resnet2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for _ in range(n_layers)
        ])

        self.attention_blocks = nn.ModuleList([UnetAttentionBlock(channels=out_channels, n_groups=n_groups, n_heads=n_heads, n_embd=n_embd) for _ in range(n_layers)])

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)


    def forward(self, x, p, t):
        """
        :param x: Input image (latent)
        :param p: Prompts tensor (b, 77, 512)
        :param t: Timestep tensor (b,)
        :return: Returns the downsampled output tensor along with it's corresponding skip connection tensor
        """

        h = x

        for i in range(self.n_layers):
            res = h  # residual connection
            h = self.resnet1[i](h)

            t_embd = self.t_embd_linear(t)  # Create timestep embedding

            # Need to unsqueeze it to match the dimensions of x and p
            h += t_embd.unsqueeze(-1).unsqueeze(-1)

            h += (self.residual_conv(res) if i == 0 else res) / (2 ** 0.5)  # Add residual and stabilize gradients
            res = h  # "midpoint" residual

            h = self.resnet2[i](h)
            h = self.attention_blocks[i](h, p)

            h += res / (2 ** 0.5)  # Gradient stabilization

        # Return h and skip connection
        return self.downsample(h), h


class BottleNeck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_groups: int,
                 t_embd: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float):
        """
        Bottleneck portion of the UNet
        I kept the number of channels the same. Can adjust it so that it doubles the number of channels for output

        :param in_channels: Channels of input image
        :param n_groups: Number of norm groups for groupnorm
        :param t_embd: Embedding dimension for timesteps
        :param n_embd: Embedding dimension for prompts tensors (Fixed at 512 for CLIP)
        :param n_heads: Number of attention heads
        :param n_layers: Number of sequential layers for this block
        :param dropout: Dropout prob
        """

        super().__init__()

        self.n_layers = n_layers

        self.resnet1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for _ in range(n_layers)
        ])

        self.t_embd_linear = nn.Linear(t_embd, in_channels)

        self.resnet2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for _ in range(n_layers)
        ])

        self.attention_blocks = nn.ModuleList([UnetAttentionBlock(channels=in_channels, n_groups=n_groups, n_heads=n_heads, n_embd=n_embd) for _ in range(n_layers)])


    def forward(self, x, p, t):
        h = x

        for i in range(self.n_layers):
            res = h  # residual connection
            h = self.resnet1[i](h)

            t_embd = self.t_embd_linear(t)  # Create timestep embedding

            h += t_embd.unsqueeze(-1).unsqueeze(-1)

            h += res / (2 ** 0.5)  # Add residual and stabilize gradients
            res = h  # "midpoint" residual

            h = self.resnet2[i](h)
            h = self.attention_blocks[i](h, p)

            h += res / (2 ** 0.5)  # Gradient stabilization

        return h



class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_groups: int,
                 t_embd: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float):
        """
        Decoder portion of the UNet

        :param in_channels: Channels of input image
        :param out_channels: Desired channels for output image
        :param n_groups: Number of norm groups for groupnorm
        :param t_embd: Embedding dimension for timesteps
        :param n_embd: Embedding dimension for prompts tensors (Fixed at 512 for CLIP)
        :param n_heads: Number of attention heads
        :param n_layers: Number of sequential layers for this block
        :param dropout: Dropout prob
        """

        super().__init__()
        self.n_layers = n_layers

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv_skip = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)

        self.resnet1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for i in range(n_layers)
        ])

        self.t_embd_linear = nn.Linear(t_embd, out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.resnet2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(n_groups, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=dropout),
            )
            for _ in range(n_layers)
        ])

        self.attention_blocks = nn.ModuleList([UnetAttentionBlock(channels=out_channels, n_groups=n_groups, n_heads=n_heads, n_embd=n_embd) for _ in range(n_layers)])


    def forward(self, x, skip, p, t):
        h = x
        h = self.upsample(h)
        h = torch.cat((h, skip), dim=1)  # Concat skip connection
        h = self.conv_skip(h)

        for i in range(self.n_layers):
            res = h  # residual connection
            h = self.resnet1[i](h)

            t_embd = self.t_embd_linear(t)  # Create timestep embedding
            h += t_embd.unsqueeze(-1).unsqueeze(-1)

            h += (self.residual_conv(res) if i == 0 else res) / (2 ** 0.5)  # Add residual and stabilize gradients
            res = h  # "midpoint" residual

            h = self.resnet2[i](h)
            h = self.attention_blocks[i](h, p)

            h += res / (2 ** 0.5)  # Gradient stabilization

        return h


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels: tuple,
                 n_groups: int,
                 T: int,
                 t_embd: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float):
        """
        The UNet model, core of the diffusion process

        :param in_channels: The value of input channels. 3 if working in pixel space, 4 if latent space (using VAE)
        :param channels: A tuples of channels (e.g. (64, 128, 256, 512))
        :param n_groups: Number of norm groups for groupnorm
        :param T: The maximum number of timesteps for this diffusion process. Usually 1000
        :param t_embd: Embedding dimension for timesteps
        :param n_embd: Embedding dimension for prompts tensors (Fixed at 512 for CLIP)
        :param n_heads: Number of attention heads
        :param n_layers: Number of sequential layers for each Encoder/BottleNeck/Decoder block
        :param dropout: Dropout prob
        """

        super().__init__()

        assert len(channels) == 4, f"Expected number of channels: 4, instead got {len(channels)=}"

        self.time_encoder = TimeEncoding(T, t_embd)

        self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList([
            Encoder(
                in_channels=channels[i],
                out_channels=channels[i+1],
                n_groups=n_groups,
                t_embd=t_embd,
                n_embd=n_embd,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            for i in range(len(channels)-1)
        ])


        self.bottlenecks = nn.ModuleList([
            BottleNeck(
                in_channels=channels[-1],
                n_groups=n_groups,
                t_embd=t_embd,
                n_embd=n_embd,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
        ])


        self.decoders = nn.ModuleList([
            Decoder(
                in_channels=channels[i],
                out_channels=channels[i-1],
                n_groups=n_groups,
                t_embd=t_embd,
                n_embd=n_embd,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            for i in range(len(channels)-1, 0, -1)
        ])


        self.final_layer = nn.Sequential(
            nn.GroupNorm(n_groups, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, p, t):
        """
        Given input image, prompt, and timestep tensors, returns output tensor of same shape representing predicted noise

        :param x: Input image/latent tensor of shape (b, c, h, w)
        :param p: Prompt tensor of shape (b, 77, 512)
        :param t: Timestep tensor of shape (b,)
        :return: Returns predicted noise in image
        """

        assert len(x.shape) == 4, f"Given tensor x should be of shape (b, c, h, w), instead got {x.shape}"
        assert list(p.shape) == [x.shape[0], 77, 512], f"Prompt tensor should be of shape ({x.shape[0]}, 77, 512) but got {p.shape=}"
        assert len(t) == x.shape[0], f"Expected num timesteps == batch_size, {len(t)=}, {x.shape[0]=}"
        assert len(t.shape) == 1, "timesteps t should be a vector of shape (b)"

        h = x
        ts = self.time_encoder(t)

        h = self.in_conv(h)

        skip_connections = []
        for enc in self.encoders:
            h, skip = enc(h, p, ts)
            skip_connections.append(skip)

        for bn in self.bottlenecks:
            h = bn(h, p, ts)

        for dec in self.decoders:
            h = dec(h, skip_connections.pop(), p, ts)

        return self.final_layer(h)


class DiffusionModel(nn.Module):
    def __init__(self, unet_params, diffusion_params):
        """
        Primary Diffusion Model

        :param unet_params: A dictionary containing all parameters for the UNet
        :param diffusion_params: A dictionary containing all the parameters for the Diffusion Model
        """

        super().__init__()
        self.T = unet_params["T"]
        self.beta1 = diffusion_params["beta1"]
        self.beta2 = diffusion_params["beta2"]

        # The SD implementation at https://github.com/CompVis/latent-diffusion in `ldm/modules/diffusionmodules/util.py`
        # file `make_beta_schedule` function actually uses this curved beta as opposed to a linear one
        self.betas = torch.linspace(self.beta1 ** 0.5, self.beta2 ** 0.5, self.T, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)


        self.unet = UNet(in_channels=unet_params["in_channels"],
                         channels=unet_params["channels"],
                         n_groups=unet_params["n_groups"],
                         T=unet_params["T"],
                         t_embd=unet_params["t_embd"],
                         n_embd=unet_params["n_embd"],
                         n_heads=unet_params["n_heads"],
                         n_layers=unet_params["n_layers"],
                         dropout=unet_params["dropout"],
                         )

    def _forward_process(self, clean_images: torch.Tensor, timesteps: torch.Tensor):
        assert len(clean_images.shape) == 4
        assert len(timesteps.shape) == 1
        assert clean_images.shape[0] == timesteps.shape[0]

        noise = torch.randn(clean_images.shape, device=clean_images.device)

        # Need to change these two to match along batch dimension
        sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1).to(clean_images.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1).to(clean_images.device)
        noised_images = sqrt_alpha_bar * clean_images + sqrt_one_minus_alpha_bar * noise

        return noised_images, noise

    def _reverse_process(self, noised_images: torch.Tensor, prompt_tensor: torch.Tensor, timesteps: torch.Tensor):
        assert len(noised_images.shape) == 4
        assert len(timesteps.shape) == 1
        assert noised_images.shape[0] == timesteps.shape[0]

        return self.unet(noised_images.repeat(2, 1, 1, 1), torch.cat([prompt_tensor, torch.zeros_like(prompt_tensor)], dim=0), timesteps.repeat(2))

    def forward(self, clean_images: torch.Tensor, prompt_tensor: torch.Tensor, timesteps: torch.Tensor, cfg_scale: float):
        noised_images, true_noise = self._forward_process(clean_images, timesteps)

        pred_noise = self._reverse_process(noised_images, prompt_tensor, timesteps)
        conditional_output, unconditional_output = pred_noise.chunk(2, dim=0)
        pred_noise = cfg_scale * (conditional_output - unconditional_output) + unconditional_output


        return pred_noise, true_noise

    @torch.no_grad()
    def generate(self, vae: AutoencoderKL, prompt_tensor: torch.Tensor, num_images: int, img_dim: int, num_steps: int, cfg_scale: float, device: str):
        self.eval()

        # Start with complete gaussian noise
        latents = torch.randn((num_images, 4, img_dim // 8, img_dim // 8), device=device) * 0.18215

        # Denoising loop
        for t in reversed(torch.linspace(0, self.T - 1, num_steps)):
            print(t)
            t = t.to(torch.int)

            # Create timestep tensor
            timesteps = torch.full((num_images,), t, device=device, dtype=torch.long)

            # Predict the noise
            predicted_noise = self._reverse_process(latents, prompt_tensor, timesteps)
            conditional_output, unconditional_output = predicted_noise.chunk(2, dim=0)
            predicted_noise = cfg_scale * (conditional_output - unconditional_output) + unconditional_output

            # Compute the mean of the reverse process
            alpha_t = self.alphas[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

            mean = (latents - (self.betas[t] / sqrt_one_minus_alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)

            # Sample the previous timestep
            if t > 0:
                noise = torch.randn_like(latents)
                sigma_t = torch.sqrt(self.betas[t])
                latents = mean + sigma_t * noise
            else:
                latents = mean  # No noise for the last step


        generated_images = vae.decode(latents / 0.18215).sample  # Decode from latent space
        # Rescale images to [0, 1] range
        generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0

        self.train()
        return generated_images
