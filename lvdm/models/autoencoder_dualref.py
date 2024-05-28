#### https://github.com/Stability-AI/generative-models
from einops import rearrange, repeat
import logging
from typing import Any, Callable, Optional, Iterable, Union

import numpy as np
import torch
import torch.nn as nn
from packaging import version
logpy = logging.getLogger(__name__)

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warning("no module 'xformers'. Processing without...")

from lvdm.modules.attention_svd import LinearAttention, MemoryEfficientCrossAttention


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        )
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
        "memory-efficient-cross-attn-fusion",
    ], f"attn_type {attn_type} unknown"
    if (
        version.parse(torch.__version__) < version.parse("2.0.0")
        and attn_type != "none"
    ):
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        # attn_type = "vanilla-xformers"
    logpy.info(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        logpy.info(
            f"building MemoryEfficientAttnBlock with {in_channels} in_channels..."
        )
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "memory-efficient-cross-attn-fusion":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapperFusion(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)

class MemoryEfficientCrossAttentionWrapperFusion(MemoryEfficientCrossAttention):
    # print('x.shape: ',x.shape, 'context.shape: ',context.shape) ##torch.Size([8, 128, 256, 256]) torch.Size([1, 128, 2, 256, 256])
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0, **kwargs):
        super().__init__(query_dim, context_dim, heads, dim_head, dropout, **kwargs)
        self.norm = Normalize(query_dim)
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x, context=None, mask=None):
        if self.training:
            return checkpoint(self._forward, x, context, mask, use_reentrant=False)
        else:
            return self._forward(x, context, mask)

    def _forward(
        self,
        x,
        context=None,
        mask=None,
    ):
        bt, c, h, w = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, "b c h w -> b (h w) c")
        q = self.to_q(h_)


        b, c, l, h, w = context.shape
        context = rearrange(context, "b c l h w -> (b l) (h w) c")
        k = self.to_k(context)
        v = self.to_v(context)
        k = rearrange(k, "(b l) d c -> b l d c", l=l)
        k = torch.cat([k[:, [0] * (bt//b)], k[:, [1]*(bt//b)]], dim=2)
        k = rearrange(k, "b l d c -> (b l) d c")

        v = rearrange(v, "(b l) d c -> b l d c", l=l)
        v = torch.cat([v[:, [0] * (bt//b)], v[:, [1]*(bt//b)]], dim=2)
        v = rearrange(v, "b l d c -> (b l) d c")


        b, _, _ = q.shape  ##actually bt
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        out = self.to_out(out)
        out = rearrange(out, "bt (h w) c -> bt c h w", h=h, w=w, c=c)
        return x + out 

class Combiner(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch,ch,1,padding=0)

        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, context):
        if self.training:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)
    
    def _forward(self, x, context):
        ## x: b c h w, context: b c 2 h w
        b, c, l, h, w = context.shape
        bt, c, h, w = x.shape
        context = rearrange(context, "b c l h w -> (b l) c h w")
        context = self.conv(context)
        context = rearrange(context, "(b l) c h w -> b c l h w", l=l)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=bt//b)
        x[:,:,0] = x[:,:,0] + context[:,:,0]
        x[:,:,-1] = x[:,:,-1] + context[:,:,1]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla-xformers",
        attn_level=[2,3], 
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.attn_level = attn_level
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logpy.info(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        self.attn_refinement = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

            if i_level in self.attn_level:
                self.attn_refinement.insert(0, make_attn_cls(block_in, attn_type='memory-efficient-cross-attn-fusion', attn_kwargs={}))
            else:
                self.attn_refinement.insert(0, Combiner(block_in))
        # end
        self.norm_out = Normalize(block_in)
        self.attn_refinement.append(Combiner(block_in))
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetBlock

    def _make_conv(self) -> Callable:
        return torch.nn.Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, ref_context=None, **kwargs):
        ## ref_context: b c 2 h w, 2 means starting and ending frame
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if ref_context:
                h = self.attn_refinement[i_level](x=h, context=ref_context[i_level])
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        if ref_context:
            # print(h.shape, ref_context[i_level].shape) #torch.Size([8, 128, 256, 256]) torch.Size([1, 128, 2, 256, 256])
            h = self.attn_refinement[-1](x=h, context=ref_context[-1])
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

#####


from abc import abstractmethod
from lvdm.models.utils_diffusion import timestep_embedding

from torch.utils.checkpoint import checkpoint
from lvdm.basics import (
    zero_module,
    conv_nd,
    linear,
    normalization,
)
from lvdm.modules.networks.openaimodel3d import Upsample, Downsample
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            # print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        else:
            return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
#####

#####
from lvdm.modules.attention_svd import *
class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"====>{self.__class__.__name__} is using checkpointing")
        else:
            print(f"====>{self.__class__.__name__} is NOT using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps, use_reentrant=False)
        else:
            return self._forward(x, context, timesteps=timesteps)

    def _forward(self, x, context=None, timesteps=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x)) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x)) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight

#####

#####
import functools
def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls
######

class VideoResBlock(ResnetBlock):
    def __init__(
        self,
        out_channels,
        *args,
        dropout=0.0,
        video_kernel_size=3,
        alpha=0.0,
        merge_strategy="learned",
        **kwargs,
    ):
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = ResBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=True,
            skip_t_emb=True,
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, bs):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError()

    def forward(self, x, temb, skip_video=False, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps

        b, c, h, w = x.shape

        x = super().forward(x, temb)

        if not skip_video:
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = self.time_stack(x, temb)

            alpha = self.get_alpha(bs=b // timesteps)
            x = alpha * x + (1.0 - alpha) * x_mix

            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class AE3DConv(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, Iterable):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)

        self.time_mix_conv = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding,
        )

    def forward(self, input, timesteps, skip_video=False):
        x = super().forward(input)
        if skip_video:
            return x
        x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        x = self.time_mix_conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class VideoBlock(AttnBlock):
    def __init__(
        self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"
    ):
        super().__init__(in_channels)
        # no context, single headed, as in base class
        self.time_mix_block = VideoTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=True,
            ff_in=True,
            attn_mode="softmax",
        )

        time_embed_dim = self.in_channels * 4
        self.video_time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, self.in_channels),
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def forward(self, x, timesteps, skip_video=False):
        if skip_video:
            return super().forward(x)

        x_in = x
        x = self.attention(x)
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")

        x_mix = x
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)  # b, n_channels
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        alpha = self.get_alpha()
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        return x_in + x

    def get_alpha(
        self,
    ):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


class MemoryEfficientVideoBlock(MemoryEfficientAttnBlock):
    def __init__(
        self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"
    ):
        super().__init__(in_channels)
        # no context, single headed, as in base class
        self.time_mix_block = VideoTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=True,
            ff_in=True,
            attn_mode="softmax-xformers",
        )

        time_embed_dim = self.in_channels * 4
        self.video_time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, self.in_channels),
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def forward(self, x, timesteps, skip_time_block=False):
        if skip_time_block:
            return super().forward(x)

        x_in = x
        x = self.attention(x)
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")

        x_mix = x
        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.video_time_embed(t_emb)  # b, n_channels
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        alpha = self.get_alpha()
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        return x_in + x

    def get_alpha(
        self,
    ):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


def make_time_attn(
    in_channels,
    attn_type="vanilla",
    attn_kwargs=None,
    alpha: float = 0,
    merge_strategy: str = "learned",
):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
    ], f"attn_type {attn_type} not supported for spatio-temporal attention"
    print(
        f"making spatial and temporal attention of type '{attn_type}' with {in_channels} in_channels"
    )
    if not XFORMERS_IS_AVAILABLE and attn_type == "vanilla-xformers":
        print(
            f"Attention mode '{attn_type}' is not available. Falling back to vanilla attention. "
            f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
        )
        attn_type = "vanilla"

    if attn_type == "vanilla":
        assert attn_kwargs is None
        return partialclass(
            VideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy
        )
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return partialclass(
            MemoryEfficientVideoBlock,
            in_channels,
            alpha=alpha,
            merge_strategy=merge_strategy,
        )
    else:
        return NotImplementedError()


class Conv2DWrapper(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


class VideoDecoder(Decoder):
    available_time_modes = ["all", "conv-only", "attn-only"]

    def __init__(
        self,
        *args,
        video_kernel_size: Union[int, list] = [3,1,1],
        alpha: float = 0.0,
        merge_strategy: str = "learned",
        time_mode: str = "conv-only",
        **kwargs,
    ):
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.time_mode = time_mode
        assert (
            self.time_mode in self.available_time_modes
        ), f"time_mode parameter has to be in {self.available_time_modes}"
        super().__init__(*args, **kwargs)

    def get_last_layer(self, skip_time_mix=False, **kwargs):
        if self.time_mode == "attn-only":
            raise NotImplementedError("TODO")
        else:
            return (
                self.conv_out.time_mix_conv.weight
                if not skip_time_mix
                else self.conv_out.weight
            )

    def _make_attn(self) -> Callable:
        if self.time_mode not in ["conv-only", "only-last-conv"]:
            return partialclass(
                make_time_attn,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy,
            )
        else:
            return super()._make_attn()

    def _make_conv(self) -> Callable:
        if self.time_mode != "attn-only":
            return partialclass(AE3DConv, video_kernel_size=self.video_kernel_size)
        else:
            return Conv2DWrapper

    def _make_resblock(self) -> Callable:
        if self.time_mode not in ["attn-only", "only-last-conv"]:
            return partialclass(
                VideoResBlock,
                video_kernel_size=self.video_kernel_size,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy,
            )
        else:
            return super()._make_resblock()