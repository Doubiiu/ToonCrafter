from functools import partial
from abc import abstractmethod
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.common import checkpoint
from lvdm.basics import (
    zero_module,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization
)
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


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
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
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

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware
            )

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple, self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb, batch_size=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TemporalConvBlock(nn.Module):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return identity + x

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 context_dim=None,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 transformer_depth=1,
                 use_linear=False,
                 use_checkpoint=False,
                 temporal_conv=False,
                 tempspatial_aware=False,
                 temporal_attention=True,
                 use_relative_position=True,
                 use_causal_attention=False,
                 temporal_length=None,
                 use_fp16=False,
                 addition_attention=False,
                 temporal_selfatt_only=True,
                 image_cross_attention=False,
                 image_cross_attention_scale_learnable=False,
                 default_fs=4,
                 fs_condition=False,
                ):
        super(UNetModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition

        ## Time embedding blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        ## Input Block
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )
        if self.addition_attention:
            self.init_attn=TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only, 
                    causal_attention=False, relative_position=use_relative_position, 
                    temporal_length=temporal_length))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, 
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False, 
                            video_length=temporal_length, image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,                      
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                                causal_attention=use_causal_attention, relative_position=use_relative_position, 
                                temporal_length=temporal_length
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, 
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv
            ),
            SpatialTransformer(ch, num_heads, dim_head, 
                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length, 
                image_cross_attention=self.image_cross_attention,image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable                
            )
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(ch, num_heads, dim_head,
                    depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                    causal_attention=use_causal_attention, relative_position=use_relative_position, 
                    temporal_length=temporal_length
                )
            )
        layers.append(
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware, 
                use_temporal_conv=temporal_conv
                )
        )

        ## Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        ## Output Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, 
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable    
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                                causal_attention=use_causal_attention, relative_position=use_relative_position, 
                                temporal_length=temporal_length
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        b,_,t,_,_ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        
        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t*16: ## !!! HARD CODE here
            context_text, context_img = context[:,:77,:], context[:,77:,:]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)
        
        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id ==0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            ## plug-in adapter features
            if ((id+1)%3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        h = self.middle_block(h, emb, context=context, batch_size=b)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)
        h = h.type(x.dtype)
        y = self.out(h)
        
        # reshape back to (b c t h w)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        return y