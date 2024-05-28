import os
from contextlib import contextmanager
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import pytorch_lightning as pl
from lvdm.modules.networks.ae_modules import Encoder, Decoder
from lvdm.distributions import DiagonalGaussianDistribution
from utils.utils import instantiate_from_config

TIMESTEPS=16
class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 test=False,
                 logdir=None,
                 input_dim=4,
                 test_args=None,
                 additional_decode_keys=None,
                 use_checkpoint=False,
                 diff_boost_factor=3.0,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.test = test
        self.test_args = test_args
        self.logdir = logdir
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if self.test:
            self.init_test()
    
    def init_test(self,):
        self.test = True
        save_dir = os.path.join(self.logdir, "test")
        if 'ckpt' in self.test_args:
            ckpt_name = os.path.basename(self.test_args.ckpt).split('.ckpt')[0] + f'_epoch{self._cur_epoch}'
            self.root = os.path.join(save_dir, ckpt_name)
        else:
            self.root = save_dir
        if 'test_subdir' in self.test_args:
            self.root = os.path.join(save_dir, self.test_args.test_subdir)

        self.root_zs = os.path.join(self.root, "zs")
        self.root_dec = os.path.join(self.root, "reconstructions")
        self.root_inputs = os.path.join(self.root, "inputs")
        os.makedirs(self.root, exist_ok=True)

        if self.test_args.save_z:
            os.makedirs(self.root_zs, exist_ok=True)
        if self.test_args.save_reconstruction:
            os.makedirs(self.root_dec, exist_ok=True)
        if self.test_args.save_input:
            os.makedirs(self.root_inputs, exist_ok=True)
        assert(self.test_args is not None)
        self.test_maximum = getattr(self.test_args, 'test_maximum', None) 
        self.count = 0
        self.eval_metrics = {}
        self.decodes = []
        self.save_decode_samples = 2048

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        try:
            self._cur_epoch = sd['epoch']
            sd = sd["state_dict"]
        except:
            self._cur_epoch = 'null'
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        # self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x, return_hidden_states=False, **kwargs):
        if return_hidden_states:
            h, hidden = self.encoder(x, return_hidden_states)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, hidden
        else:
            h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior

    def decode(self, z, **kwargs):
        if len(kwargs) == 0: ## use the original decoder in AutoencoderKL
            z = self.post_quant_conv(z)
        dec = self.decoder(z, **kwargs)  ##change for SVD decoder by adding **kwargs
        return dec

    def forward(self, input, sample_posterior=True, **additional_decode_kwargs):
        input_tuple = (input, )
        forward_temp = partial(self._forward, sample_posterior=sample_posterior, **additional_decode_kwargs)
        return checkpoint(forward_temp, input_tuple, self.parameters(), self.use_checkpoint)
        

    def _forward(self, input, sample_posterior=True, **additional_decode_kwargs):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, **additional_decode_kwargs)
        ## print(input.shape, dec.shape) torch.Size([16, 3, 256, 256]) torch.Size([16, 3, 256, 256])
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if x.dim() == 5 and self.input_dim == 4:
            b,c,t,h,w = x.shape
            self.b = b
            self.t = t 
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

from lvdm.models.autoencoder_dualref import VideoDecoder
class AutoencoderKL_Dualref(AutoencoderKL):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 test=False,
                 logdir=None,
                 input_dim=4,
                 test_args=None,
                 additional_decode_keys=None,
                 use_checkpoint=False,
                 diff_boost_factor=3.0,
                 ):
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels, monitor, test, logdir, input_dim, test_args, additional_decode_keys, use_checkpoint, diff_boost_factor)
        self.decoder = VideoDecoder(**ddconfig)

    def _forward(self, input, sample_posterior=True, **additional_decode_kwargs):
        posterior, hidden_states = self.encode(input, return_hidden_states=True)

        hidden_states_first_last = []
        ### use only the first and last hidden states
        for hid in hidden_states:
            hid = rearrange(hid, '(b t) c h w -> b c t h w', t=TIMESTEPS)
            hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
            hidden_states_first_last.append(hid_new)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, ref_context=hidden_states_first_last, **additional_decode_kwargs)
        ## print(input.shape, dec.shape) torch.Size([16, 3, 256, 256]) torch.Size([16, 3, 256, 256])
        return dec, posterior