import functools
import torch
import torch.nn as nn

import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.core import ConvAlgo

from collections import OrderedDict
from torch_scatter import scatter_mean

from model.SFPNet import ResFBlock


class ResidualBlock(SparseModule):
    """
        ResidualBlock with SubMConv3d
    """
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = (pair_in != -1)
    valid_pair_in, valid_pair_out = pair_in[valid_mask].long(), pair_out[valid_mask].long()
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next

class Cosine_aug(nn.Module):
    """
        Frequence augmentation (optional)
    """
    def __init__(self, in_dim = 3, out_dim = 6, alpha = 10000, beta = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        cur_dev = xyz.get_device()
        N, _ = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().to(cur_dev)
        dim_embed = torch.pow(self.beta, feat_range / feat_dim)
        div_embed = torch.div(self.alpha * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=3).flatten(2)
        position_embed = position_embed.reshape(N, self.out_dim)
        return position_embed
    
class UBlock(nn.Module):
    """
        Unet
    """
    def __init__(self, nPlanes,
        focal_r,
        focal_th,
        focal_h,
        norm_fn, 
        block_reps, 
        block,
        drop_path=0.0,
        indice_key_id=1, 
        grad_checkpoint_layers=[], 
        unet_layers=[1,2,3,4,5],
        ctx_mode=0,
    ):

        super().__init__()
        focal_level = 3

        self.ctx_mode = ctx_mode
        self.nPlanes = nPlanes
        self.indice_key_id = indice_key_id
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.unet_layers = unet_layers

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)


        if indice_key_id in unet_layers:
            self.focalblk = ResFBlock(nPlanes[0], drop=0., drop_path=drop_path[0], focal_level=focal_level,
                                      focal_x=focal_r[0],
                                      focal_y=focal_th[0], focal_z=focal_h[0],
                                      indice_key='focal{}'.format(indice_key_id))

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )
            self.u = UBlock(nPlanes[1:],
                            focal_r[1:],
                            focal_th[1:],
                            focal_h[1:],
                            norm_fn,
                            block_reps,
                            block,
                            drop_path=drop_path[1:],
                            indice_key_id=indice_key_id + 1,
                            grad_checkpoint_layers=grad_checkpoint_layers,
                            unet_layers=unet_layers,
                            ctx_mode=0,
                            )


            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id), algo=ConvAlgo.Native)
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)


    def forward(self, inp, xyz, batch):

        assert (inp.indices[:, 0] == batch).all()
        
        output = self.blocks(inp)

        # core
        if self.indice_key_id in self.unet_layers:
            output = self.focalblk(output)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)

            indice_pairs = output_decoder.indice_dict['spconv{}'.format(self.indice_key_id)].indice_pairs
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)

            output_decoder = self.u(output_decoder, xyz_next, batch_next.long())
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        return output


class Semantic(nn.Module):
    """
        Semantic segmentation Network
    """
    def __init__(self, 
        input_c, 
        m, 
        classes, 
        block_reps,
        layers,
        focal_r,
        focal_th,
        focal_h,
        drop_path_rate=0.0,
        grad_checkpoint_layers=[], 
        unet_layers=[1,2,3,4,5],
    ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block = ResidualBlock
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        # input
        self.pose_ini = Cosine_aug(3, 6, 10000, 1)
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c+6, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        # unet
        self.unet = UBlock(layers,
            focal_r,
            focal_th,
            focal_h,
            norm_fn,
            block_reps,
            block,
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers,
            unet_layers=unet_layers,
        )

        # output
        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        self.apply(self.set_bn_init)
        # semantic segmentation
        self.linear = nn.Linear(m, classes)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, xyz, batch):
        ret_new_cos = []
        for i in range(len(torch.unique(batch))):
            xyz_c_b = xyz[batch == i]
            ret_new_cos.append(self.pose_ini(xyz_c_b))

        ret_cos = torch.cat(ret_new_cos, dim=0)
        input = input.replace_feature(torch.cat([input.features, ret_cos], dim=1))

        output = self.input_conv(input)
        output = self.unet(output, xyz, batch)
        output = self.output_layer(output)

        # semantic segmentation
        semantic_scores = self.linear(output.features)

        return semantic_scores

