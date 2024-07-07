# Demo codes for core module SFPM Block (Class ResFBlock).
# SFPM block can replace transformer block in the mainstream backbones

import torch
from torch import nn
import spconv.pytorch as spconv
from timm.models.layers import DropPath


class Mlp(nn.Module):
    """ 
        Feedforward neural network by MLP
    """

    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Importance3D(nn.Module):
    '''
        Demo implementation of multi-level context extraction by SubMConv3d
    '''

    def __init__(self, dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1, indice_key=None, bias=True):
        super().__init__()

        self.ctx = spconv.SubMConv3d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=groups, indice_key=indice_key, bias=bias)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ctx(x)
        x = x.replace_feature(self.act(x.features))
        x = x.replace_feature(self.ln(x.features))
        return x


class SparseFocalModulation(nn.Module):
    """ Spase Focal Point Modulation
    Args:
        dim (int): Number of input channels.
        focal_level (int, default=3): Number of focal levels
        focal_x, focal_y, focal_z (default=[3, 1, 1, 1] )
        focal_factor (int, default=2): Step to increase the focal window
    """

    def __init__(self, dim, proj_drop=0., focal_level=3, focal_x=[3, 1, 1, 1], focal_y=[3, 1, 1, 1],
                 focal_z=[3, 1, 1, 1], focal_factor=2, indice_key=None):

        super().__init__()
        self.dim = dim
        self.group = 1
        self.focal_level = focal_level
        self.focal_x = focal_x[0]
        self.focal_y = focal_y[0]
        self.focal_z = focal_z[0]
        self.focal_factor = focal_factor
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = spconv.SubMConv3d(dim, dim, kernel_size=1, stride=1,
                                   padding=0, groups=1, indice_key=indice_key + "1*1", bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        for k in range(self.focal_level):
            kernel_size = (self.focal_factor * k + self.focal_x, self.focal_factor * k + self.focal_y,
                           self.focal_factor * k + self.focal_z)
            dilation_size = (focal_x[k + 1], focal_y[k + 1], focal_z[k + 1])
            padding_size = ((kernel_size[0] + (kernel_size[0] - 1) * (dilation_size[0] - 1)) // 2,
                            (kernel_size[1] + (kernel_size[1] - 1) * (dilation_size[1] - 1)) // 2,
                            (kernel_size[2] + (kernel_size[2] - 1) * (dilation_size[2] - 1)) // 2)
            level_k = "level" + str(k + 1)
            self.focal_layers.append(
                Importance3D(dim, kernel_size=kernel_size, stride=1, padding=padding_size, groups=self.group,
                             indice_key=indice_key + level_k, bias=True),
            )

    def Avgpoolgate(self, ctx, gate):
        id = ctx.indices[:, 0]
        pooled_ctx = []
        for batch_idx in torch.unique(id):
            cur_id = (id == batch_idx).nonzero(as_tuple=False).squeeze(dim=1)
            cur_ctx = ctx.features[cur_id]
            cur_ctx = self.act(cur_ctx.mean(0, keepdim=True))
            pooled_ctx.append(cur_ctx * gate[cur_id])
        out_ctx = torch.cat(pooled_ctx, dim=0)
        return out_ctx

    def forward(self, x):
        """ 
            x: sparsetensor
        """
        C = x.features.shape[1]
        x_fea = self.f(x.features)
        # split for q, ctx, and gates
        q, ctx_fea, gates = torch.split(x_fea, (C, C, self.focal_level + 1), 1)
        ctx = x.replace_feature(ctx_fea)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = (self.focal_layers[l](ctx))  # equation (4)
            ctx_all = ctx_all + ctx.features * gates[:, l:l + 1]

        ctx_global = self.Avgpoolgate(ctx, gates[:, self.focal_level:])
        ctx_all = ctx_all + ctx_global

        ctx_all = ctx.replace_feature(ctx_all)
        ctx_all = self.h(ctx_all)  # equation (6)

        x_out = q * ctx_all.features

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        out = ctx_all.replace_feature(x_out)
        return out


class ResFBlock(nn.Module):
    """ 
        Spase Focal Point Modulation Block.
    Args:
        dim (int): Number of input channels.
        focal_level (int, default=3): Number of focal levels
        focal_x, focal_y, focal_z (default=[3, 1, 1, 1] )
        focal_factor (int, default=2): Step to increase the focal window
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=3, focal_x=[3, 1, 1, 1], focal_y=[3, 1, 1, 1], focal_z=[3, 1, 1, 1], indice_key=None):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.focal_z = focal_z
        self.focal_level = focal_level
        self.norm1 = norm_layer(dim)

        self.modulation = SparseFocalModulation(dim=self.dim, proj_drop=drop, focal_level=self.focal_level,
                                                focal_x=self.focal_x,
                                                focal_y=self.focal_y, focal_z=self.focal_z, indice_key=indice_key)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, sparsetensor
        """
        shortcut = x.features
        x = x.replace_feature(self.norm1(x.features))
        x = self.modulation(x)
        focal_x = shortcut + self.drop_path(x.features)
        focal_x = focal_x + self.drop_path(self.mlp(self.norm2(focal_x)))
        x = x.replace_feature(focal_x)
        return x

