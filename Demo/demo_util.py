import torch
import numpy as np
from torch_scatter import scatter_mean
from util.voxelize import voxelize

def collation_fn_voxelmean_tta(batch_list):

    samples = []
    batch_list = list(zip(*batch_list))

    for batch in batch_list:
        coords, xyz, feats, labels, inds_recons = list(zip(*batch))
        inds_recons = list(inds_recons)

        accmulate_points_num = 0
        offset = []

        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)

        coords = torch.cat(coords)
        xyz = torch.cat(xyz)
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_recons = torch.cat(inds_recons)

        sample = (coords, xyz, feats, labels, offset, inds_recons)
        samples.append(sample)

    return samples

def data_prepare(coord, feat, label, split='train', voxel_size=np.array([0.1, 0.1, 0.1]), voxel_max=None, transform=None, xyz_norm=False):
    if transform:
        coord, feat = transform(coord, feat)
    coord_min = np.min(coord, 0)
    coord_norm = coord - coord_min
    if split == 'train':
        uniq_idx,id_back = voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))

        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        if voxel_max and label.shape[0] > voxel_max:
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
        idx_recon = voxelize(coord_norm, voxel_size, mode=1)

    if xyz_norm:
        coord_min = np.min(coord, 0)
        coord -= coord_min

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)

    if split == 'train':
        coord_voxel = torch.LongTensor(coord_voxel)
        return coord_voxel, coord, feat, label
    else:
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coords_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coords_voxel, coord, feat, label, idx_recon
