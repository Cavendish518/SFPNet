#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import torch
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import struct
import spconv.pytorch as spconv

# change to your files
from semutil.data_util import data_prepare
from semutil.backbone import Semantic as Model


# dummy rosnode (test with rslidar16 and ros1)

# =========================
# parameter
# =========================
VOXEL_SIZE = np.array([0.1, 0.1, 0.2], dtype=np.float32)
VOXEL_MAX = None
XYZ_NORM = False

# =========================
# PointCloud2 → numpy
# =========================
def pointcloud2_to_xyzi(msg):
    """
    Args:
        msg: sensor_msgs/PointCloud2
    Return:
        points_np: (N, 4) float32 [x, y, z, intensity]
    """
    points = []
    for p in pc2.read_points(
        msg,
        field_names=("x", "y", "z", "intensity"),
        skip_nans=True
    ):
        points.append(p)
    if len(points) == 0:
        return None
    return np.asarray(points, dtype=np.float32)

# =========================
# preprocess（depend on your Dataset）
# =========================
def preprocess_points(points_np,ROI = False):
    """
    Dataset.get_single_sample(test)
    """
    # ===== NaN / Inf =====
    mask = np.isfinite(points_np).all(axis=1)
    points_np = points_np[mask]

    # ===== if you want（ROI）=====
    if ROI:
        x_min, x_max = -20.0, 20.0
        y_min, y_max = -20.0, 20.0
        z_min, z_max = -2.0, 6.0

        xyz = points_np[:, :3]
        roi_mask = (
            (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) &
            (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) &
            (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
        )
        points_np = points_np[roi_mask]

    #todo: filter robots points (change it)
    xyz = points_np[:, :3]
    exclude_mask = ~(  (xyz[:, 0] <= 0.1) & (xyz[:, 0] >= -1.0) &
            (xyz[:, 1] <= 0.1) & (xyz[:, 1] >= -0.70) )   
    points_np = points_np[exclude_mask]

    xyz = points_np[:, :3]
    feats = points_np

    coords, xyz, feats, inds_reconstruct = data_prepare(
        xyz,
        feats,
        voxel_size=VOXEL_SIZE,
        xyz_norm=XYZ_NORM
    )

    return coords, xyz, feats, inds_reconstruct

# =========================
# ROS Node
# =========================
class SemanticInferNode:

    def __init__(self):

        rospy.loginfo("[SemanticInferNode] Initializing...")

        # ====== load model ======
        self.model = Model(input_c=4,
        m=32,
        classes=1, # change here
        block_reps=2,
        layers=[32, 64, 128, 256, 256],
        focal_r=[ [ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ] ],
        focal_th=[ [ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ] ],
        focal_h=[ [ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ],[ 3, 1, 1, 1 ] ],
        drop_path_rate=0.3,
        unet_layers=[1,2,3,4,5])

        checkpoint = torch.load("/home/your_path/model_clean.pth", map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to("cuda:0")
        self.model.eval()

        rospy.Subscriber(
            "/rslidar_points",
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )

        self.pub_semantic = rospy.Publisher(
            "/semantic_points",
            PointCloud2,
            queue_size=1
        )

        rospy.loginfo("[SemanticInferNode] Ready.")

    @torch.no_grad()
    def pointcloud_callback(self, msg):
        # =========================
        # 1. PointCloud2 → numpy
        # =========================
        points_np = pointcloud2_to_xyzi(msg)
        if points_np is None:
            rospy.logwarn("Empty point cloud")
            return

        # =========================
        # 2. preprocess
        # =========================
        coords, xyz, feats, inds_reconstruct = preprocess_points(points_np)
        inds_reconstruct.to("cuda:0")


        num_voxels = coords.shape[0]
        batch = torch.zeros(num_voxels, dtype=torch.long, device=coords.device)

        coords = torch.cat([batch.unsqueeze(-1), coords], dim=-1)

        spatial_shape = np.clip(
            (coords.max(0)[0][1:] + 1).numpy(),
            128,
            None
        )

        coords = coords.to("cuda:0")
        feats = feats.to("cuda:0")
        xyz = xyz.to("cuda:0")
        batch = batch.to("cuda:0")


        sinput = spconv.SparseConvTensor(feats, coords.int(), spatial_shape, 1)


        # =========================
        # inference
        # =========================
        output = self.model(sinput, xyz, batch)

        # =========================
        # voxel → point
        # =========================
        # output = output[inds_reconstruct, :] # if you need
        output = output.max(1)[1] + 1

        # =========================
        #（xyz + label）
        # =========================

        xyz_np = xyz.cpu().numpy()            # (N, 3)
        label_np = output.cpu().numpy()       # (N,)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
            PointField('label', 12, PointField.UINT16, 1),
        ]

        cloud_data = []
        for i in range(xyz_np.shape[0]):
            cloud_data.append((
                xyz_np[i, 0],
                xyz_np[i, 1],
                xyz_np[i, 2],
                int(label_np[i])
            ))

        semantic_cloud = pc2.create_cloud(
            header,
            fields,
            cloud_data
        )

        self.pub_semantic.publish(semantic_cloud)


# =========================
# main
# =========================
if __name__ == "__main__":
    rospy.init_node("semantic_infer_node", anonymous=False)
    node = SemanticInferNode()
    rospy.spin()

