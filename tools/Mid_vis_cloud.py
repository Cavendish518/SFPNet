# Frame visualization tools for S.MID
import open3d as o3d
import numpy as np
import argparse

def viscloud(point, label, color_map, windowname):
    clouds = point[:, :3]
    colors = np.zeros_like(clouds)

    for l, c in color_map.items():
        colors[label.squeeze() == l] = np.array(c)
    colors = colors / 255.0

    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(clouds)
    test_pcd.colors = o3d.utility.Vector3dVector(colors)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([test_pcd, coord_frame], window_name=windowname, zoom=0.20, front=[-5, -1.0, 3], lookat=[3, -2.0, 2.0], up=[0, 0, 1])
    return

def viscloud_err(point, label, windowname):
    clouds = point[:, :3]
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(clouds)

    colors = np.zeros_like(clouds)
    red_id = np.where(label == 0)[0]
    colors[red_id] = [1, 0, 0]
    test_pcd.colors = o3d.utility.Vector3dVector(colors)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([test_pcd, coord_frame], window_name=windowname, zoom=0.20, front=[-5, -1, 3], lookat=[3, -2.0, 2.0], up=[0, 0, 1])
    return

def bin_err(label_p, labelgt):
    label_err1 = label_p.copy()
    label_err1[label_p != labelgt] = 0
    label_err1[label_p == labelgt] = 1
    label_err1[labelgt == 0] = 1
    return label_err1

def main():
    parser = argparse.ArgumentParser(description="Frame visualization of S.MID")
    parser.add_argument("--vis_type", type=str, required=True, choices=["ground_truth", "prediction", "difference"], help="Type of visualization to display")
    parser.add_argument("--pt_path", type=str, help="Path to the bin file")
    parser.add_argument("--gt_path", type=str, help="Path to the ground truth label file")
    parser.add_argument("--pd_path", type=str, help="Path to the prediction label file")
    args = parser.parse_args()

    map = {
        0: 0, 1: 0, 101: 1, 102: 2, 103: 3, 104: 3, 105: 1, 106: 4, 107: 5, 108: 0,
        109: 6, 110: 7, 111: 8, 112: 9, 116: 10, 117: 11, 118: 0, 119: 12, 120: 13,
        121: 14, 122: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 3
    }

    color_map14 = {
        0: [0, 0, 0], 1: [100, 150, 245], 2: [100, 230, 245], 3: [30, 60, 150], 4: [80, 30, 180],
        5: [255, 30, 30], 6: [150, 30, 90], 7: [255, 0, 255], 8: [255, 150, 255], 9: [75, 0, 75],
        10: [255, 150, 0], 11: [0, 100, 80], 12: [0, 175, 0], 13: [135, 60, 0], 14: [150, 240, 80]
    }

    pointm = np.fromfile(args.pt_path, dtype=np.float32).reshape(-1, 4)

    if args.vis_type == "ground_truth":
        label_gt = np.fromfile(args.gt_path, dtype=np.uint32).reshape(-1, 1)
        label_gt = label_gt & 0xFFFF
        for key, value in map.items():
            label_gt[label_gt == key] = value
        viscloud(pointm, label_gt, color_map14, windowname="ground_truth")
    elif args.vis_type == "prediction":
        label_p = np.fromfile(args.pd_path, dtype=np.uint32).reshape(-1, 1)  # you can modify here to fit your storage
        label_p = label_p & 0xFFFF  # you can modify here to fit your storage
        for key, value in map.items():
            label_p[label_p == key] = value # you can modify here to fit your storage
        viscloud(pointm, label_p, color_map14, windowname="prediction")
    elif args.vis_type == "difference":
        label_gt = np.fromfile(args.gt_path, dtype=np.uint32).reshape(-1, 1)
        label_gt = label_gt & 0xFFFF
        for key, value in map.items():
            label_gt[label_gt == key] = value
        label_p = np.fromfile(args.pd_path, dtype=np.uint32).reshape(-1, 1)  # you can modify here to fit your storage
        label_p = label_p & 0xFFFF  # you can modify here to fit your storage
        for key, value in map.items():
            label_p[label_p == key] = value # you can modify here to fit your storage
        label_err = bin_err(label_p, label_gt)
        viscloud_err(pointm, label_err, windowname="difference")

if __name__ == "__main__":
    main()