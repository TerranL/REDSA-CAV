import argparse

import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d.datasets.utils import BEVBox3D

from classification import *
from shadow import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Shadow-Catcher - Classifies each box as Genuine or Anomalous')

    parser.add_argument('--model', help="Object Detection model\
                                                PointPillars or \
                                                PointRCNN (requires CUDA)",
                        default='PointPillars')

    parser.add_argument('--dataset_path', help='Path to dataset',
                        default='./sample_datasets/Adv_Kitti')

    parser.add_argument('--dataset_split', help='Training / Testing / Validation',
                        default='training')

    parser.add_argument('--frame', help="Frame wanted to Analyse \
                                        eg. 0: 0000 1: 0001... ",
                        default=4, type=int)

    parser.add_argument('--visualize', help="Visualize Anomalies",
                        default=1, type=int, choices=[0, 1])

    parser.add_argument('--config_path', help="Path to Configuration File")

    parser.add_argument('--ckpt_path', help="Path to pretrained model")

    parser.add_argument('--device', help="Specify device (cpu or cuda)",
                        default='cpu')

    args, _ = parser.parse_known_args()

    return args

args = parse_args()
if args.config_path is not None:
    cfg_path = args.config_path
else:
    cfg_path = "configs/" + args.model.lower() + ".yml"
if args.ckpt_path is not None:
    ckpt_path = args.ckpt_path
else:
    ckpt_path = "./pretrained_ckpt/" + args.model.lower() + ".pth"
cfg = _ml3d.utils.Config.load_from_file(cfg_path)
model_ = _ml3d.utils.get_module("model", args.model, "torch")
model = model_(device=args.device, **cfg.model)
pipeline = ml3d.pipelines.ObjectDetection(model, device=args.device,
                                          **cfg.pipeline)
pipeline.load_ckpt(ckpt_path=ckpt_path)
dataset = ml3d.datasets.KITTI(args.dataset_path)
training_dataset = dataset.get_split(args.dataset_split)
frame_ = dataset.get_split_list(args.dataset_split)[args.frame]
frame = frame_.replace(args.dataset_path +
                       "/training/velodyne/", '').replace('.bin', '')
data = training_dataset.get_data(args.frame)

def get_cam2velo_calib():
    # Calibration
    calib_path = frame_.replace('velodyne', 'calib').replace('.bin', '.txt')
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    R0_ = list(filter(lambda a: 'R0_rect' in a, lines))
    R0_ = R0_[0].strip().split(' ')[1:]
    R0 = np.array(np.reshape(R0_, [3, 3]), dtype=np.float64)

    V2C_ = list(filter(lambda a: 'Tr_velo_to_cam' in a, lines))
    V2C_ = V2C_[0].strip().split(' ')[1:]
    V2C = np.array(np.reshape(V2C_, [3, 4]), dtype=np.float64)

    inv_Tr = np.zeros_like(V2C, dtype=np.float64)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(V2C[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(V2C[0:3, 0:3]), V2C[0:3, 3])

    C2V = inv_Tr
    return R0, C2V

def viz(lines, pcd_in_roi, boxes, pcd, idx):
    """
    Utilizes Open3D visualization GUI
    """
    name_ = "Shadow Region for Box ID " + str(idx)
    vis = o3d.visualization.Visualizer()
    if vis.create_window(window_name = name_, width = 1920, height =1440, visible = True):
        vis.add_geometry(lines)
        vis.add_geometry(pcd_in_roi[0])
        vis.add_geometry(pcd_in_roi[1])
        vis.add_geometry(boxes)
        #vis.add_geometry(pcd)
        vis.run()
    vis.destroy_window()
    return

def main():
    # Get Camera to Velodyne Coordinate Calibration
    R0, C2V = get_cam2velo_calib()
  
    # Point Cloud Data
    points = np.array(data['point'][:, [0, 1, 2]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    gt_num = len(data['bounding_boxes'])
    
    # Run Inference
    pred_boxes = pipeline.run_inference(data)[0]
    
    gt = []
    pred = []
    gt.extend([BEVBox3D.to_dicts(b) for b in [data['bounding_boxes']]])
    pred.extend(BEVBox3D.to_dicts(b) for b in [pred_boxes])
    #fp = get_false_posi_iou(gt, pred, cfg)

    bboxes_data = data['bounding_boxes']
    bboxes_data.extend(pred_boxes)
   
    # Get Bounding Boxes Information
    objects = {}
    object_idx = 0  
    obj_coords = []
    for bboxes in bboxes_data:
        objects[object_idx] = {}
        # [x, y, z, h, w, l, yaw]
        objects[object_idx]['3d_box'] = bboxes.to_dict()['bbox']
        object_idx += 1
        
        # Object Corners in Velo Coordinates
        pts_3d_rect = np.array(bboxes.generate_corners3d(), dtype=np.float32)
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(R0), 
                                         np.transpose(pts_3d_rect)))
        n = pts_3d_ref.shape[0]
        pts_3d_hom = np.hstack((pts_3d_ref, np.ones((n, 1))))
        pts_3d_velo = np.dot(pts_3d_hom, np.transpose(C2V))
        obj_coords.append(pts_3d_velo)
    
    # Shadow-Catcher 
    bev_projection_lines = get_bev_projection(objects, obj_coords)
    pcd_in_roi, roi = get_roi(obj_coords, bev_projection_lines, pcd)
    count = count_pts_in_roi(pcd_in_roi, roi, verbose=False)
    shadow_region_lines = get_shadow_lines(bev_projection_lines)
    score = get_score(count, shadow_region_lines)
    print('\n' + frame_.replace('./sample_datasets/', '').replace((
                        '/training/velodyne/' + frame + '.bin'), '') + " Data Frame: " + str(frame))
    anomalies = score_classifier(score, gt_num)

    # Display Anomalies
    if args.visualize == 1:
        boxes, lines = get_draw_lines(obj_coords, bev_projection_lines)  
        for idx in anomalies:
            viz(lines[idx], pcd_in_roi[idx], boxes[idx], pcd, idx)
    

    return


if __name__ == '__main__':
    main()
