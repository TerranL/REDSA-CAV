import argparse
import os
import random
import shutil
from math import atan, atan2, cos, pi, sin, sqrt

import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d import metrics
from open3d._ml3d.datasets.utils import BEVBox3D
from open3d.ml import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spoofing Target Objects to Front-near for specified \
                     3D Kitti Data & Object')

    parser.add_argument('--model', help="Object Detection model\
                                                PointPillars or \
                                                PointRCNN (requires CUDA)",
                        default='PointPillars')

    parser.add_argument('--dataset_path', help='Path to Pristine Cloud Dataset',
                        default='./sample_datasets/KITTI')

    parser.add_argument('--dataset_split', help='Training / Testing / Validation',
                        default='training')

    parser.add_argument('--frame', help="Frame wanted to Analyse \
                                        eg. 0: 000035 1: 000134 2: 000200",
                        default=2, type=int)
    
    parser.add_argument('--target_box', help='Specify Ground Truth Object \
                                              to target',
                        default=0, type=int)

    parser.add_argument('--visualize', help="0: Don't Visualize Spoofed Data \
                                             1: Visualize Spoofed Data",
                        default=0, type=int, choices=[0, 1])

    parser.add_argument('--save_pcd', help="0: Don't Save Spoofed PCD \
                                            1: Save Spoofed PCD",
                        default=0, type=int, choices=[0, 1])
    
    parser.add_argument('--save_path', help="Path to Save Spoofed PCD. \
                                             Also, copies GT labels & Calib",
                        default='./sample_datasets/Adv_KITTI')
        
    parser.add_argument('--config_path', help="Path to Configuration File")

    parser.add_argument('--ckpt_path', help="Path to Pretrained Model")

    parser.add_argument('--device', help="Specify device (cpu / gpu / cuda)",
                        default='cpu')

    parser.add_argument('--print_gt_only', help="Print Ground Truth Info Only",
                         default=0, type=int, choices=[0,1])

    args, _ = parser.parse_known_args()


    return args

# Set up configs, ckpt, model & pipeline based on parsed arguments
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
frame = frame_.replace(args.dataset_path + "/training/velodyne/", '').replace('.bin', '')
print("\nKitti Data Frame: " + str(frame))
data = training_dataset.get_data(args.frame)


def get_occlusion_rating():
    """
    Retrieves Occlution Rating in GT label file
    Return: List of Occlution ratings for each GT label 
     """
    list = []
    label_2_path = frame_.replace('velodyne', 'label_2').replace('.bin', '.txt')

    with open(label_2_path, 'r') as f:
        lines = f.readlines()
    for i in lines:
        data = i.strip().split(' ')[1:]
        list.append(data[1])
    return list


def get_linear_distance(point, ref):
    """
    Calculates linear distance for 3D points
    Args:
        point [x1, y1, z1]
        ref [x0, y0 , z0]
    Return:
        distance (float)
    """
    dist = sqrt((point[0]-ref[0])**2 + (point[1]-ref[1])**2 + 
                 (point[2]-ref[2])**2)
    return dist


def get_linear_distance_list(box, ref):
    """
    Get a list of distances from list of boxes
    Args:
        box: list of boxes with centre coords [x,y,z]
        ref: reference point [x,y,z]
    Return:
        list: list of distances
    """

    list = []
    for i in box:
        dist = get_linear_distance(i, ref)
        list.append(dist)
    return list


def get_points_in_boxes(pc, boxes, box_info):
    """
    Get a list of points in each box
    Args:
        pc: point cloud space as np.array
        boxes: list of boxes as o3d.BEVBox3d
        box_info: list of boxes as [x,y,z,w,h,l,r]
    Return:
        list: list of np.arrays
    """
    is_points_in_box = datasets.utils.operations.points_in_box(
        pc, box_info)
    list = []
    for i, box in enumerate(boxes):
        pts_in_box = pc[is_points_in_box[:, i]]
        list.append(pts_in_box)
    return list


def get_pred_box_id(iou):
    """
    Get Index of Box which Intersects
    Args:
        iou: List of np.arrays containing IoU data
    Return:
        if found:
            i: non-zero index in IoU data
            _: IoU value (float)
        else:
            -1, -1 (no value found)
    """
    for i in range(len(iou)):
        for j in range(len(iou[i])):
            if iou[i][args.target_box] != 0:
                return i, iou[i][args.target_box]
    return -1, -1

def get_pred(data):
    """
    Function to produce prediction results
    Args:
        data: dictionary with data points and calibration
    Return: 
        boxes: List of BEVBox3D Objects
        dict: dictionary of BEVBox3D info
        xyzwhlr: box info in [x,y,z,h,l,w,r]
    """
    boxes = pipeline.run_inference(data)[0]
    dict =[]
    dict.extend(BEVBox3D.to_dicts(b) for b in [boxes])
    xyzwhlr = [box.to_xyzwhlr() for box in boxes]

    return boxes, dict, xyzwhlr


def save_spoofed_pcd(pts):
    """
    Function to save spoofed data
    Args:
        pts: point cloud data as np.array
    """
    print("\nSaving Spoofed PCD Data ...")
    # Get Number of Files in Save Path (eg. Adv_Kitti) Velodyne Dir
    dataset_ = ml3d.datasets.KITTI(args.save_path)
    file_ = dataset_.get_split_list(args.dataset_split)
    idx = len(file_)

    # Save Spoofed PCD in .bin file             
    bin_path = args.save_path + "/" + args.dataset_split + "/velodyne/"
    bin_file_name = "{:04d}.bin".format(idx)
    bin_file_path = os.path.join(bin_path, bin_file_name)
    pts.tofile(bin_file_path)

    # Copy Relevant Label_2 File
    label_2_path = frame_.replace('velodyne', 'label_2').replace('.bin', '.txt')
    spoofed_label_path = bin_file_path.replace('velodyne', 'label_2').replace('.bin', '.txt')
    shutil.copy2(label_2_path, spoofed_label_path)

    # Copy Relevant Calib File Path
    calib_path = label_2_path.replace('label_2', 'calib')
    spoofed_calib_path = spoofed_label_path.replace('label_2','calib')
    shutil.copy2(calib_path, spoofed_calib_path)

    print("Saved: ")
    print(" PCD data: " + bin_file_path)
    print(" GT labels: " + spoofed_label_path)
    print(" Calibration: " + spoofed_calib_path)
    print("")
    return


def main(args):
    # Define Reference Box at [x,y,z], size[w,h,l], [yaw], [label], [score]
    reference_box = BEVBox3D([0, 0, 0], [2,2,3], pi/2,'Car',-1)
    ref_box_xyzwhlr = reference_box.to_xyzwhlr()
   
    # Get ground-truth boxes
    gt = []
    gt.extend([BEVBox3D.to_dicts(b) for b in [data['bounding_boxes']]])
    gt_xyzwhlr = [box.to_xyzwhlr() for box in data['bounding_boxes']] 
    distance_list = get_linear_distance_list(gt_xyzwhlr, ref_box_xyzwhlr)
    points_in_gt_boxes = get_points_in_boxes(data['point'], 
                                             data['bounding_boxes'], 
                                             gt_xyzwhlr)
    occ = get_occlusion_rating()

    # Print Ground Truth Bounding Box Info
    print(("==================== Ground Truth ===================="))
    print(("Box ID" + "{:>8}".format("Label") +
           "{:>22}".format("Distance (m)") + 
           "{:>14}".format("Occ Rating")))
    for i in range(len(data['bounding_boxes'])):
        print(("  {:<8}".format(str(i)) +
                "{:<10}".format(gt[0]['label'][i]) +
                "{:>10.1f}".format(distance_list[i]) +
                "{:>15}".format(occ[i])))
    print(("Occ Rating: {0: Fully Visible, 1: Partly Occluded," +
            " 2: Largely Occluded, 3: Unknown}"))
    if args.print_gt_only == 1:
        print("")
        return
    # Get Prediction Results without Spoofing
    pred_boxes, pred_dict, pred_xyzwhlr = get_pred(data)
    pred_dist_list = get_linear_distance_list(pred_xyzwhlr, 
                                              ref_box_xyzwhlr)
    pts_in_pred_boxes = get_points_in_boxes(data['point'],
                                            pred_boxes,
                                            pred_xyzwhlr)

    # Define target box based on parsed arguments
    target_box = gt_xyzwhlr[args.target_box]
    target_box_pts = points_in_gt_boxes[args.target_box]

    # Get Positional Index for All Points in Target Box Points
    pos_list = []
    for i in range(len(target_box_pts)):
        pos = np.where(np.all(data['point'] == target_box_pts[i], axis=1))[0]
        pos_list.extend(pos)
    pos_list.sort()

    # Truncate Index to a Total of 200 Points
    # Or less if Number of Points in Target Box is Less than 200
    idx_list = []
    num = 200 # Total Number of Spoofed Points
    if len(target_box_pts) < num:
        num = len(target_box_pts) 
    for i in range(num):
        idx = random.choice(pos_list)
        idx_list.append(idx)
    idx_list.sort()

    spoofed_pts = np.array(data['point'])
    data_pts = np.array(data['point'])

    # Spoof Data to Front-Near
    # theta - rotation with reference to y-axiz around the z-axis
    # tao - linear distance difference to approximately (8m)
    theta = 2*pi - atan2(target_box[1]-ref_box_xyzwhlr[1], 
                         target_box[0]-ref_box_xyzwhlr[0])
    tao = 8 - get_linear_distance(target_box, ref_box_xyzwhlr)
    for i in idx_list:
        alpha = atan(data_pts[i][1]/data_pts[i][0])
        spoofed_pts[i][0] = (data_pts[i][0]*cos(theta) - 
                             data_pts[i][1]*sin(theta) + 
                             tao*cos(theta+alpha))
        spoofed_pts[i][1] = (data_pts[i][0]*sin(theta) + 
                             data_pts[i][1]*cos(theta) + 
                             tao*sin(theta+alpha))

    # Define Spoofed Data using Pristine Cloud Calibration
    spoofed_data = {
        'point': spoofed_pts,
        'calib': data['calib']
    }
    
    # Spoofed Results
    spoof_boxes, spoof_dict, spoof_xyzwhlr = get_pred(spoofed_data)
    spoofed_pred_dist_list = get_linear_distance_list(spoof_xyzwhlr, ref_box_xyzwhlr)

    # Get List of Detected Spoofed Boxes based on IoU with Non-spoofed Predictions
    iou_ = metrics.iou_3d(spoof_dict[0]['bbox'].astype(np.float32), 
                         pred_dict[0]['bbox'].astype(np.float32))
    fn_list = []
    for i in range(len(iou_)):
        if np.all(iou_[i] == 0):
            fn_list.append(i)
    
    # Screen Dump Results
    print("\nTargeting Box ID: " + str(args.target_box))
    print("Box ID: " + str(args.target_box) + " has a total of " + 
            str(len(target_box_pts)) + " points")
    print("Total of " + str(num) + " points Spoofed to Front-near Location") 
    print("\nAt Front-near, model detected " + str(len(fn_list)) + " object(s):")
    for i in fn_list:
        print(("Box ID: " + str(i + len(pred_boxes) + 
                len(data['bounding_boxes']) + 2)
                + " (" + spoof_dict[0]['label'][i] 
                + ") with a Conf. Score: " + 
                "{:.3f}".format(spoof_dict[0]['score'][i]) 
                + " at " + "{:.1f}".format(spoofed_pred_dist_list[i]) + 
                "m infront of Reference Vehicle"))


    # Check if Spoofed Predictions still detected box at target box location
    iou_gt = metrics.iou_3d(gt[0]['bbox'].astype(np.float32),
                            spoof_dict[0]['bbox'].astype(np.float32))
    if np.all(iou_gt[args.target_box] == 0):
        print("And, model didn't detect an object at targeted location")
    else: 
        print("And, model still detected an object at targeted location")
    print("")

    # Visualize
    if args.visualize == 1:
        # Append Reference box for Visual Reference
        data['bounding_boxes'].append(reference_box)
        vis = ml3d.vis.Visualizer()
        vis_data = [
        {
            'name': 'Ground_Truth',
            'points': data['point'],
            'bounding_boxes': data['bounding_boxes']  
        },
        {
            'name': "Pristine_PC",
            'points': data_pts,
            'bounding_boxes': pred_boxes
        },
        {
            'name': 'Spoofed_PC',
            'points': spoofed_pts,
            'bounding_boxes': spoof_boxes
        }]
        vis.visualize(vis_data)

    if args.save_pcd == 1:
        save_spoofed_pcd(spoofed_pts)


    return


if __name__ == '__main__':
    main(args)
