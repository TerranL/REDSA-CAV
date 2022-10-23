import argparse
import os
import random
import shutil
from math import atan, pi, sqrt

import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d import metrics
from open3d._ml3d.datasets.utils import BEVBox3D
from open3d.ml import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distributed Object Removal Attacks for Kitti Scene')

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
                        default=0, type=int)

    parser.add_argument('--save_pcd', help="0: Don't Save Spoofed PCD \
                                            1: Save Spoofed PCD",
                        default=0, type=int, choices=[0, 1])

    parser.add_argument('--save_path', help="Path to Save Spoofed PCD. \
                                             Also, copies GT labels & Calib",
                        default='./sample_datasets/Adv_KITTI')

    parser.add_argument('--config_path', help="Path to Configuration File")

    parser.add_argument('--ckpt_path', help="Path to pretrained model")

    parser.add_argument('--device', help="Specify device (cpu / gpu / cuda)",
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
frame = frame_.replace(args.dataset_path + \
                        "/training/velodyne/", '').replace('.bin', '')
print("\nKitti Data Frame: " + str(frame))
data = training_dataset.get_data(args.frame)

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
    list_ = []
    for i, box in enumerate(boxes):
        pts_in_box = pc[is_points_in_box[:, i]]
        list_.append(pts_in_box)
    return list_

def get_pts_in_horizontal_view_range(target_box, target_box_pts):
    """
    Gets Points within a 10º Horizontal Angle (Azimuth angle)
    Args:
        target_box: box info [x,y,z,w,h,l,r]
        target_box_pts: np.array of points within bounding box
    Return:
        count: number of points within horizontal angle
        possible_attack_points: nx4 np.array of points within angle
    """
    count=0
    possible_attack_pts = []
    for i in target_box_pts:
        if (-pi/36) < (atan(i[1]/i[0]) - 
                        atan(target_box[1]/target_box[0])) < (pi/36):
            possible_attack_pts.append(i)
            count+=1
    return count, possible_attack_pts


def get_spoofed_points(pos_attack_pts, data_pts, num):
    """
    ORA - Randomly removes specified 'num' of points
    Args:
        pos_attack_pts: nx4 np.array of points to remove
        data_pts: nx4 np.array of points for reference
        num: number of points to remove (int)
    Return:
        num: number of points removed
        spoofed_pts: poisioned pcd (nx4 np.array)
        pos_attack_pts: delete from original and returned
    """
    # If number of possible attack points is less than 'num'
    if len(pos_attack_pts) < num:
        num = len(pos_attack_pts)

    # Get random nx4 np.array of points to remove
    attack_pts = []
    for i in range(num):
        idx = random.randrange(len(pos_attack_pts))
        attack_pts.append(pos_attack_pts[idx])
        pos_attack_pts = np.delete(pos_attack_pts, idx, 0)
    attack_pts = np.array(attack_pts)

    # Get Index of Points to remove
    pos_list = []
    for i in range(len(attack_pts)):
        pos = np.where(np.all(data_pts == attack_pts[i], axis=1))[0]
        pos_list.extend(pos)
    pos_list.sort()

    spoofed_pts = data_pts

    # Remove all indices of Points
    for i in range(len(pos_list)):
        if i == 0:
            spoofed_pts = np.delete(spoofed_pts, pos_list[i], 0)
        else: 
            spoofed_pts = np.delete(spoofed_pts, (pos_list[i] - i), 0)
    
    
    return num, spoofed_pts, pos_attack_pts

def get_pred_box_id(iou, target):
    """
    Get Index of Box which Intersects Target Box
    Args:
        iou: List of np.arrays containing IoU data
        target (int): index of target box
    Return:
        if found:
            i: Index of Maximum IoU
            max_list[target]: (float) Maximum IoU value with Box
        else:
            -1, -1 (no value found)
    """
    
    max_list = np.amax(iou, axis=0)
    for i in range(len(iou)):
            if iou[i][target] == max_list[target]:
                return i, max_list[target]
    return -1, -1

def get_prediction_info(data, idx, gt):
    """
    Runs Object Detector, and gets information on boxes, its confidence score and IoU with GT
    Args:
        data: Dictionary of [nx4] np.array of points and calibration
        idx: Index of GT box targeted
        gt: Ground Truth Bounding boxes as Dictionary
    Return:
        bboxes: List of BEVBox3D objects
        info_: Dictionary of IoU and Score for each Targeted Box
    """
    bboxes = pipeline.run_inference(data)[0]
    box = [] 
    box.extend(BEVBox3D.to_dicts(b) for b in [bboxes])
    iou = metrics.iou_3d(box[0]['bbox'].astype(np.float32), 
                         gt[0]['bbox'].astype(np.float32))


    info_ = {}
    for i in idx:
        info_[i] = {}
        id_, iou_val = get_pred_box_id(iou, i)
        info_[i]['iou'] = iou_val
        if iou_val == 0:
            info_[i]['score'] = 0
        else:    
            info_[i]['score'] = box[0]['score'][id_]

    return bboxes, info_

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

def main():
    # Define Reference Box at [x,y,z], size[w,h,l], [yaw], [label], [score]
    reference_box = BEVBox3D([0, 0, 0], [2,2,3], pi/2,'Car',-1)
    ref_box_xyzwhlr = reference_box.to_xyzwhlr()

    # Ground Truth Information
    gt = []
    gt.extend([BEVBox3D.to_dicts(b) for b in [data['bounding_boxes']]])
    bbox = data['bounding_boxes']
    gt_xyzwhlr = [box.to_xyzwhlr() for box in bbox] 
    distance_list = get_linear_distance_list(gt_xyzwhlr, ref_box_xyzwhlr)
    pts_in_box = get_points_in_boxes(data['point'],
                                              data['bounding_boxes'],
                                              gt_xyzwhlr)
    print(("=================== Ground Truth ==================="))
    print(("Box ID" + "{:>8}".format("Label") + 
            "{:>22}".format("Distance (m)") + 
            "{:>15}".format("No. of Points")))
    for i in range(len(data['bounding_boxes'])):
        print(("  {:<8}".format(str(i)) + 
                "{:<10}".format(gt[0]['label'][i]) + 
                "{:>11.1f}".format(distance_list[i]) + 
                "{:>14.0f}".format(len(pts_in_box[i]))))

    # Get Possible Attack Points within 10º Horizontal Angle
    num_pts = {}
    pos_attack_pts = []
    for i in range(len(bbox)):
        cnt, pts_ = get_pts_in_horizontal_view_range(gt_xyzwhlr[i],
                                                        pts_in_box[i])
        num_pts[i] = cnt
        pos_attack_pts.append(pts_)
    # Sort by Least to Most Number of Points
    asc_num_ = {k: v for k, v in sorted(num_pts.items(), 
                            key=lambda item: item[1])}

    # Get Spoofed Points and Record Number of Removed Points for Each Target Box
    spoofed_pts = np.array(data['point'])
    total_spoofed = 0
    count_ = {}
    for keys in asc_num_.keys():
        count_[keys] = np.array(0)
    while total_spoofed < 200:
        for keys in asc_num_.keys():
            if len(pos_attack_pts[keys]) > 40:
                spoof_num = 40
            else:
                spoof_num = len(pos_attack_pts[keys])
            if total_spoofed + spoof_num >= 200:
                spoof_num = 200 - total_spoofed
            (num_, spoofed_pts, 
                pos_attack_pts[keys]) = get_spoofed_points(pos_attack_pts[keys], 
                                                            spoofed_pts,
                                                            spoof_num)
            count_[keys] = count_[keys] + num_
            total_spoofed+=spoof_num 

    # For Each GT-Box, Record If Points were Removed
    target_idx = []
    for i in count_.keys():
        if count_[i] > 0:
            target_idx.append(i)
    target_idx.sort()
    
    # Print IoU and Confidence of Targeted Boxes before D-ORA
    print("\nBefore D-ORA: ")
    pred_boxes, pred_info = get_prediction_info(data, target_idx, gt)
    for i in pred_info.keys():
        if pred_info[i]['score'] == 0:
            print("Box ID: " + str(i) + " was not detected")
        else:
            print(("Box ID: " + str(i) + " was detected with IoU: " +
                   "{:.3f}".format(pred_info[i]['iou']) +  ", Conf: " + 
                   "{:.3f}".format(pred_info[i]['score'])))
    

    #Print IoU and Confidence of Targeted Boxes after D-ORA
    print("\nD-ORA targeted Box ID(s): " + str(target_idx))
    spoofed_data = {
        'point': spoofed_pts,
        'calib': data['calib']
    }
    spoofed_boxes, spoofed_info = get_prediction_info(spoofed_data, 
                                                        target_idx, gt)
    for i in spoofed_info.keys():
        if spoofed_info[i]['score'] == 0:
            print(("After removing " + str(count_[i]) + 
                    " points from Box ID: " + str(i) + 
                    ", was not detected"))
        else: 
            print("After removing " + str(count_[i]) + 
                    " points from Box ID: " + str(i) + 
                    ", was detected with IoU: " + 
                    "{:.3f}".format(spoofed_info[i]['iou']) + 
                    ", Conf: " + "{:.3f}".format(spoofed_info[i]['score']))
   
    # Save PCD  
    if args.save_pcd == 1:
        save_spoofed_pcd(spoofed_pts) 


    return


if __name__ == '__main__':
    main()
