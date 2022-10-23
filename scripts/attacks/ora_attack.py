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
        description='Object Removal Attacks for specified 3D Kitti \
                     Data & Object')

    parser.add_argument('--model', help="Object Detection model\
                                                PointPillars or \
                                                PointRCNN (requires CUDA)",
                        default='PointPillars')

    parser.add_argument('--dataset_path', help='Path to Pristine Cloud Dataset',
                        default='./sample_datasets/KITTI')

    parser.add_argument('--dataset_split', help='Training / Testing / Validation',
                        default='training')

    parser.add_argument('--frame' , help="Frame wanted to Analyse \
                                        eg. 0: 000035 1: 000134 2: 000200",
                        default=1, type=int)
    
    parser.add_argument('--target_box', help='Specify Ground Truth Object \
                                              to target',
                        default = 0, type=int)
    
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
    
    parser.add_argument('--print_gt_only', help="Print Ground Truth Info Only",
                         default=0, type=int, choices=[0,1])


    args, _ = parser.parse_known_args()


    return args

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

def get_pred_box_id(iou, target):
    """
    Get Index of Box which Intersects Target Box
    Args:
        iou: List of np.arrays containing IoU data
        target (int): index of target box
    Return:
        if found:
            i: non-zero index in IoU data
            _: IoU value (float)
        else:
            -1, -1 (no value found)
    """
    for i in range(len(iou)):
            if iou[i][target] != 0:
                return i, iou[i][target]
    return -1, -1


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
    """
    # If number of possible attack points is less than 'num'
    if len(pos_attack_pts) < num:
        num = len(pos_attack_pts)

    # Get random nx4 np.array of points to remove
    attack_pts = []
    for i in range(num):
        idx = random.randrange(len(pos_attack_pts))
        attack_pts.append(pos_attack_pts[idx])
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
    
    
    return num, spoofed_pts

def save_spoofed_pcd(pts, args):
    """
    Function to save spoofed data
    Args:
        pts: point cloud data as np.array
    """
    print("Saving Spoofed PCD Data ...")
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


def main(args, pipeline, data):
    # Define Reference Box at [x,y,z], size[w,h,l], [yaw], [label], [score]
    reference_box = BEVBox3D([0, 0, 0], [2,2,3], pi/2,'Car',-1)
    ref_box_xyzwhlr = reference_box.to_xyzwhlr()

    # Get ground-truth boxes
    gt = []
    gt.extend([BEVBox3D.to_dicts(b) for b in [data['bounding_boxes']]])
    data_pts = np.array(data['point'])
    bbox = data['bounding_boxes']
    xyzwhlr = [box.to_xyzwhlr() for box in bbox] 
    distance_list = [] 
    for i in xyzwhlr:
        dist = get_linear_distance(i, ref_box_xyzwhlr)  
        distance_list.append(dist)

    is_points_in_box = datasets.utils.operations.points_in_box(
                         data['point'], xyzwhlr)

    pts_in_box_list = []
    for i, box in enumerate(bbox):
        pts_in_box = data['point'][is_points_in_box[:, i]]
        pts_in_box_list.append(pts_in_box)
   
    print(("=================== Ground Truth ==================="))
    print(("Box ID" + "{:>8}".format("Label") + 
            "{:>22}".format("Distance (m)") + 
            "{:>15}".format("No. of Points")))
    for i in range(len(data['bounding_boxes'])):
        print(("  {:<8}".format(str(i)) + 
                "{:<10}".format(gt[0]['label'][i]) + 
                "{:>11.1f}".format(distance_list[i]) + 
                "{:>14.0f}".format(len(pts_in_box_list[i]))))
    # To display gt info only
    if args.print_gt_only == 1:
        print("")
        return
    
    # Get Prediction Results without ORA Attack
    pred_boxes = pipeline.run_inference(data)[0]
    pred = [] 
    pred.extend(BEVBox3D.to_dicts(b) for b in [pred_boxes])
    iou = metrics.iou_3d(pred[0]['bbox'].astype(np.float32), 
                         gt[0]['bbox'].astype(np.float32))
    pred_box_id, iou_val = get_pred_box_id(iou, args.target_box)

    print("\nTargeting Box ID: " + str(args.target_box))
    target_box_pts = pts_in_box_list[args.target_box]
    target_box = xyzwhlr[args.target_box]
    
    print("Box ID: " + str(args.target_box) + " has a total of " + 
            str(len(target_box_pts)) + " data points")
    

    
    count, pos_attack_pts = get_pts_in_horizontal_view_range(target_box, 
                                                        target_box_pts)
 
    print("but only contains " + str(count) + 
            " points within a 10 degree horizontal viewing angle")
    
    print("\nBefore ORA, Object ID: " + str(args.target_box) + 
            " was detected with a confidence score of: " 
            + "{:.3f}".format(pred[0]['score'][pred_box_id]) +
              " and IoU of: " + "{:.3f}".format(iou_val)) 
    
    # ORA of 100 Points
    num_1, spoofed_pts_100 = get_spoofed_points(pos_attack_pts, data_pts, 100)

    # Run Predictions on ORA attack of 100 points     
    ora_100 = {
        'point': spoofed_pts_100,
        'calib': data['calib']
    }
    pred_100 = []
    pred_boxes_100 = pipeline.run_inference(ora_100)[0]
    pred_100.extend(BEVBox3D.to_dicts(b) for b in [pred_boxes_100])
    
    # Get IoU intersection of Target Box wrt Ground Truth
    # If no IoU found, target box was not detected
    # Else: record Conf Score and IoU rating
    iou_100 = metrics.iou_3d(pred_100[0]['bbox'].astype(np.float32), 
                         gt[0]['bbox'].astype(np.float32))
    pred_box_id_100, iou_val_100 = get_pred_box_id(iou_100, args.target_box)
    if pred_box_id_100 == -1 or iou_val_100 == -1:
        print("After ORA of " + str(num_1) + " points, Box ID: " +
                str(args.target_box) + " was not detected")
    else:
        print("After ORA of " + str(num_1) + " points, confidence score: " 
                + "{:.3f}".format(pred_100[0]['score'][pred_box_id_100]) 
                + " and IoU of: " + "{:.3f}".format(iou_val_100)) 

    # ORA of 200 Points
    num_2, spoofed_pts_200 = get_spoofed_points(pos_attack_pts, data_pts, 200)
    # If number of spoofed points is less than ORA-100 Nothing to compute
    if num_2 <= num_1:
        print("")
    else:
        # Else Run prediction on new pcd with ORA 200 points
        ora_200 = {
            'point': spoofed_pts_200,
            'calib': data['calib']
        }
        
        pred_200 = []
        pred_boxes_200 = pipeline.run_inference(ora_200)[0]
        pred_200.extend(BEVBox3D.to_dicts(b) for b in [pred_boxes_200])
        iou_200 = metrics.iou_3d(pred_200[0]['bbox'].astype(np.float32), 
                            gt[0]['bbox'].astype(np.float32))

        pred_box_id_200, iou_val_200 = get_pred_box_id(iou_200, args.target_box)
        if pred_box_id_200 == -1 or iou_val_200 == -1:
            print("After ORA of " + str(num_2) + " points, Box ID: " +
                    str(args.target_box) + " was not detected")
        else:
            print("After ORA of " + str(num_2) + " points, confidence score: " 
                    + "{:.3f}".format(pred_200[0]['score'][pred_box_id_200])
                    + " and IoU of: " + "{:.3f}".format(iou_val_200))
        print("")

    if args.save_pcd == 1:
        if num_2 <= num_1:
            save_spoofed_pcd(spoofed_pts_100, args) 
        else:
            save_spoofed_pcd(spoofed_pts_200, args)

    return


if __name__ == '__main__':
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

    main(args, pipeline, data)
