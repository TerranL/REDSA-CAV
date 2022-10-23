import argparse
import os

import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Applies Object Detection Models (PointPillars \
                        or PointRCNN(requires cuda) on Raw Kitti Data \
                        Frame")
    
    parser.add_argument('--model', help = "Object Detection model\
                                                PointPillars or \
                                                PointRCNN",
                        default = 'PointPillars',
                        required = False)

    parser.add_argument('--dataset_path', help = "Path to Datataset \n\
                                                eg: './sampledatasets/0001'",
                        default = './sample_datasets/0001')

    parser.add_argument('--frame', help = "Select Frame from dataset",
                        default = 1, type = int)

    parser.add_argument('--fv', help = "To Visualise Front-View Only",
                        default = 1, type = int)

    parser.add_argument('--cam2cam_calib_path', help = "Path to cam_to_cam \
                                                calibration file")
    
    parser.add_argument('--velo2cam_calib_path', help = "Path to velo_to_cam \
                                                calibration file")
    
    parser.add_argument('--config_path', help = "Path to Configuration File")

    parser.add_argument('--ckpt_path', help = "Path to pretrained model")
    
    parser.add_argument('--device', help = "Specify device (cpu or cuda)",
                        default = 'cpu')
    
    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not provided".
              format(k))

    return args


def get_calib(dataset_path, calib_1, calib_2):
    if calib_1 is not None:
        cam2cam_calib_path = calib_1
    else:
        cam2cam_calib_path = dataset_path + "/calib_cam_to_cam.txt"
    if calib_2 is not None:
        velo2cam_calib_path = calib_2
    else:
        velo2cam_calib_path = dataset_path + "/calib_velo_to_cam.txt"

    with open(cam2cam_calib_path, 'r') as f:
        lines = f.readlines()
    
    P2 = list(filter(lambda a: 'P_rect_02' in a, lines))
    P2 = P2[0].strip().split(' ')[1:]
    P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
    P2 = np.concatenate((P2, np.array([[0., 0., 1., 0.]],
                                        dtype=np.float32)), axis=0)

    _rect_4x4 = list(filter(lambda a: 'R_rect_00' in a, lines))
    _rect_4x4 = _rect_4x4[0].strip().split(' ')[1:]
    rect_4x4 = np.eye(4, dtype=np.float32)
    rect_4x4[:3, :3] = np.array(_rect_4x4, dtype=np.float32).reshape(3, 3)
    
    with open(velo2cam_calib_path, 'r') as f:
        lines = f.readlines()

    _Tr_velo2cam_R = list(filter(lambda a: 'R' in a, lines))
    _Tr_velo2cam_R = _Tr_velo2cam_R[0].strip().split(' ')[1:]
    _Tr_velo2cam_T = list(filter(lambda a: 'T' in a, lines))
    _Tr_velo2cam_T = _Tr_velo2cam_T[0].strip().split(' ')[1:]
    _Tr_velo2cam = _Tr_velo2cam_R + _Tr_velo2cam_T
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3] = np.array(_Tr_velo2cam,
                                dtype=np.float32).reshape(3, 4)

    world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)
    cam_img = np.transpose(P2)

    return {'world_cam': world_cam, 'cam_img': cam_img}


def extract_pc_data(dataset_path, frame, fv):
    data_path = dataset_path + "/velodyne_points/data/"
    

    count = 0
    for files in os.listdir(data_path):
        
        if files.startswith('0') and files.endswith('.bin'):
            if count==frame:
                pc_file = data_path + files

                points = np.fromfile(pc_file, dtype=np.float32, count=-1,
                                    sep='', offset=0)
                X = points[0::4]
                Y = points[1::4]
                Z = points[2::4]
                W = points[3::4]
                
                pc = np.zeros((np.size(X), 4))
                pc[:, 0] = X
                pc[:, 1] = Y
                pc[:, 2] = Z
                pc[:, 3] = W
                
            else:
                count += 1

    if fv == 1:
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        min_val = np.array(point_cloud_range[:3])
        max_val = np.array(point_cloud_range[3:])
        pcd = np.array(pc[:, 0:4], dtype=np.float32)
        pcd = pcd[np.where(
                    np.all(np.logical_and(pc[:, :3] >= min_val,
                                            pc[:, :3] < max_val),
                                            axis=-1))]
        return pcd

    return pc

def main(args):
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
    model = model_(device = args.device, **cfg.model)
    pipeline = ml3d.pipelines.ObjectDetection(model, device = args.device, 
                                                **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path = ckpt_path)

    calib = get_calib(args.dataset_path, args.cam2cam_calib_path, 
                        args.velo2cam_calib_path)

    points = extract_pc_data(args.dataset_path, args.frame, args.fv)

    data = {
            'point': points,
            'calib': calib
        }
    
    results = pipeline.run_inference(data)[0]

    vis = ml3d.vis.Visualizer()
    vis.visualize([{
        'name': "Kitti_Frame_" + str(args.frame),
        'points': points
        }], bounding_boxes = results)
    

    return
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
