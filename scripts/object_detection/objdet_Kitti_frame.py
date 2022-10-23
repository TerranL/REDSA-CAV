import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from open3d.ml import datasets

import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for inference of object detection')

    parser.add_argument('--model', help="Object Detection model\
                                                PointPillars or \
                                                PointRCNN (requires cuda)",
                        default='PointPillars',
                        required=False)

    parser.add_argument('--dataset_path', help='Path to dataset',
                        default = './sample_datasets/KITTI')
    
    parser.add_argument('--frame' , help="Frame wanted to Analyse \
                                        eg. 0: 000035 1: 000134 3: 000200",
                        default=1, type=int)

    parser.add_argument('--config_path', help="Path to Configuration File")

    parser.add_argument('--ckpt_path', help="Path to pretrained model")

    parser.add_argument('--device', help="Specify device (cpu or cuda)",
                        default='cpu')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


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

    dataset = ml3d.datasets.KITTI(args.dataset_path)
    dataset_split = dataset.get_split('training')

    data = dataset_split.get_data(args.frame)

    results = pipeline.run_inference(data)[0]
    
    boxes = data['bounding_boxes']
    boxes.extend(results)
    
    vis = ml3d.vis.Visualizer()
    vis.visualize([
    {
        'name': 'PC',
        'points': data['point']
    }
    ], bounding_boxes=boxes)
    
    return    


if __name__=='__main__':
    args = parse_args()
    main(args)