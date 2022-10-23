import argparse

import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from open3d._ml3d import metrics
from open3d._ml3d.datasets.utils import BEVBox3D


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyses Prediction from Detection Models PointPillar or \
                     PointRCNN (requires CUDA) and returns IoU & mAP')

    parser.add_argument('--model', help="Object Detection model\
                                                PointPillars or \
                                                PointRCNN (requires cuda)",
                        default='PointPillars',
                        required=False)

    parser.add_argument('--dataset_path', help='Path to dataset',
                        default='./sample_datasets/KITTI')
    
    parser.add_argument('--dataset_split', help='Training / Testing / Validation',
                        default='training')
    
    parser.add_argument('--frame' , help="Frame wanted to Analyse \
                                        eg. 0: 000035 1: 000134 2: 000200",
                        default=1, type=int)

    parser.add_argument('--config_path', help="Path to Configuration File")

    parser.add_argument('--ckpt_path', help="Path to pretrained model")

    parser.add_argument('--device', help="Specify device (cpu or cuda)",
                        default='cpu')
    
    parser.add_argument('--display_gt', help="Display Ground Truth Info",
                        default=1, type=int, choices=[0,1] )

    args, _ = parser.parse_known_args()

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
    model = model_(device=args.device, **cfg.model)
    pipeline = ml3d.pipelines.ObjectDetection(model, device=args.device,
                                              **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    dataset = ml3d.datasets.KITTI(args.dataset_path)
    dataset_split = dataset.get_split(args.dataset_split)
    frame_ = dataset.get_split_list(args.dataset_split)[args.frame]
    frame = frame_.replace(args.dataset_path +
                       "/training/velodyne/", '').replace('.bin', '')
    print('\n' + frame_.replace('./sample_datasets/', '').replace((
                        '/training/velodyne/' + frame + '.bin'), '') + " Data Frame: " + str(frame))
    
    data = dataset_split.get_data(args.frame)

    pred_boxes = pipeline.run_inference(data)[0]

    gt = []
    pred = []

    gt.extend([BEVBox3D.to_dicts(b) for b in [data['bounding_boxes']]])
    pred.extend(BEVBox3D.to_dicts(b) for b in [pred_boxes])

    overlaps = cfg.pipeline.get("overlaps", [0.5]) 
    similar_classes = cfg.pipeline.get("similar_classes", {})
    difficulties = cfg.pipeline.get("difficulties", [0])
    
    d, f = metrics.precision_3d(pred[0], gt[0], model.classes, 
                                difficulties, overlaps, 
                                bev=False, 
                                similar_classes = similar_classes)
    iou = metrics.iou_3d(pred[0]['bbox'].astype(np.float32), 
                         gt[0]['bbox'].astype(np.float32))
    
    iou_statement = []
    for i in range(len(iou)):
        if all(v == 0 for v in iou[i]):
            statement = ["\033[91mPred_Box ID: " + str(i + len(iou[i])) + 
            " doesn't Intersect any GT Boxes\033[0m"]
            iou_statement.extend(statement)
        for j in range(len(iou[i])):
            if iou[i][j] != 0:
                colour = "\033[0m"
                if iou[i][j] < 0.1:
                    colour = "\033[93m"
                statement = [colour + "Pred_Box ID: " + str(i + len(iou[i])) + 
                " has IoU of " + "{:.3f}".format(iou[i][j]) +
                " with GT_Box ID: " + str(j) + "\033[0m"]
                iou_statement.extend(statement)

    car_cnt = 0
    cyc_cnt = 0
    ped_cnt = 0
    for i in gt[0]['label']:
        if i == 'Car':
            car_cnt += 1 
        if i == 'Cyclist':
            cyc_cnt += 1
        if i == 'Pedestrian':
            ped_cnt += 1

    
    tp_ped = np.sum(d[0][2], axis=0, dtype=int)[1]
    fp_ped = np.sum(d[0][2], axis=0, dtype=int)[2]
    fn_ped = np.sum(f[0][2], dtype=int)
    ped_res = [tp_ped, fp_ped, fn_ped]

    tp_cyc = np.sum(d[1][2], axis=0, dtype=int)[1]
    fp_cyc = np.sum(d[1][2], axis=0, dtype=int)[2]
    fn_cyc = np.sum(f[1][2], dtype=int)
    cyc_res = [tp_cyc, fp_cyc, fn_cyc]
    
    tp_car = np.sum(d[2][2], axis=0, dtype=int)[1]
    fp_car = np.sum(d[2][2], axis=0, dtype=int)[2]
    fn_car = np.sum(f[2][2], dtype=int)
    car_res = [tp_car, fp_car, fn_car]

    res = [ped_res, cyc_res, car_res]

    if (tp_ped + fn_ped) == 0:
        ped_rec = 0
    else:
        ped_rec = tp_ped / (tp_ped + fn_ped)
    if (tp_ped + fp_ped) == 0:
        ped_prec = 0
    else: 
        ped_prec = tp_ped / (tp_ped + fp_ped)
    ped_ap = ped_rec * ped_prec
    ped_metric = [ped_rec, ped_prec, ped_ap] 
    
    if (tp_cyc + fn_cyc) == 0:
        cyc_rec = 0
    else:
        cyc_rec = tp_cyc / (tp_cyc + fn_cyc)
    if (tp_cyc + fp_cyc) == 0:
        cyc_prec = 0
    else:
        cyc_prec = tp_cyc / (tp_cyc + fp_cyc)
    cyc_ap = cyc_rec * cyc_prec
    cyc_metric = [cyc_rec, cyc_prec, cyc_ap]

    if (tp_car + fn_car) == 0:
        car_rec = 0
    else:
        car_rec = tp_car / (tp_car + fn_car)
    if (tp_car + fp_car) == 0:
        car_prec = 0
    else:
        car_prec = tp_car / (tp_car + fp_car)
    car_ap = car_rec * car_prec
    car_metric = [car_rec, car_prec, car_ap]

    metric = np.array([ped_metric, cyc_metric, car_metric])

    if args.display_gt == 1:
        print(("========== GT Box Info =========="))
        print(("Box ID" + "{:>7}".format("Type") + "{:>14}".format("Label")))
        for i in range(len(data['bounding_boxes'])):
            print(("  {:<8}".format(str(i)) + "{:<10}".format("GT") + 
                    "{:<10}".format(gt[0]['label'][i])))
        print("")

    print('Object Detection Model: ' + args.model)
    print(("================= Pred Box Info ================="))
    print(("Box ID" + "{:>7}".format("Type") + "{:>14}".format("Label") + 
            "{:>16}".format("Score")))
    for i in range(len(pred_boxes)):
        print(("  {:<8}".format(str(i + len(data['bounding_boxes']))) + 
                "{:<10}".format("Pred") + 
                "{:<10}".format(pred[0]['label'][i]) + 
                "{:>13.3f}".format(pred[0]['score'][i])))
    print("")

    print(("================ Ground Truth vs Prediction ================"))
    print(("Ground Truth: " + "{:>3.0f}".format(ped_cnt) + " x " + 
            "Pedestrians" + "{:>3.0f}".format(cyc_cnt) + " x " + 
            "Cyclists" "{:>3.0f}".format(car_cnt) + " x " + "Cars"))
    print(("Predictions: " + "{:>4.0f}".format(tp_ped+fp_ped) + " x " + 
            "Pedestrians" + "{:>3.0f}".format(tp_cyc+fp_cyc) + " x " + 
            "Cyclists" "{:>3.0f}".format(tp_car+fp_car) + " x " + "Cars"))
    print("IoU for Pred Boxes to GT Boxes:")
    for i in range(len(iou_statement)): 
        print(iou_statement[i])

    print("")
    print(("========== Prediction Results =========="))
    print(("class \\ results " + 
            "{:>9}".format("Tp") + "{:>6}".format("Fp") + 
            "{:>6}".format("Fn")))
    for i, c in enumerate(model.classes):
        print(("{:<19} " + "{:>5.0f} " * len(difficulties)).format(
                c + ":", *res[i]))
    print("[Tp: True Positives, Fp: False Positives, Fn: False Negatives]")

    print("")
    print(("========== Prediction Metrics =========="))
    print(("class \\ metrics " + 
            "{:>9}".format("Rec") + "{:>7}".format("Prec") + 
            "{:>5}".format("AP")))
    for i, c in enumerate(model.classes):
        print(("{:<20} " + "{:>5.2F} " * len(difficulties)).format(
                c + ":", *metric[i]))
    print("Overall mAP: {:.3F}".format(np.mean(metric[:, 2])))
    print("[Rec: Recall, Prec: Precision, AP: Average Precision," +
            " mAP: mean AP]")
    print("")


    return


if __name__ == '__main__':
    args = parse_args()
    main(args)

