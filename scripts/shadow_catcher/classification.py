import numpy as np
from open3d._ml3d import metrics

def get_score(obj_points, lines, alpha=0.25):
    """
    Calculates scores based on Hau et al. 
    "Ghosbuster: Looking Into Shadows to Detect Ghost Objects in AV 3D Sensing"
    ref: https://arxiv.org/abs/2008.12008.
    Args:
        obj_points: PCD in ROI
        lines: Dictionary of each box 3D Shadow Lines
    Return:
        obj_score: Dictionary of key: object box id, value: score
    """
    obj_score = {}
    for obj_id in obj_points.keys() :
        points = obj_points[obj_id]['pts_in_shadow']
        if len(points) == 0 :
            obj_score[obj_id] =  0
            continue
        mid_line = lines[obj_id]['midline']
        max_grad = lines[obj_id]['max_grad']
        min_grad = lines[obj_id]['min_grad']
        # new : start line
        start_line = lines[obj_id]['start_line']
        end_line = lines[obj_id]['end_line']
                                   
        score_denom = len(points)
        score_num = 0
        for point in points :
            dist_mid = mid_line.get_dist_to_point(point[0],point[1])
            dist_max =  max_grad.get_dist_to_point(point[0],point[1])
            dist_min =  min_grad.get_dist_to_point(point[0],point[1])
            dist_bound = np.minimum(dist_max, dist_min)
            score_1 = np.exp((np.log(0.5)/alpha)*(dist_mid/(dist_mid+dist_bound)))
            dist_start = start_line.get_dist_to_point(point[0],point[1])
            dist_end = end_line.get_dist_to_point(point[0],point[1])
            score_2 = np.exp((np.log(0.5)/alpha)*(dist_start/(dist_start+dist_end)))
            score_ = score_1*score_2
            score_num += score_
        min_weight = (np.exp((np.log(0.5)/alpha)))**2
        score = (score_num-(score_denom*min_weight))/(score_denom*(1-min_weight))
        obj_score[obj_id] =  score
    return obj_score


def score_classifier(scores, gt_num, threshold=0.241):
    """
    Classifies scores anomolous or genuine based on threshold
    Args:
        scores: Dictionary of scores for each box
        gt_num (int): Number of Ground Truth boxes to categorize GT or Predictions
        threshold (int): Default threshold is 0.241 (as determined by Hau et al.)
    """
    anomaly = []
    print(("============= SC Classifier ============="))
    print(("Box ID" + "{:>7}".format("Type") + "{:>11}".format("Score") + 
            "{:>16}".format("Classifier")))
    for obj in range(len(scores)) :
        if obj < gt_num:
            obj_type = 'GT'
        else:
            obj_type = 'Pred'
        if scores[obj] > threshold :
            score_label = ('\033[91m' + "{:>14}".format('Anomaly') + 
                           '\033[0m')
            anomaly.append(obj)
        else :
            score_label = 'Genuine'
        print(" {:<5}".format(str(obj)) + "{:>6}".format(obj_type) + 
              "{:>12.3f}".format(scores[obj]) + 
              "{:>14}".format(score_label))
    print("")


    return anomaly


def get_false_posi_roi(pcd_in_roi, gt_num, num_thresh = 5, dens_thresh = 50):  
    n = -1
    dbs_dict = {}
    for i in range(gt_num, len(pcd_in_roi)):
        dbs_dict[i] = {}
        cluster_labels = np.array(pcd_in_roi[i][1].cluster_dbscan(eps=0.2, 
                                                            min_points=15, 
                                                            print_progress=False))
        if len(cluster_labels) == 0:
            num_clusters = 0
        else:
            num_clusters = cluster_labels.max() + 1

        dbs_dict[i]['num_of_clusters'] = num_clusters
        #print(f"Point cloud has {num_clusters} clusters")
        unique, count_ = np.unique(cluster_labels, return_counts=True)
        dict_clusters = dict(zip(unique, count_))

        if n in dict_clusters.keys():
            val_ = dict_clusters.pop(-1)

            
        if num_clusters > 0:
            sum_clusters_pts = sum(dict_clusters.values())   
            average_dens = sum_clusters_pts/num_clusters
            dbs_dict[i]['average_density'] = average_dens
        else:
            dbs_dict[i]['average_density'] = 0


    
    #print(dbs_dict)
    fp_ = []
    for i in dbs_dict.keys():
        if (dbs_dict[i]['num_of_clusters'] == 0 or 
            dbs_dict[i]['num_of_clusters'] > num_thresh or 
            dbs_dict[i]['average_density'] < dens_thresh):
            fp_.append(i)

    return fp_

def get_false_posi_iou(gt, pred, cfg):
    overlaps = cfg.pipeline.get("overlaps", [0.5]) 
    similar_classes = cfg.pipeline.get("similar_classes", {})
    difficulties = cfg.pipeline.get("difficulties", [0])
    classes = cfg.model.get("classes")
    
    d, f = metrics.precision_3d(pred[0], gt[0], classes, 
                                difficulties, overlaps, 
                                bev=False, 
                                similar_classes = similar_classes)
                                
    iou = metrics.iou_3d(pred[0]['bbox'].astype(np.float32), 
                         gt[0]['bbox'].astype(np.float32))
    

    iou_dict = {}
    for i in range(len(iou)):
        score = []
        idx = i + len(iou[i])
        iou_dict[idx] = {}
        if all(v == 0 for v in iou[i]):
            score.append(0)
            iou_dict[idx]['score'] = score

        else:
            for j in range(len(iou[i])):
                if iou[i][j] != 0:
                    score.append(iou[i][j])
            iou_dict[idx]['score'] = score

    fp_ = []
    for keys in iou_dict.keys():
        for v in iou_dict[keys].values():
            if all(v_ <= 0.1 for v_ in v):
                fp_.append(keys)
        

    return fp_

    
        
