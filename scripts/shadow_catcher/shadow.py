import numpy as np
import open3d as o3d
from math import sqrt

from polygon import *


MAX_LIDAR = 80
MAX_HEIGHT = 1.73
GROUND_LVL = -1.73

# Reference Lidar Coord
LIDAR_REF = [0, 0, 0]

def get_bev_projection(objects, obj_coords):
    """
    Get Max & Min Bounding Line and Start & End Line
                      /|
                     / |
                    /  |                                    
                   /|  |
                  / |  |
                 /--|--|--> Max Bounding Line 
                /   |  |
          \    /____|  |
           \  /|    |  |--> End Line
            \/ |    |  |
  Lidar <---|  |BOX | -|--> Shadow Region (ROI)
 [0,0,0]    /\ |    |  |
           /  \|____|--|--> Start Line    
          /    \    |  |   
                \   |  | 
                 \--|--|--> Min Bounding Line   
                  \ |  |
                   \|  |
                    \  |
                     \ |
                      \|
    
    Args:
        objects: Dictionary of 3D boxes info [x,y,z,h,l,w,r]
        obj_coords [8x3] np.array: Each corner of the bounding box
    Return:
        bev_projection: dictionary of each box
            {box idx: {
                        min_grad_line: type(< Line >)
                        max_grad_line: type(< Line >)
                        max_dist_point_idx (int): index of maximum point
                        min_grad_points [2x3] np.array: start and end points of line
                        max_grad_points [2x3] np.array: start and end points of line
                        starting_line [2x3] np.array: start and end points of line
                        ending_line [2x3] np.array: start and end points of line
                        length (int): shadow_length value
                    }}
    """ 
    bev_projection = {}
    obj_idx = 0
    for i in range(len(obj_coords)):
        coords_ = obj_coords[i]
        coord_idx = 0
        min_grad = 0
        min_grad_idx = 0 
        max_grad = 0
        max_grad_idx = 0
        max_dist = 0 
        max_dist_idx = 0
        for list_ in coords_:
            bev_projection[obj_idx] = {}
            grad_ = list_[1] / list_[0]
            if coord_idx == 0:
                min_grad = grad_
                min_grad_idx = coord_idx
                max_grad = grad_
                max_grad_idx = coord_idx
                max_dist = sqrt(list_[0]**2 + list_[1]**2)
                max_dist_idx = coord_idx
                coord_idx += 1
            else:
                dist_ = sqrt(list_[0]**2 + list_[1]**2)
                if dist_ >= max_dist:
                    max_dist_idx = coord_idx
                
                if grad_ >= max_grad:
                    max_grad = grad_
                    max_grad_idx = coord_idx
                    coord_idx += 1
                elif grad_ <= min_grad:
                    min_grad = grad_
                    min_grad_idx = coord_idx
                    coord_idx += 1
                else:
                    coord_idx += 1
                    continue
        
        bev_projection[obj_idx]['min_grad_line'] = Line(min_grad, c_ = 0, 
                                                    idx_ = min_grad_idx)
        bev_projection[obj_idx]['max_grad_line'] = Line(max_grad, c_ = 0, 
                                                    idx_ = max_grad_idx)
        bev_projection[obj_idx]['max_dist_point_idx'] = max_dist_idx
        obj_idx += 1
    

    # Length Of lines
    for i in objects.keys():
        obj_height = objects[i]['3d_box'][3] # index 3 is height 

        shadow_length = 0
        idx = 0
        max_idx = 0
        for row_ in obj_coords[i]:
            # Use Max Shadow Length
            if obj_height >= MAX_HEIGHT:
                shadow_length = MAX_LIDAR
                max_idx = idx
                idx += 1
            else:
                length_ = (sqrt(row_[0]**2 + row_[1]**2) * 
                            ((MAX_HEIGHT - abs(row_[2])) / 
                            (abs(row_[2]))))
                if idx == 0:
                    shadow_length = length_
                    idx += 1
                elif length_ >= shadow_length:
                    shadow_length = length_
                    max_idx = idx
                    idx += 1
                else:
                    idx += 1

        max_pt = obj_coords[i][max_idx]
        if shadow_length + sqrt(max_pt[0]**2 + max_pt[1]**2) > MAX_LIDAR:
            shadow_length = MAX_LIDAR - sqrt(max_pt[0]**2 + max_pt[1]**2)
        ratio = 1 + (shadow_length / sqrt(max_pt[0]**2 + max_pt[1]**2))
        max_shadow_pt = ((ratio * max_pt[0]), (ratio * max_pt[1]))
        grad = (max_pt[1] / max_pt[0]) # Gradient line thru min point
        grad2 = -1 / grad # Perpendicular Gradient thru min point
        c_perpen = max_shadow_pt[1] - (grad2 * max_shadow_pt[0]) # c = y - mx (for Perpendicular)
        line_perpen = Line(grad2, c_perpen, None)
        max_grad_end = bev_projection[i]['max_grad_line'].get_line_intersect_coords(line_perpen)
        min_grad_end = bev_projection[i]['min_grad_line'].get_line_intersect_coords(line_perpen)

        # Gradient of Line that intersects min & max shadow bounds
        min_grad_idx = bev_projection[i]['min_grad_line'].get_idx() 
        max_grad_idx = bev_projection[i]['max_grad_line'].get_idx()
        grad_shadow_min_max = ((obj_coords[i][max_grad_idx][1] - 
                                obj_coords[i][min_grad_idx][1]) /
                                (obj_coords[i][max_grad_idx][0] - 
                                obj_coords[i][min_grad_idx][0]))
        max_dist_pt_idx = bev_projection[i]['max_dist_point_idx']
        c_line_max = (obj_coords[i][max_dist_pt_idx][1] - 
                        (grad_shadow_min_max * 
                        obj_coords[i][max_dist_pt_idx][0]))
        line_start = Line(c_=c_line_max, m_=grad_shadow_min_max)


        # Ground Level
        min_z = GROUND_LVL

        # Intersection Points
        p1 = bev_projection[i]['max_grad_line'].get_line_intersect_coords(line_start)
        p2 = bev_projection[i]['min_grad_line'].get_line_intersect_coords(line_start)
        
        #
        bev_projection[i]['min_grad_points'] = [
                                                LIDAR_REF, 
                                                [min_grad_end[0], 
                                                min_grad_end[1], 
                                                min_z]
                                            ]
        bev_projection[i]['max_grad_points'] = [
                                                LIDAR_REF,
                                                [max_grad_end[0], 
                                                 max_grad_end[1],
                                                 min_z]
                                            ]
        bev_projection[i]['starting_line'] = [
                                            [p1[0],p1[1], min_z], 
                                            [p2[0],p2[1], min_z]
                                        ]
        bev_projection[i]['ending_line'] = [
                                            [min_grad_end[0],
                                             min_grad_end[1], 
                                             min_z], 
                                            [max_grad_end[0],
                                             max_grad_end[1],
                                             min_z]
                                        ]
        bev_projection[i]['length'] = shadow_length


    return bev_projection

def get_roi(obj_coords, bev_proj, pcd):
    """
    Splits BEV projection into two regions and converts into 
    3D Shadow Region by using uniform height above ground (0.2m):
                      /|
                     / |
                    /  |                                    
                   /|  |
                  / |  |
                 /  |  | 
                /   |  |
               /    |  |
              /     |  |
             /      |  |
  roa_2<----|---    | -|--> roa_
             \      |  |
              \     |  |    
               \    |  |   
                \   |  | 
                 \  |  |  
                  \ |  |
                   \|  |
                    \  |
                     \ |
                      \|
    
    From FV Shadow Region of Interest (Shadow Region in 3D will contain 8 Vertices & 6 Faces):

        roi[7]<---- _____________________ ---->roi[6]
                   |\                   /|  
  roi[2]<------------\÷_______________÷/--------------->roi[3]  
                   |__|_______________|__|
      roi[5]<-------\ |               | /------>roi[4]    
                     \|_______________|/
          roi[0]<-----÷               ÷----->roi[1]         

                            LIDAR
    Args:
        obj_coords [8x3] np.array: Each corner of the bounding box
        bev_proj: Dictionary of each boxes BEV projection lines
        pcd: Point Cloud Data
    Return:
        polygons: (roa_, roa_2) PCD in each region
        roi [8x3] (np.array): For each box
    """ 
    polygons = []
    roi = [] 

    for i in range(len(obj_coords)):
        max_ = np.max(obj_coords[i], axis=0)
        max_z = max_[2]
        min_ = np.min(obj_coords[i], axis=0)
        min_z = min_[2]
        min_z = GROUND_LVL

        array_ = np.asarray([
                            bev_proj[i]['starting_line'][0],
                            bev_proj[i]['starting_line'][1],
                            [bev_proj[i]['starting_line'][0][0], 
                             bev_proj[i]['starting_line'][0][1],max_z],
                            [bev_proj[i]['starting_line'][1][0], 
                             bev_proj[i]['starting_line'][1][1],max_z],
                            bev_proj[i]['ending_line'][0], 
                            bev_proj[i]['ending_line'][1],
                            [bev_proj[i]['ending_line'][0][0], 
                             bev_proj[i]['ending_line'][0][1],min_z+0.001],
                            [bev_proj[i]['ending_line'][1][0], 
                            bev_proj[i]['ending_line'][1][1],min_z+0.001]
                        ]).astype("float64")

        roi.append(array_)

        # Get Shadow Region Polygon Vol
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = 'Y'
        max_ = np.max(array_, axis=0)
        vol.axis_max = max_[1]
        min_ = np.min(array_, axis=0)
        vol.axis_min = min_[1]
        vol.bounding_polygon = o3d.utility.Vector3dVector(array_)
        roa_ = vol.crop_point_cloud(pcd)

        boxes = o3d.geometry.OrientedBoundingBox()
        boxes = boxes.create_from_points(o3d.utility.
                                            Vector3dVector(obj_coords[i]))

        # Get Object Region Polygon Vol
        vol2 = o3d.visualization.SelectionPolygonVolume()
        vol2.orthogonal_axis = 'Y'
        array_2 = np.asarray(boxes.get_box_points())
        max_2 = np.max(array_2, axis=0)
        vol2.axis_max = max_2[1]
        min_2 = np.min(array_2, axis=0)
        vol2.axis_min = min_2[1]
        vol2.bounding_polygon = boxes.get_box_points()
        roa2_ = vol2.crop_point_cloud(pcd)

        polygons.append((roa_, roa2_))

        # Check if roa2_ is box objects:
        bbox_new = roa2_.get_axis_aligned_bounding_box()

    return polygons, roi


def get_shadow_lines(bev_proj):
    """
    Gets Shadow Lines, truncating object region in 3D
    (i.e. FV roi seen above)
    Args:
        bev_proj: Dictionary of each boxes BEV projection lines
    Return:
        obj_line: Dictionary of each box 3D Shadow Lines
        {box idx: {
                    midline: type(< Line >)
                    min_grad: type(< Line >)
                    max_grad: type(< Line >)
                    start_line : type(< Line >)
                    end_line: type(< Line >) 
                }}
    """
    obj_lines = {}
    for obj_id in bev_proj.keys() :
        obj_lines[obj_id] = {} 
        
        start_ = bev_proj[obj_id]['starting_line']
        
        # Mid point from start line (front plane)
        p1 = [(start_[0][0] + start_[1][0])/2, 
              (start_[0][1] + start_[1][1])/2, 
              (start_[0][2] + start_[1][2])/2]

        end_ = bev_proj[obj_id]['ending_line']
        # Mid point from end line (back plane)
        p2 = [(end_[0][0] + end_[1][0])/2, 
              (end_[0][1] + end_[1][1])/2, 
              (end_[0][2] + end_[1][2])/2]
        # Mid point line
        grad = (p2[1] - p1[1])/(p2[0] - p1[0])
        c = p1[1] - (grad*p1[0])
        obj_lines[obj_id]['midline'] = Line(m_ = grad, c_ = c)

        obj_lines[obj_id]['min_grad'] = bev_proj[obj_id]['min_grad_line']
        obj_lines[obj_id]['max_grad'] = bev_proj[obj_id]['max_grad_line']

        # Truncated start and end line
        start_line_coords = bev_proj[obj_id]['starting_line']
        end_line_coords = bev_proj[obj_id]['ending_line']
        start_line_m = ((start_line_coords[0][1] - start_line_coords[1][1]) /
                        (start_line_coords[0][0] - start_line_coords[1][0]))
        start_line_c = (start_line_coords[0][1] - 
                        (start_line_m * start_line_coords[0][0]))
        obj_lines[obj_id]['start_line'] = Line(m_ = start_line_m, 
                                                c_ = start_line_c)
        end_line_m = ((end_line_coords[0][1] - end_line_coords[1][1]) / 
                      (end_line_coords[0][0] - end_line_coords[1][0]))
        end_line_c = end_line_coords[0][1] - (end_line_m*end_line_coords[0][0])
        obj_lines[obj_id]['end_line'] = Line(m_ = end_line_m, 
                                                c_ = end_line_c)
    

    return obj_lines

def count_pts_in_roi(pcd, roi, verbose = False):
    """
    Counts the number of points within the Shadow (ROI)
    Args:
        pcd: Point Cloud Data within ROI
        roi: [8x3] np.array ROI coordinate points for each box
        verbose: Screen Dump number of Points Analysed
    Return
        count_dict: dictionary {keys - box id: value - count}
    """
    count_dict = {}
    for idx in range(len(roi)):
        count_dict[idx] = {}
        pts_in_region = pcd[idx][0]  
        roi_ = roi[idx]
        front_plane = Face([Vector(roi_[0]), Vector(roi_[1]), 
                            Vector(roi_[3]), Vector(roi_[2])])
        back_plane = Face([Vector(roi_[5]), Vector(roi_[7]), 
                           Vector(roi_[6]), Vector(roi_[4])])
        left_plane = Face([Vector(roi_[0]), Vector(roi_[2]), 
                           Vector(roi_[7]), Vector(roi_[5])])
        right_plane = Face([Vector(roi_[1]), Vector(roi_[4]), 
                            Vector(roi_[6]), Vector(roi_[3])])
        top_plane = Face([Vector(roi_[2]), Vector(roi_[3]), 
                          Vector(roi_[6]), Vector(roi_[7])])
        bottom_plane = Face([Vector(roi_[0]), Vector(roi_[5]), 
                             Vector(roi_[4]), Vector(roi_[1])])
        
        shadow_polygon = [front_plane, back_plane, left_plane, 
                          right_plane, top_plane, bottom_plane]

        total_num_pts = 0
        num_pts_shadow = 0
        pts_in_shadow = []
        for pts in np.asarray(pts_in_region.points):
            total_num_pts += 1
            if isInPoly(pts, shadow_polygon):
                num_pts_shadow += 1
                pts_in_shadow.append(pts)
            else:
                continue
        count_dict[idx]['total_analysed'] = total_num_pts
        count_dict[idx]['total_in_shadow'] = num_pts_shadow
        count_dict[idx]['pts_in_shadow'] = pts_in_shadow

        if verbose:
            print("Object : %i" %idx)
            print((" Total number of points: \n  -- Analysed : %i" %(total_num_pts)  + 
                "\n  -- In Shadow Region : %i " %num_pts_shadow))
            print("")

    return count_dict


def get_draw_lines(obj_coords, bev_proj):
    """
    Gets BEV Shadow Lines With Reference to Lidar Reference Position
    (i.e. BEV Triangle with 3 vertices)
    Args:
        obj_coords: Box Corners [8x3] np.array in Velodyne Coordinates
        bev_proj: Lines defining BEV projection
    Return:
        boxes: List of o3d.geometry.OrientedBoundingBox
        lines_: List of o3d.geometry.LineSet 
    """
    boxes = []
    lines_ = []
    for obj_ in range(len(obj_coords)):
        # print(obj_)
        # draw bounding boxes
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_coords[obj_])
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox = bbox.create_from_points(
            o3d.utility.Vector3dVector(obj_coords[obj_]))
    
        boxes.append(bbox)
        
        # visualize shadow
        points = [bev_proj[obj_]['min_grad_points'][0], bev_proj[obj_]['min_grad_points'][1],
                  bev_proj[obj_]['max_grad_points'][0], bev_proj[obj_]['max_grad_points'][1],
                  bev_proj[obj_]['starting_line'][0], bev_proj[obj_]['starting_line'][1],
                  bev_proj[obj_]['ending_line'][0], bev_proj[obj_]['ending_line'][1]]
        lines = [[0, 1], [2, 3], [4, 5], [6, 7]]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                        lines=o3d.utility.Vector2iVector(lines))

        lines_.append(line_set)

    return boxes, lines_