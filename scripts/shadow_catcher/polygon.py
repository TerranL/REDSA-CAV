import numpy as np

class Line(object):
    """
    Line class
        y = mx + c
    """
    def __init__(self, m_, c_ = 0, idx_ = None) :
        self.C = c_
        self.M = m_
        self.idx = idx_

    def get_c(self) :
        return self.C
    
    def get_m(self):
        return self.M
    
    def get_idx(self):
        return self.idx
    
    def get_y(self, x_) :
        return ((self.M)*x_ + self.C)
    
    def get_x(self,y_) :
        return ((y_ - self.C)/self.M)
    
    def get_orth_grad(self) :
        return -(1/self.M)
    
    def get_line_intersect_coords(self, line_) :
        x = (line_.get_c() - self.C)/(self.M - line_.get_m())
        y = self.get_y(x)
        return (x,y)
    
    def get_dist_to_point(self, x_ , y_) :
        ## line : ax + by + c = 0 ( a = M , b = -1 )
        ## point : x_ , y_
        ## distance = |a*x_ + b*y_ + c | / sqrt(a^2 + b^2)
        numerator  = np.absolute((self.M)*x_ - y_ + self.C)
        denominator = np.sqrt((self.M)**2 + (-1)**2)
        return (numerator/denominator)

class Vector(object) :
    """
    Vector Class
    """
    def __init__ (self, array_) :
        self.x = array_[0]
        self.y = array_[1]
        self.z = array_[2]
        self.array = array_
        
    def minus(self, vector_):
        array_ = [(self.x-vector_.x), (self.y-vector_.y), (self.z-vector_.z)]
        return Vector(array_)
    
    def add(self, vector_):
        array_ = [(self.x+vector_.x), (self.y+vector_.y), (self.z+vector_.z)]
        return Vector(array_)
    
    def multi(self, l):
        array_ = [(l*self.x), (l*self.y), (l*self.z)]
        return Vector(array_)
    
    def div(self, l):
        array_ = [(self.x)/l, (self.y)/l, (self.z)/l]
        return Vector(array_)
    
    def cross(self, vector_) :
        return Vector(np.cross(self.array, vector_.array))
    
    def dot(self, vector_) :
        return np.dot(self.array, vector_.array)
    
    def norm(self) :
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def unit_vec(self):
        return self.array / np.linalg.norm(self.array)
    
    def dist_to_pt(self, array_) :
        v2 = Vector(array_)
        temp = v2.cross(self)
        return temp.norm()/self.norm()

##ref:  https://stackoverflow.com/questions/8877872/determining-if-a-point-is-inside-a-polyhedron
class Face(object):
    """
    Face of object (Plane) class
     _______
    |       |
    |_______|
    """
    def __init__(self, array_vectors) :
        self.vectors = array_vectors
    
    def normal(self) :
        dir1 = self.vectors[1].minus(self.vectors[0])
        dir2 = self.vectors[2].minus(self.vectors[0])
        n = dir1.cross(dir2)
        d = n.norm()
        array_ = [n.x/d, n.y/d, n.z/d]
        return Vector(array_)
            
def isInPoly(point_, array_ ):
    point = Vector(point_)
    for face in array_:
        p2f = face.vectors[0].minus(point)
        d = p2f.dot(face.normal())
        d /= p2f.norm()
        bound = 1e-15 # -1e-15 for pts on boundary
        if d < bound :
            return False
        
    return True