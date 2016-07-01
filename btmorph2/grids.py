"""
File contains:

     - :class:`VoxelGrid`

Irina Reshodko
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class VoxelGrid(object) :
    """
    Represents voxelized 3D model of an object with given dimensions and resolution
    Dimensions: real dimensions of an object in micrometers
    Resolution: resolution in voxels
    """
    def __str__(self):
        return "VoxelGrid, dimensions=" + str(self.dim) + ",resoultion=" + str(self.res) + ",size=" + \
            str(len(self.grid)) + ", encompassing box=" +str(self.encompassingBox) + \
            ", voxel dimension:=" + str(self.dV) + ", total volume=" + str(len(self.grid)*(self.dV**3))\
            + ", offset=" +str(self.offset)
            
    @staticmethod
    def check_key(dims, key):
        """
        Check key type and range
        """
        if not isinstance(key, tuple) :
            raise TypeError("The key must be a tuple of 3 integers")
        if len(key) < 3:
            raise TypeError("The key must be a tuple of 3 integers")        
        (x,y,z) = key
        if not (isinstance(x, int) and isinstance(y, int) and isinstance(z, int)):
            raise TypeError("The key must be a tuple of 3 integers") 
        if (x < 0 or x > dims[0]):
            raise IndexError("Index is out of range:"  + str(key))
        if (y < 0 or y > dims[1]):
            raise IndexError("Index is out of range:"+ str(key))
        if (z < 0 or z > dims[2]):
            raise IndexError("Index is out of range:" + str(key))
        return True
        
    def __getitem__(self, key):
        """
        Right [] operator overload
        """
        VoxelGrid.check_key(self.res, key)
        if not key in self.grid:
            return False
        else:
            return True
            
    def __setitem__(self, key, value):
        """
        Left [] operator overload
        """
        VoxelGrid.check_key(self.res, key)
        if not isinstance(value, bool):
            raise TypeError("The value must be boolean")
        if key in self.grid and value == False:
            del self.grid[key]
        elif value == True:
            for i in range(0,3):
                if key[i] < self.encompassingBox[i][0]:
                    self.encompassingBox[i][0] = key[i]
                if key[i] > self.encompassingBox[i][1]:
                    self.encompassingBox[i][1] = key[i]            
            self.grid[key] = value
        
    def __init__(self, dimensions, resolution, tree = None):
        """
        Generate a voxel grid for given dimensions and resolution
        Note: the dimensions ratio (x:y:z) must be the same as resolution ratio (rx:ry:rz)
        If this is not the case, the dimensions will be expanded to meet this criteria
        
        Parameters
        ----------
        dimensions : numpy.array
        The grid's real dimensions
        resolution : array(int)
        The grid's resolution (number of voxels in each dimension). Must be a power of two
        """
        if not (len(dimensions) == 3 and len(resolution) == 3):
            raise TypeError("Dimensions and resolution must be number iterables of length 3")
        for i in range(0,3):
            if not VoxelGrid.is_power_of_two(resolution[i]):
                raise IndexError("Resolution must be power of 2")
        dimensions = [dimensions[0],dimensions[1], dimensions[2]]
        self.dim = VoxelGrid.adjust_dimensions(dimensions, resolution)
        self.res = resolution
        self.grid = {}
        self.dV = self.dim[0]/float(self.res[0])
        self.encompassingBox = [[], [], []]
        self.encompassingBox[0] = [self.res[0], 0]
        self.encompassingBox[1] = [self.res[1], 0]
        self.encompassingBox[2] = [self.res[2], 0]
        self.offset = (0,0,0)
        self.add_tree(tree)
        
    @staticmethod    
    def adjust_dimensions(dimensions, resolution):
        """
        Adjusts the grid dimensions(x,y,z) in such a way so their ratio is the same as resolution (rx,ry,rz) ratio.
        x:y:z = rx:ry:rz
        
        Parameters
        ----------
        dimensions : numpy.array
        The grid's real dimensions
        resolution : array(int)
        The grid's resolution (number of voxels in each dimension). Must be a power of two
        
        Returns
        ----------
        New dimensions :  numpy.array
        An expanded (if neccessary) dimensions
        """
        if not (len(dimensions) == 3 and len(resolution) == 3):
            raise TypeError("Dimension and resolution must be number iterables of length 3")
        for i in range(0,3):
            if not (dimensions[i] >= 0 and resolution[i] >= 0):
                raise IndexError("Dimensions and resolution must be positive")
        # Check if all dimensions match
        # Is more than one dimension and/or resolution is zero?
        if (resolution.count(0) > 1 or (len(dimensions) - np.count_nonzero(dimensions)) > 1):
            return None
        # Is there a case where dimension/resolution is zero but not both of them are zero?
        for i in range(0, 3):
            if resolution[i]*dimensions[i] == 0 and resolution[i]+dimensions[i] != 0 and resolution[i] != 1:
                return None
        [x,y,z] = dimensions
        [x_new,y_new,z_new] = [x,y,z]
        [rx,ry,rz] = resolution
        if x > y * float(rx)/float(ry):
            y_new = x * float(ry)/float(rx)
            if z > x * float(rz)/float(rx):
                x_new = z * float(rx)/float(rz)
                y_new = x_new * float(ry)/float(rx)
            else:
                z_new = x * float(rz)/float(rx)
        else:
            x_new = y * float(rx)/float(ry)
            if z > y * float(rz)/float(ry):
                y_new = z * float(ry)/float(rz)
                x_new = y_new * float(rx)/float(ry)
            else:
                z_new = y * float(rz)/float(ry)  
        return [x_new,y_new,z_new]
    
    @staticmethod
    def is_power_of_two(int_num) :
        """
        Checks if the number is a power of two
        
        Parameters
        ----------
        int_num : int
        Input number
        
        Returns
        ----------
        True if N=2^m and False otherwise
        """
        return isinstance(int_num, int) and int_num > 0 and (int_num & (int_num - 1) == 0)
        
    def plot(this):
        """ 
        Plot the grid as a scattered 3d plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        keys = this.grid.keys();
        xs = map(lambda (x,y,z): x, keys)
        ys = map(lambda (x,y,z): y, keys)
        zs = map(lambda (x,y,z): z, keys)
        ax.scatter(xs, ys, zs, zdir="z")
    
    def calc_encompassing_box_sphere(self, center, radius):
        """
        Calculate encompassing box for a sphere of the given radius and center
        
        Parameters
        ------------
        center : array or tuple of numbers (real dimensions)
        The center of the sphere
        radius : number (real dimension)
        The sphere's radius
        
        Returns
        ------------
        Array of ranges (mapped to resolution) for x, y and z: [(x1,x2), (y1,y2), (z1,z2)]
        or None if there is no intersection between the sphere and the grid
        """
        if radius < 0:
            return None
        if radius == 0:
            c = self.dimension_to_voxel(center)
            return [(c[0], c[0]), (c[1], c[1]), (c[2], c[2])]
        ranges = [0, 0, 0]
        for i in range(0,3):
            ranges[i] = (int(round((center[i] - radius)/self.dV)), int(round((center[i] + radius)/self.dV)))
            if ranges[i][0] > self.res[i] and ranges[i][1] > self.res[i] or ranges[i][0] < 0 and ranges[i][1] < 0:
                return None
            ranges[i]= (max(ranges[i][0], 0), min(ranges[i][1], self.res[i]))
        return ranges
    
    def falls_into_sphere(self, point, center, radius):
        """
        Check if the point falls into the sphere of given radius and center
        
        Parameters
        ------------
        point : coordinates of the point of interest (voxel coordinates)
        center : array or tuple of numbers (real dimensions)
        The center of the sphere
        radius : number (real dimension)
        The sphere's radius
        
        Returns:
        True if the point falls within the sphere and False otherwise
        """
        if radius < 0:
            return False
        if radius == 0:
            center_vox = self.dimension_to_voxel(center)
            return bool(center_vox == point)
        s = 0
        for i in range(0, 3):
            s += (point[i]*self.dV - center[i])**2
        return bool(s <= radius**2)
        
    def falls_into_frustum(self, point, center1, radius1, center2, radius2):
        """
        Check if the point falls into the frustum with given radii and centers
        
        Parameters
        ------------
        point : coordinates of the point of interest (voxel coordinates)
        center1 : array or tuple of numbers (real dimensions)
        The center of the first base
        center2 : array or tuple of numbers (real dimensions)
        The center of the second base
        radius1 : number (real dimension)
        Radius of the first base
        radius2 : number (real dimension)
        Radius of the second base
        
        Returns:
        True if the point falls within the frustum and False otherwise
        """
        if radius1 < 0 or radius2 < 0:
            return False
        point = self.voxel_to_dimension(point)
        #voxel_c = point
        point = (point[0] - center1[0], point[1] - center1[1], point[2] - center1[2])
        abs_p = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        if center1 == center2:
            return point[2] == 0 and abs_p <= max(radius1,radius2)
        a = (center2[0]-center1[0], center2[1]-center1[1], center2[2]-center1[2])
        if abs_p == 0 or point == a:
            return True
        abs_a = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        n = (a[0]/abs_a, a[1]/abs_a, a[2]/abs_a)
        dot_pn = point[0]*n[0] + point[1]*n[1] + point[2]*n[2]        
        l = dot_pn
        if l < 0 or l > abs_a:
            return False
        epsilon = 0.0001
        c = dot_pn/abs_p
        if abs(c - 1) < epsilon or abs(c + 1) < epsilon:
            c = 1.0
        s = math.sqrt(1 - c**2)
        proj_plane = abs_p * s
        r = l * (radius2-radius1)/abs_a + radius1
        # One of the points falls into the voxel
        fiv = False#center1[0] - voxel_c[0] < self.dV or center1[1] - voxel_c[1] < self.dV or center1[2] - voxel_c[2] < self.dV
        # Voxel falls in frustum or frustum falls intro voxel        
        return proj_plane <= r or fiv
    
    def calc_encompassing_box_frustum(self, center1, radius1, center2, radius2):
        """
        Calculate encompassing box for a frustum (cut cone) with given base centers and radii
        
        Parameters
        -----------
        center1 : tuple of 3 numbers 
        Center of the first base
        radius1 : number
        Radius of the first base
        center2 : tuple of 3 numbers 
        Center of the second base
        radius12 : number
        Radius of the second base 
        
        Returns
        -----------
        List of ranges for each axis (in voxels)
        [(x1,x2), (y1,y2), (z1,z2)]
        """
        if radius1 == None or radius2 == None or center1 == None or center2 == None:
            return None
        if radius1 < 0 or radius2 < 0 :
            return None
        (x1,y1,z1) = center1
        (x2,y2,z2) = center2
        r1 = radius1
        r2 = radius2
        rangeX = (max(min(x1-r1, x2-r2), 0), min(max(x1+r1,x2+r2),self.dim[0]))
        rangeY = (max(min(y1-r1, y2-r2), 0), min(max(y1+r1,y2+r2),self.dim[1]))
        rangeZ = (max(min(z1-r1, z2-r2), 0), min(max(z1+r1,z2+r2),self.dim[2]))
        rangeX = (int(round(rangeX[0]/self.dV)), int(round(rangeX[1]/self.dV)))
        rangeY = (int(round(rangeY[0]/self.dV)), int(round(rangeY[1]/self.dV)))  
        rangeZ = (int(round(rangeZ[0]/self.dV)), int(round(rangeZ[1]/self.dV)))
        return [rangeX, rangeY, rangeZ]
    
       
    def add_frustum(self, center1, radius1, center2, radius2):
        """
        Adds a voxelized filled frustum of the given radii and base centers to the grid
        
        Parameters
        ------------
        center1 : array or tuple of numbers (real dimensions)
        The center of the first base
        radius1 : number (real dimension)
        The first base's radius
        center2 : array or tuple of numbers (real dimensions)
        The center of the second base
        radius2 : number (real dimension)
        The second base's radius
        """
        if radius1 < 0 or radius2 < 0:
            return
        # Filter out frustums completely out of grid
        if min(center1[0] - radius1, center2[0] - radius2) > self.dim[0] or\
           min(center1[1] - radius1, center2[1] - radius2) > self.dim[1] or\
           min(center1[2], center2[2]) > self.dim[2] or\
           max(center1[0] + radius1, center2[0] + radius2) < 0 or\
           max(center1[1] + radius1, center2[1] + radius2) < 0 or\
           max(center1[2], center2[2]) < 0:
               return
        # Calculate encompassing box
        ranges = self.calc_encompassing_box_frustum(center1, radius1, center2, radius2)
        if ranges == None:
            return
        [(x1,x2), (y1,y2), (z1,z2)] = ranges
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                for z in range(z1, z2+1):                    
                    if self.falls_into_frustum((x,y,z), center1, radius1, center2, radius2):
                        self[(x,y,z)] = True
    
    def add_sphere(self, center, radius):
        """
        Adds a voxelized filled sphere of the given radius and center to the grid
        
        Parameters
        ------------
        center : array or tuple of numbers (real dimensions)
        The center of the sphere
        radius : number (real dimension)
        The sphere's radius
        """
        ranges = self.calc_encompassing_box_sphere(center, radius)
        if ranges == None:
            return
        [(x1,x2), (y1,y2), (z1,z2)] = ranges
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                for z in range(z1, z2+1):
                    self[(x,y,z)] = self.falls_into_sphere((x,y,z), center, radius)
    
    def add_tree(self, tree):
        """
        Voxelize the whole tree
        
        Parameters
        ------------
        tree : STree2
        A tree to be voxelized
        """
        # If tree == None => do nothing
        if None == tree:
            return
        nodes = tree.get_nodes()
        if None == nodes or len(nodes) == 0:
            return
        # point with min x y and z
        minX = self.dim[0]
        minY = self.dim[1]
        minZ = self.dim[2]
        for node in nodes:
            p = node.content['p3d']
            (x,y,z) = tuple(p.xyz)
            if x < minX:
                minX = x
            if y < minY:
                minY = y
            if z < minZ:
                minZ = z
        self.offset = (minX, minY, minZ)
        # Add soma as sphere
        p = tree.get_node_with_index(1).content['p3d']
        r = p.radius
        (x,y,z) = tuple(p.xyz)
        center = (x - minX, y - minY, z - minZ)
        self.add_sphere(center, r)
        # Add all segments
        for node in nodes:
            p = node.content['p3d']
            (x,y,z) = tuple(p.xyz)
            center1 = (x - minX, y - minY, z - minZ)
            r1 = p.radius
            pNode = node.parent
            if pNode == None:
                continue
            parentP = pNode.content['p3d']
            (x,y,z) = tuple(parentP.xyz)
            center2 = (x - minX, y - minY, z - minZ)
            r2 = parentP.radius
            self.add_frustum(center1, r1, center2, r2)
            
    def voxel_to_dimension(self, point):
        """
        Converts voxel coordinates to dimension coordinates
        
        Parameters
        ------------
        point : tuple of 3 numbers
        A point to convert
        
        Returns
        ------------
        Coordinates in real dimension values (micrometers)
        """
        if point == None:
            return None
        return (point[0]*self.dV, point[1]*self.dV, point[2]*self.dV)
        
    def dimension_to_voxel(self, point):
        """
        Converts real dimension coordinates to voxel coordinates
        
        Parameters
        ------------
        point : tuple of 3 numbers
        A point to convert
        
        Returns
        ------------
        Voxel coordinates
        """
        if point == None:
            return None
        return (int(round(point[0]/self.dV)), int(round(point[1]/self.dV)), int(round(point[2]/self.dV)))

