import numpy as np
import pybiomech as bm
import scipy.stats as stat
from scipy.optimize import minimize
from . import viewer




class Segment():

    def __init__(self, mass, moment_of_inertia, orientation,
                 position = np.array([0.0, 0.0, 0.0]), landmarks = None,
                 standard_dimensions = None, mesh = None, name = None):
        """
        This instantiates a Segment object, which contains the mass, moment of inertia, orientation, and landmarks required
        to manage the object.

        Input:              mass    :   This is the linear mass of the segment.
                moment of inertia   :   This is the moment of inertia tensor of the object.
                orientation         :   This is the initial orientation matrix of the segment.
                landmarks           :   A dictionary of landmark names, as well as their (x,y,z) coordinates in
                                        local coordinates.
                standard_dimensions :   An array of 3-tuples which name two landmarks as well as the average distance
                                        between them with the standard deviation. For example:
                                            sd_dim = [('p1', 'p2', [mean, sd]), ('p3', 'p4', [mean, sd])]
                                        Each individual dimension is in the form ('point1', 'point2', [mean, sd]).
                                        These are used to scale the model
                mesh                :   This is a triangular mesh object (defined in the viewer class).
        :return:
        """
        print("Initializing Segment named " + name)
        self.name = name
        self.mass = mass
        self.I = moment_of_inertia
        self.R = np.eye(3)
        self.joint_euler_angles = np.array([0.0, 0.0, 0.0])

        self.U = orientation
        self.position = position
        self.landmarks = landmarks
        self.std_dims = standard_dimensions
        self.mesh = mesh

        self.list_of_children = []
        #self.point_of_rotation = np.array([0.0, 0.0, 0.0])
        self.parent = None
        self.parent_landmark = None


    def rotmtrx(self):
        """
        Purpose: Returns the rotation matrix of the local orientation
        :return:
        """
        return bm.angle2dcm(self.joint_euler_angles[0], self.joint_euler_angles[1], self.joint_euler_angles[2])


    def draw(self):
        """
        Purpose: Calculates the position and orientation of the rigid body, and draws it in the GFXwindow.
                Note: There must be an active GFXWindow for this to work right!!
        :return:
        """
        if (self.mesh is not None):
            self.mesh.euler_angle = 180.0 * np.array(bm.dcm2angle(self.R.T)) / np.pi
            self.mesh.translation = self.position
            viewer.draw_coordinate_system(self.position, self.R.dot(self.U), c = 0.04)
            self.draw_landmarks()
            self.mesh.reorient()
            self.mesh.draw()

            for child in self.list_of_children:
                child[2].draw()
            pass
        pass


    def draw_landmarks(self):
        for landmark in self.landmarks.keys():
            x = self.landmark_to_global(landmark)
            viewer.draw_point(x[0], x[1], x[2])


    def flat_scale(self, scale):
        """
        Purpose: Applies a flat scale to all of the landmarks and mesh values.
        :param scale:
        :return:
        """
        self.position *= scale
        for landmark in self.landmarks:
            self.landmarks[landmark] *= scale
        self.mesh.apply_scale(scale)


    def scale_to_percentile(self, percentile):
        """
        Purpose: Scales the values in the model to the appropriate percentile
                 This uses a least-squares protocol to match measured lengths
                 to the approximate lengths of the percentile.
        :param percentile:
        :return:
        """
        c = find_scaling_factors_for_percentile(self.landmarks, self.std_dims, percentile, self.U)
        self.mesh.apply_transformation_matrix((self.U.T).dot(np.diag(c).dot(self.U)))
        self.landmarks = apply_func_to_landmark_database(self.landmarks, lambda x: x.dot(np.diag(c)))
        return c



    def landmark_to_global(self, landmark):
        """
        Purpose: Given a landmark identifier, converts it to the global coordinate system.
        :param landmark:
        :return:
        """
        return self.point_to_global(self.landmarks[landmark])
        #return (self.U.dot(self.rotmtrx().T)).dot(self.landmarks[landmark]) + self.position


    def landmark_global(self, landmark):
        """
        Purpose: Given a landmark identifier, converts the corresponding point to
                 a global coordinate system with a origin coincident with the position of the origin of
                 this body. These are related by the transformation:
                        x_global = U x_local
                    Where U is the local coordinate system U = [i j k]
                THIS DOESN'T ADD THE POSITION ONTO THE LANDMARK CALCULATION
        :param landmark:
        :return:
        """
        return self.point_to_global(self.landmarks[landmark]) - self.position
        #return (self.U.dot(self.rotmtrx().T)).dot(self.landmarks[landmark])



    def point_to_global(self, point):
        """
        Purpose: Given a triplet (x,y,z) in the local coordinate system of the body, converts it into the
                 global coordinate system. These are related by the transformation:
                        x_global = U x_local + r
                    Where U = [i j k], and i is the i-unit vector expressed in global coordinates
                          r is the position of the origin of the body in global coordinates.
        :param point:
        :return:
        """
        #return (self.rotmtrx()).dot(self.U).dot(point) self.R.dot(self.U)
        return self.position + (self.R.dot(self.U)).dot(point)



    def add_child_from_landmarks(self, child, landmark_in_parent, landmark_in_child):
        """
        Purpose: Adds a child in the current body with the provided position in the parent segment
                 and the parent's position in the child body.
        """
        #dr = self.landmark_global(landmark_in_parent) - child.landmark_global(landmark_in_child)
        self.list_of_children.append((landmark_in_parent, landmark_in_child, child))
        self.recompute_positions()

        pass


    def recompute_positions(self):
        """

        :return:
        """
        for child in self.list_of_children:
            child[2].position = self.landmark_to_global(child[0]) - child[2].landmark_global(child[1])
            child[2].recompute_positions()


    def apply_rotation_about_point(self, rot, r):
        """
        Applies a rotation to the coordinates stored in the segment

        TODO:
        - will eventually need to update all the points which attach here.
        :param R:
        :param r:
        :return:
        """
        self.position = rot.T.dot(self.position - r) + r
        self.R = rot.T.dot(self.R)
        for child in self.list_of_children:
            child[2].apply_rotation_about_point(rot, r)



    def apply_joint_angles(self):
        M = self.R.dot(self.U)
        for child in self.list_of_children:
            #child[2].apply_rotation_about_point(child[2].rotmtrx(), self.landmark_to_global(child[0]))
            child[2].apply_rotation_about_point(M.dot(child[2].rotmtrx()).dot(M.T), self.landmark_to_global(child[0]))
            child[2].apply_joint_angles()


    def reset_R(self):
        self.R = np.eye(3)
        for child in self.list_of_children:
            child[2].reset_R()


    '''

     def apply_rotation_about_point(self, R, r):
        """
        Applies a rotation to the coordinates stored in the segment

        TODO:
        - will eventually need to update all the points which attach here.
        :param R:
        :param r:
        :return:
        """
        self.position = R.dot(self.position - r) + r
        for child in self.list_of_children:
            child[2].apply_rotation_about_point(R, r)

    def nudge_position(self, r):
        """
        Purpose: Nudges the position of the model and all of its children.
        :param r:
        :return:
        """
        self.position += r
        for child in self.list_of_children:
            child[1].nudge_position(r)


    def rotate_child(self, child_to_rotate, R):
        """
        Purpose: Rotates a specific child whose name is specified in child_to_rotate by the provided
                 rotation matrix, R.
        :param child_to_rotate:
        :param R:
        :return:
        """
        for child in self.list_of_children:
            if (child[2].name == child_to_rotate):
                child[2].apply_rotation_about_point(R, self.landmark_to_global(child[0]))


    def rotate_children(self, R):
        """
        Purpose: Rotates all of the children in accordance to the rotation matrix specified in R
        :param R:
        :return:
        """
        for child in self.list_of_children:
            child[2].apply_rotation_about_point(R, self.landmark_to_global(child[0]))
    '''



class Tissue():

    def __init__(self):
        """
        Purpose: the tissue class is a
        :return:
        """
        self.list_of_points = []
        self.colour = np.array([0.0, 0.0, 0.0])
        self.name = None


    def add_point(self, seg, x):
        """

        :param seg:
        :param x:
        :return:
        """
        self.list_of_points.append((seg, x))

    def draw(self):
        """

        :return:
        """
        if (len(self.list_of_points) >= 2):
            viewer.begin_line(colour = self.colour)
            for pt in self.list_of_points:
                x = pt[0].point_to_global(pt[1])
                viewer.add_point(x)
            viewer.end_line()





class Ligament(Tissue):

    def __init__(self):
        """

        :return:
        """
        Tissue.__init__(self)
        self.colour = np.array([0.0, 0.0, 0.2])



class Muscle(Tissue):

    def __init__(self):
        """
        Purpose: The muscle
        """
        Tissue.__init__(self)
        self.colour = np.array([0.2, 0.0, 0.0])












"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            USEFUL FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Listed below are some functions that I've used to manage what I call a "landmark dataset". This is
    a dictionary of points and 3D locations in local coordinates. For example:

        landmarks = {"anterior-superior vertebral body": [x, y, z],
                     "blah blah blah": [x, y, z]}

    This is in contrast to a corresponding dimension database which houses the mean and standard deviation
    distances between two landmarks. It is al ist of triplets in this way:

        dims = [("anterior-superior vertebral body", "blah blah blah", [mu, sigma]),
                ("blah blah blah", "blah blah blah", [mu, sigma)]

    Where mu and sigma are the mean and standard deviation for the distance between the two points. If
    the same point is repeated twice, the functions computing the lengths will assume that it is a bi-
    laterally symmetric point.

    In addition to the unpacking functions that allow for the reading of a csv file which holds all of
    the information.

    Lastly, this also implements some logic for finding scaling factors based on the means and standard
    deviations for some specified distances between landmarks, as well as their standard deviations.
"""


def perc2val(mean, sd, percentile):
    """
    Purpose: Computes the value corresponding to the given percentile of a normal distribution
             with given mean and standard deviation.
    :param mean:
    :param sd:
    :param percentile:
    :return:
    """
    if (percentile > 100.0):
        percentile /= 100.0
    return mean + stat.norm.ppf(percentile)*sd



def val2perc(mean, sd, x):
    """
    Purpose: Returns the corresponding percentile for a given value, in a distribution with
             the provided mean and standard deviation. This function is the inverse of
             perc2val (defined above).
    :param mean:
    :param sd:
    :param x:
    :return:
    """
    return stat.norm.cdf((x-mean)/sd)


def apply_func_to_landmark_database(data, func):
    """
    Purpose: Applies the provided function to the landmark data
    :param data:
    :param func:
    :return:
    """
    for landmark in data.keys():
        data[landmark] = func(data[landmark])
    return data

def norm(vec):
    """
    Purpose: Returns the norm of a vector
    :param vec:
    :return:
    """
    return np.sqrt(np.sum((vec)**2))


def normalize(vec):
    """
    Purpose: Normalizes the provided vector.
    :param vec:
    :return:
    """
    return vec/norm(vec)


def compute_percentile(dims, percentile):
    """
    Purpose: Converts the dimensions in a dimensions database to a a given percentile.
    :param dims:
    :param percentile:
    :return:
    """
    actual = []
    for dim in dims:
        actual.append(perc2val(dim[2][0], dim[2][1], percentile))
    return np.array(actual)


def compute_vector(data, p1, p2 = None):
    """
    Purpose: Computes the vector between two points in the landmark databse.
    :param data: the landmark database
    :param p1:   the identifier of the first landmark
    :param p2:   the identifier of the second landmark.
                 Note: If p2 is None, OR the same as p1, this function will treat the
                       point as being bilaterally symmetric.
    :return:
    """
    if (p2 == None or p1 == p2):
        return np.array([2*np.abs(data[p1][0]), 0.0, 0.0]) # symmetry
    else:
        return data[p1] - data[p2]


def compute_vectors_squared(data, dims, U = np.eye(3)):
    """
    Purpose: Computes the vectors in the landmark database with square
             entries correspoinding to the lengths in the dimensions
             database
    :param data: the landmark database (in local coordinates)
    :param dims: the dimensions database
    :param U:    the local coordinate system of the rigid body
    :return:     an array of vectors with squared magnitudes.
    """
    vecs = []
    for dim in dims:
        vecs.append((U.dot(compute_vector(data, dim[0], dim[1])))**2)
    return np.array(vecs)


def find_scaling_factors_for_percentile(data, dims, percentile, U = np.eye(3)):
    """
    Purpose: This function does all the heavy lifting for finding scaling factors.
    :param data:            the landmark database
    :param U:               the local coordinate system
    :param dims:            the dimensions database
    :param percentile:      the chosen percentile
    :return:
    """
    actual = compute_percentile(dims, percentile)
    vecs = compute_vectors_squared(data, dims, U)
    objective = lambda c: np.sum((np.sqrt([norm(x*(c**2)) for x in vecs]) - actual)**2)
    c = minimize(objective, np.array([1.0, 1.0, 1.0]), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    return c.x



def unpack_file(filename):
    """
    Purpose: This unpacks a file specified by filename. These are comma separated files where the first entry in the
             file determines whether the subsequent values are:
                            s : a scale-number. These detail two landmarks, and a mean and standard deviation
                                of the distance between the two landmarks
                            i : the i-unit vector of the local coordinate system
                            j : the j-unit vector of the local coordinate system
                            k : the k-unit vector of the local coordinate system
                            p : a landmark. These will have the name of the landmark, as well as the (x,y,z)-values
                                of the landmark in the local coordinate system
                            v : a vertex (x,y,z) for the triangular mesh
                            vn : a normal vector for a given vertex
                            f : a face indicator. These are a triplet of integers specifying the vertices by which to make
                                a triangular mesh.
                                >> Note: In most .obj editing software, these begin at 1, whereas python indexes from 0!
    Input: filename is the file's name (and directory) where it can be found
    Output: unpack_file will return an
    :param filename:
    :return:
    """
    vertices = []
    faces = []
    normals = []
    landmarks = dict()
    dims = []
    i = np.array([1.0, 0.0, 0.0])
    j = np.array([0.0, 1.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    moi_x = np.zeros([3,1])
    moi_y = np.zeros([3,1])
    moi_z = np.zeros([3,1])
    mass = 0.0


    f = open(filename)
    for line in f:
        vals = line.strip().split(",")

        if (vals[0] == "s"):
            dims.append((vals[1], vals[2], np.array([float(x) for x in vals[3:]])))
        elif (vals[0] == "f"):
            vals = [x.split("//",1)[0] for x in vals[1:4]]
            faces.append([int(x)-1 for x in vals])
            #faces.append([int(x.strip("//")[0])-1 for x in vals[1:]])
        elif (vals[0] == "v"):
            vertices.append([float(x) for x in vals[1:4]])
        elif (vals[0] == "vn"):
            normals.append([float(x) for x in vals[1:4]])
        elif (vals[0] == "i"):
            i = np.array([float(x) for x in vals[1:4]])
        elif (vals[0] == "j"):
            j = np.array([float(x) for x in vals[1:4]])
        elif (vals[0] == "k"):
            k = np.array([float(x) for x in vals[1:4]])
        elif (vals[0] == "p"):
            landmarks[vals[1]] = np.array([float(x) for x in vals[2:5]])
        elif (vals[0] == "moix"):
            moi_x = [float(x) for x in vals[1:4]]
        elif (vals[0] == "moiy"):
            moi_y = [float(x) for x in vals[1:4]]
        elif (vals[0] == "moiz"):
            moi_z = [float(x) for x in vals[1:4]]
        elif (vals[0] == "mass"):
            mass = float(vals[1])
        else:
            pass
    U = np.array([i, j, k])
    I = np.array([moi_x, moi_y, moi_z])
    vertices = np.array(vertices).dot(U)
    normals = np.array(normals)
    faces = np.array(faces)
    f.close()
    return U.T, mass, I, landmarks, dims, vertices, normals, faces



def segment_from_file(segmentname, filename):
    U, mass, I, landmarks, dims, vertices, normals, faces = unpack_file(filename)
    return Segment(name = segmentname,
                   mass = mass,
                   moment_of_inertia = I,
                   orientation = U,
                   position = np.array([0.0, 0.0, 0.0]),
                   landmarks = landmarks,
                   standard_dimensions=dims,
                   mesh = viewer.Mesh(vertices, normals, faces))



def add_landmark_to_database(dataset, landmark_name, x):
    """
    Purpose: Adds the given landmark to the landmark dataset
    :param dataset:
    :param landmark_name:
    :param x:
    :return:
    """
    dataset[landmark_name] = x
    return dataset














