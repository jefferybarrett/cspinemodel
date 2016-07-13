import numpy as np
import pybiomech as bm
import scipy.stats as stat
from scipy.optimize import minimize
from scipy.special import erf
from . import viewer


"""
A Force is represented as a tuple of two 3-vectors: corresponding to the point of application in a local coordinate
system as well as a magnitude/direction of force (in global coordinates).
"""





class Segment():

    def __init__(self, mass = None, moment_of_inertia = None, orientation = None,
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
        self.active_frame = 0
        self.number_of_frames = 0

        self.mass = mass
        self.I = moment_of_inertia
        self.R = np.array([np.eye(3)])                                      # no data

        self.U = orientation
        self.M = None               # this is a Nx3x3 tensor where each 3x3 matrix
                                    # is the [i j k] local coordinate system
        self.position = position
        self.x = None               # position of center of mass
        self.v = None               # velocity of center of mass
        self.a = None               # acceleration of center of mass
        self.omega = None           # angular velocity of frame
        self.alpha = None           # angular acceleration of frame

        self.landmarks = landmarks
        self.global_landmarks = dict()
        self.std_dims = standard_dimensions
        self.mesh = mesh

        self.list_of_forces = []                    # list of external forces
        self.list_of_moments = []

        self.list_of_joints = []
        self.parent_joint_landmark = None
        self.parent_joint = None


    def add_force(self, force):
        """
        Purpose: Appends a force onto the list of forces currently in the model. A force is a tuple of two
                 3x1 vectors: a position of application (in local coordinates) and a magnitude/direction
                 (in global coordinates)
        :param force:
        :return:
        """
        self.list_of_forces.append(force)

    def rotmtrx(self):
        """
        Purpose: Returns the rotation matrix of the local orientation
        :return:
        """
        R = []
        for triplet in self.joint_euler_angles:
            R.append(bm.angle2dcm(triplet[0], triplet[1], triplet[2]))
        return np.array(R)

    def draw(self):
        """
        Purpose: Calculates the position and orientation of the rigid body, and draws it in the GFXwindow.
                Note: There must be an active GFXWindow for this to work right!!
        :return:
        """
        if (self.mesh is not None):
            self.mesh.euler_angle = 180.0 * np.array(bm.dcm2angle(self.R[self.active_frame].T)) / np.pi
            self.mesh.translation = self.position[self.active_frame]
            viewer.draw_coordinate_system(self.position[self.active_frame], self.R[self.active_frame].dot(self.U), c = 0.04)
            #self.draw_landmarks()
            self.mesh.reorient()
            self.mesh.draw()

        for joint in self.list_of_joints:
            joint.child_segment.draw()

        #for child in self.list_of_children:
        #    child[2].draw()
        self.active_frame += 1
        if (self.active_frame >= self.number_of_frames):
            self.active_frame = 0


    def draw_landmarks(self):
        for landmark in self.global_landmarks.keys():
            x = self.global_landmarks[landmark]
            viewer.draw_point(x[self.active_frame][0], x[self.active_frame][1], x[self.active_frame][2])


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
        #return self.position + (self.R.dot(self.U)).dot(point)
        #return self.position[self.active_frame] + (self.R[self.active_frame].dot(self.U)).dot(point)
        if (self.M is None):
            self.M = np.einsum('nik,kj->nij', self.R, self.U)
        return self.position + np.einsum('nij,j->ni', self.M, point)


    def compute_global_landmarks(self):
        for landmark in self.landmarks.keys():
            self.global_landmarks[landmark] = self.landmark_to_global(landmark)


    def add_joint(self, joint):
        """

        :param joint:
        :return:
        """
        self.list_of_joints.append(joint)


    def add_child_from_landmarks(self, child, landmark_in_parent, landmark_in_child):
        """
        Purpose: Adds a child in the current body with the provided position in the parent segment
                 and the parent's position in the child body.
        """
        self.list_of_children.append((landmark_in_parent, landmark_in_child, child))
        self.recompute_positions()
        child.parent = self
        child.parent_landmark = landmark_in_child
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
        Purpose: This function has been optimized to vectorize the computation of applying
                 a rotation about a time varying point, r, by a time-varying matrix, rot
                    rot is an (Nx3x3) rotation matrix. where N is the number of frames
                    r is an (Nx3) matrix, where N is, once again, the number of frames
        :param R:
        :param r:
        :return:
        """
        self.position = np.einsum('nij,nj->ni', rot, self.position - r) + r
        self.R = np.einsum('nij,njk->nik', rot, self.R)
        self.M = np.einsum('nik,kj->nij', self.R, self.U)
        for joint in self.list_of_joints:
            joint.child_segment.apply_rotation_about_point(rot, r)
        #for child in self.list_of_children:
        #    child[2].apply_rotation_about_point(rot, r)


    def nudge_position(self, dx):
        self.position = dx + self.position
        for joint in self.list_of_joints:
            joint.child_segment.nudge_position(dx)



    def apply_joint_angles(self):
        """
        Purpose: Imposes the specified joint angles on the rest of the model
        :return:
        """
        #M = np.einsum('nik,kj->nij',self.R, self.U)
        self.M = np.einsum('nik,kj->nij', self.R, self.U)
        for child in self.list_of_children:
            # This is old code which was commented out because it didn't result in physioloigical motion
            # The new solution is to rotate about the current segment's coordinate system, which transforms
            # the rotation matrix, R, to UMU', which has been implemented using the einsum function.
            #child[2].apply_rotation_about_point(child[2].rotmtrx(), self.landmark_to_global(child[0]))
            #T = child[2].rotmtrx()
            T = np.einsum('nij,njk->nik', self.M, np.einsum('nij,nkj->nik',child[2].rotmtrx(), self.M))
            child[2].apply_rotation_about_point(T, self.landmark_to_global(child[0]))
            child[2].apply_joint_angles()


    def apply_joints(self):
        self.M = np.einsum('nik,kj->nij', self.R, self.U)
        for joint in self.list_of_joints:
            T = np.einsum('nij,njk->nik', self.M, np.einsum('nij,nkj->nik', joint.rotmtrx(), self.M))
            dx = np.einsum('nij,nj->ni', self.M, joint.displacement())
            joint.child_segment.apply_rotation_about_point(T, self.landmark_to_global(joint.landmark_in_parent))
            joint.child_segment.nudge_position(dx)
            joint.child_segment.apply_joints()



    def reset_R(self):
        self.R = []
        for i in range(0,self.number_of_frames):
            self.R.append(np.eye(3))
        self.R = np.array(self.R)


    def reset_position(self):
        self.position = np.zeros([self.number_of_frames, 3])


    def allocate_frames(self, n):
        self.number_of_frames = n
        self.M = None
        self.reset_R()
        self.reset_position()

    def change_joint_angles(self, new_angles):
        """
        Purpose: Updates the joint angles
        :param new_angles:
        :return:
        """
        self.joint_euler_angles = new_angles
        self.number_of_frames = len(self.joint_euler_angles)
        self.reset_R()
        self.reset_position()


    def compute_kinematics(self, dt):
        """
        Purpose: Computes the velocity and acceleration of the body's center of mass as well as the
                 angular velocity and acceleration.
        :return:
        """
        self.x = self.global_landmarks['center of mass']
        self.v = bm.vdiff(self.x, dt)
        self.a = bm.vdiff(self.v, dt)

        M = np.einsum('nik,kj->nij',self.R, self.U)
        dU = bm.vdiff(M, dt)
        omega_matrix = np.einsum('nij,nkj->nik', dU, M)
        self.omega = np.zeros_like(self.x)
        self.omega[:,0] = omega_matrix[:,2,1]
        self.omega[:,1] = omega_matrix[:,0,2]
        self.omega[:,2] = omega_matrix[:,1,0]
        self.alpha = bm.vdiff(self.omega, dt)



    def inverse_dynamics(self, dt):
        """
        Purpose: Performs inverse dynamics to compute the net joint moment of the current segment.
        :return:
        """
        if (self.omega is None):
            self.compute_kinematics(dt)
            pass

        # ma = sum_F + joint_F - sum_joint_F => joint_F = ma - sum_F + sum_joint_F
        # Ia + wxIw = sum_M + joint_M - sum_joint_M - sum_joint_rxF
        # joint_M = (Ia + wxIw) - sum_M + sum_join_M + sum_joint_rxF

        I = np.einsum('nij,njk->nik', self.M, np.einsum('ij,nik->nkj', self.I, self.M))
        JRF = self.mass * self.a
        JRM = (np.einsum('nij,nj->ni', I, self.alpha) +
                       np.cross(self.omega, np.einsum('nij,nj->ni', I, self.omega)))

        for force in self.list_of_forces:
            F = np.ones([self.number_of_frames, 3]) * force[1]
            r = self.point_to_global(force[0]) - self.global_landmarks['center of mass']
            JRF -= F
            JRM -= np.cross(r, F)

        #for moment in self.list_of_moments:
        #   self.joint_M += np.ones([self.number_of_frames, 3]) * moment[1]

        for joint in self.list_of_joints:
            joint.child_segment.inverse_dynamics(dt)
            r = joint.child_segment.global_landmarks[joint.landmark_in_child] - self.global_landmarks['center of mass']
            JRF += joint.net_F
            JRM += joint.net_M + np.cross(r, joint.net_F)
        '''
        for child in self.list_of_children:
            child[2].inverse_dynamics(dt)
            r = self.global_landmarks[child[0]] - self.global_landmarks['center of mass']
            self.joint_F += child[2].joint_F
            self.joint_M += child[2].joint_M + np.cross(r, child[2].joint_F)
        '''

        if self.parent_joint_landmark is not None:
            r = self.global_landmarks[self.parent_joint_landmark] - self.global_landmarks['center of mass']
            JRM -= np.cross(r, JRF)

        print(JRF)
        self.parent_joint.net_F = JRF
        self.parent_joint.net_M = JRM





class Tissue():

    def __init__(self):
        """
        Purpose: the tissue class is a
        :return:
        """
        self.list_of_points = []
        self.global_points = []
        self.length = None
        self.length0 = 1.0
        self.force = None
        self.lengthening_velocity = []
        self.colour = np.array([0.0, 0.0, 0.0])
        self.active_frame = 0
        self.name = None
        self.groupID = None

    def compute_length(self, sampling_rate):
        """
        Purpose: Computes the length of the tissue
        :return:
        """
        self.length = np.sqrt(np.sum((self.global_points[1:][0] - self.global_points[0:-1][0])**2, axis = 1))
        self.lengthening_velocity = bm.vdiff(self.length, 1/sampling_rate)


    def compute_global_points(self, sampling_rate):
        """
        Purpose: Computes where the points are in global space
        :return:
        """
        self.global_points = []
        for pt in self.list_of_points:
            self.global_points.append(pt[0].point_to_global(pt[1]))
        self.number_of_frames = len(self.global_points[0])
        self.active_frame = 0
        self.compute_length(sampling_rate)
        self.compute_force()


    def compute_moment(self, seg, point):
        """
        Purpose: Given a segment and a point of rotation, computes the moment caused by the ligament on the segment
        :param seg:
        :param point:
        :return:
        """
        if (self.list_of_points[0][0].name == seg.name):
            return self.compute_moment_at_origin(point)
        elif (self.list_of_points[-1][0].name == seg.name):
            return self.compute_moment_at_insert(point)
        else:
            return np.zeros_like(self.global_points[0])


    def compute_moment_at_insert(self, point):
        """
        Purpose: Computes the moment (and returns an Nx3 matrix of moments)
        :param point:
        :return:
        """
        force_vector = (self.global_points[-1] - self.global_points[-2])
        force_vector = (force_vector.T * (self.force / np.sqrt(np.sum(force_vector**2,1)))).T
        moment_arm = point - self.global_points[-1]
        return np.cross(moment_arm, force_vector)


    def compute_moment_at_origin(self, point):
        """
        Purpose: Computes the moment about a given point
        :param point:
        :return:
        """
        force_vector = (self.global_points[1] - self.global_points[0])
        force_vector = (force_vector.T * (self.force / np.sqrt(np.sum(force_vector**2,1)))).T
        moment_arm = point - self.global_points[0]
        return np.cross(moment_arm, force_vector)


    def compute_force(self):
        self.force = 3 * self.length

    def get_colour(self):
        return self.colour

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
            viewer.begin_line(colour = self.get_colour())
            for pt in self.global_points:
                #x = pt[0].point_to_global(pt[1])
                viewer.add_point(pt[self.active_frame])
            viewer.end_line()
            self.active_frame += 1
            if (self.active_frame >= self.number_of_frames):
                self.active_frame = 0





class Ligament(Tissue):

    def __init__(self):
        """

        :return:
        """
        Tissue.__init__(self)
        self.colour = np.array([0.0, 0.0, 0.2])




class ILS_Ligament(Ligament):
    """
    This is a subclass of ligament which calculates its force using the ILS (Immortal Ligament
    Simplification). That is to say that it will have toe and linear regions, but will not break
    nor have a decrease in force. Currently it will also not have any viscoelasticity.

    The force calculation is done as follows:


    """

    def __init__(self, mu = -0.6, sigma = 0.23, k = 85.8, length0 = 0.0, n = 1, force_factor = 1,
                 disp_factor = 1):
        """
        This will initialize the ligament model with the provided parameters.
        :return:
        """
        Ligament.__init__(self)
        self.colour = np.array([0.0, 0.0, 0.1])
        self.mu = mu
        self.sigma = sigma
        self.k = k
        self.n = n
        self.force_factor = force_factor
        self.disp_factor = disp_factor
        self.length0 = length0



    def compute_force(self):
        if (self.length is not None):
            dx = (self.length - self.length0) * self.disp_factor * 1000.0
            self.force = self.k*self.sigma/(np.sqrt(2*np.pi)) * np.exp(-(self.mu + dx)**2 / (2*self.sigma**2))
            self.force += self.k*(self.mu + dx)/2 * (1 + erf((self.mu + dx)/(np.sqrt(2)*self.sigma)))
            self.force *= self.force_factor/self.n





class Muscle(Tissue):

    def __init__(self, PCSA = 1.0):
        """
        Purpose: The muscle
        """
        Tissue.__init__(self)
        self.activation = None
        self.PCSA = PCSA
        self.colour = np.array([0.05, 0.0, 0.0])

        self.hick = 35                                              # N/cm^2
        self.k_PE = 3                                               # dimensionless
        self.L_rest = 0.0                                           # m
        self.L_max = 0.6                                            # dimensionless
        self.S_k = 6.25                                             # dimensionless
        self.v_max = 5.0                                            # l_rests/sec
        self.a_f = 0.55                                             # unitless
        self.alpha = 1.3                                            # unitless
        self.beta = self.a_f * (self.alpha - 1) / (self.a_f + 1)    # unitless

    def set_activation(self, activation):
        """
        Purpose: An explicit setter of the activation in the muscle tissue
        :param activation:
        :return:
        """
        self.activation = activation


    def force_length(self):
        """
        Purpose: Computes the force-length factor for force production
        :return:
        """
        return np.exp(-self.S_k * (self.length - self.length0)**2)


    def force_velocity(self):
        """
        Purpose: Computes the force-velocity factor for force production. Together with force_length it makes up
                 half of the Hill type muscle model calculation for the contractile element.
        :return:
        """
        return (1 - self.lengthening_velocity/(self.v_max*self.length0)) / (1 + self.lengthening_velocity/(self.a_f * (self.v_max*self.length0)))

    def parallel_elastic(self):
        """
        Purpose: Computes parallel elastic force component.
        :return:
        """
        return np.exp(self.k_PE/self.L_max * (self.length/self.length0 - 1)) / (np.exp(self.k_PE) - 1)

    def compute_force(self):
        """
        Purpose: Computes the force in the muscle given an activation and lengthening history.
        :return:
        """
        if (self.activation is not None):
            self.force = self.hick * self.PCSA * self.activation * self.force_length() * self.force_velocity()
        else:
            self.force = np.zeros_like(self.length)
        self.force += self.parallel_elastic()

    def get_colour(self):
        """
        Purpose: This method colours the muscle in accordance to its activation.
        """
        if (self.activation is None or np.isnan(self.activation[self.active_frame])):
            return self.colour
        else:
            c0 = 0.1
            c1 = 0.9
            beta = 10.0
            A = self.activation[self.active_frame] * 10.0
            GB = c0*np.exp(-beta*A)
            R = A*c1 + (1-A)*(c0)
            return np.array([R, GB, GB])






class Joint():

    def __init__(self, parent = None, child = None, landmark_in_parent = None, landmark_in_child = None):
        """
        Initializes a 6 DOF joint
        :return:
        """
        self.joint_euler_angles = None
        self.joint_displacement = None
        self.net_M = None
        self.net_F = None


        self.parent_segment = parent
        self.landmark_in_parent = landmark_in_parent
        self.child_segment = child
        self.landmark_in_child = landmark_in_child

        parent.add_joint(self)
        child.parent_joint_landmark = landmark_in_child
        child.parent_joint = self


    def recompute_position(self):
        self.child_segment.position = self.parent_segment.landmark_to_global(self.landmark_in_parent) -\
                                      self.child_segment.landmark_global(self.landmark_in_child)

    def rotmtrx(self):
        """
        Purpose: Returns the rotation matrix of the local orientation
        :return:
        """
        R = []
        for triplet in self.joint_euler_angles:
            R.append(bm.angle2dcm(triplet[0], triplet[1], triplet[2]))
        return np.array(R)

    def displacement(self):
        return self.joint_displacement

    def angles(self):
        return self.joint_euler_angles

    def compute_force(self):
        if (self.joint_euler_angles is not None):
            return np.zeros_like(self.joint_displacement)
        else:
            return 0.0

    def compute_moment(self):
        if (self.joint_euler_angles is not None):
            return np.zeros_like(self.joint_euler_angles)
        else:
            return 0.0



class RotaryJoint(Joint):

    def __init__(self, stiffness_matrix = np.zeros(3), parent = None, child = None, landmark_in_parent = None, landmark_in_child = None):
        """
        Initializes a Rotary Joint object
        :return:
        """
        Joint.__init__(self, parent = parent, child = child, landmark_in_parent = landmark_in_parent, landmark_in_child = landmark_in_child)
        self.K = stiffness_matrix

    def displacement(self):
        return np.zeros_like(self.joint_euler_angles)

    def compute_force(self):
        return np.zeros_like(self.joint_euler_angles)

    def compute_moment(self):
        return self.K.dot(self.joint_euler_angles)











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


    I = np.einsum('nij,njk->nik', self.M, np.einsum('ij,nik->nkj', self.I, self.M))
        self.joint_F = -self.mass * self.a
        self.joint_M = -(np.einsum('nij,nj->ni', I, self.alpha) +
                       np.cross(self.omega, np.einsum('nij,nj->ni', I, self.omega)))

        for force in self.list_of_forces:
            F = np.ones([self.number_of_frames, 3]) * force[1]
            r = self.point_to_global(force[0]) - self.global_landmarks['center of mass']
            self.joint_F += F
            self.joint_M += np.cross(r, F)

        #for moment in self.list_of_moments:
        #   self.joint_M += np.ones([self.number_of_frames, 3]) * moment[1]

        for child in self.list_of_children:
            child[2].inverse_dynamics(dt)
            r = self.global_landmarks[child[0]] - self.global_landmarks['center of mass']
            self.joint_F += child[2].joint_F
            self.joint_M += child[2].joint_M + np.cross(r, child[2].joint_F)

        if self.parent_landmark is not None:
            r = self.global_landmarks[self.parent_landmark] - self.global_landmarks['center of mass']
            self.joint_M -= np.cross(r, self.joint_F)
"""


def linear_disc(params, angles):
    """

    :param params:
    :param angles:
    :return:
    """
    K = np.diag(params)
    return K.dot(angles)




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
    Purpose: Computes the vector between two points in the landmark database.
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
    print(c.x)
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
    moi_x = np.zeros([1,3])
    moi_y = np.zeros([1,3])
    moi_z = np.zeros([1,3])
    I = np.eye(3)
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
    I[:,0] = moi_x
    I[:,1] = moi_y
    I[:,2] = moi_z
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














