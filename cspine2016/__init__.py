"""
The Cervical Spine Model
Written by: Jeff M. Barrett, M.Sc. Candidate
            University of Waterloo

This package contains the information needed to produce the cervical spine model. It was initially programmed in 2016
by Jeff Barrett to fulfill the requirements for a Master's in Science in Kinesiology.

How to use this package:

"""
import numpy as np
import pybiomech as bm
import pybiomech.modeller as mod
import pybiomech.modeller.viewer as view



'''
==============================================================================================================
                    CONSTANTS
==============================================================================================================
    Listed below are some global constants:
        BONE_DIR is the directory which contains the bone files (.csv files)
        KIN_DIR  is the directory which contains some parameters for processing kinematic data. For example
                 the values for partitioning the flexion, axial rotation, and lateral bending angles.
'''
BONE_DIR = "cspine2016/anatomical_data/"
KIN_DIR = "cspine2016/kinematics_data/"
MUS_DIR = "cspine2016/muscle_data/"
LIG_DIR = "cspine2016/ligament_data/"





'''
==============================================================================================================
                    UTILITY FUNCTIONS
==============================================================================================================
    Listed below are a series of functions which carry out tasks that would be otherwise very tedious.
    While it is true that these may not be the most 'useful' functions per-se, they still provide a means
    to increase the readability of the resulting model code.
'''



def make_list_of_segments():
    """
    Purpose: Returns a list of segments used in the model. In this version, it will be a list contianing:
                - C1 - T1
                - the skull
            The order of this list is important for stringing the model together.
    :return:
    """
    list_of_segments = ["C" + str(x) for x in range(1,8)]
    list_of_segments.append("T1")
    list_of_segments.insert(0,"skull")
    return list(reversed(list_of_segments))



def make_list_of_centers_of_rotation():
    """
    Purpose: Returns a list of the names of the centers of rotations for the model.
    :return:
    """
    list_of_cor = ["C"+str(x)+"-C"+str(x+1)+" center of rotation" for x in range(1,7)]
    list_of_cor.insert(0,"C1-C0 center of rotation")
    list_of_cor.append("C7-T1 center of rotation")
    return list(reversed(list_of_cor))





class CSpine2016(object):


    def __init__(self, percentile = 0.5, mus_filename = "final_muscle_coordinates.csv",
                 kin_filename = "angle_partition.csv",
                 lig_filename = "final_ligament_coordinates.csv"):
        """
        Returns an instance of the Cervical Spine Model
        :return:
        """
        # begin by listing all of the segments that will be in the model
        # each of these needs to correspond to a .csv file that has been properly
        # formatted with a mesh and stuff.
        self.partition_coeff = None         # the coefficients for partitioning the angles
        self.segments = None                # a dictionary enumerating the segments themselves
        self.muscles = None                 # a list of Muscle objects
        self.scaling_factors = None         # a dictionary enumerating the scaling factors for each segment


        list_of_segments = make_list_of_segments()
        list_of_cor = make_list_of_centers_of_rotation()

        # first load the segment files and attach them together at the centers of rotation
        self.load_bone_files(list_of_segments, percentile)
        self.attach_model(list_of_segments, list_of_cor)

        # next we load the kinematics data. This will have information on how to partition the angles around the
        # axes.
        self.load_kinematic_variables(kin_filename)

        # now load in all of the muscle coordinates
        self.load_muscles(mus_filename)
        self.load_ligaments(lig_filename)

        # some parameters that will be populated with trial data
        self.flexion_angle = 0.0
        self.axial_rotation = 0.0
        self.lateral_bend = 0.0

    def load_kinematic_variables(self, filename):
        """
        Purpose: Loads the kinematic variables, this will allow for the partitioning of angles.
                 These populate an array known as the partition_coeff array. Here each entry is a
                 fraction of the total value which goes into a given joint. For example:
                    partition_coeff[0] = [0.242, 0.090, 0.177]
                 This would be the partition-coefficients for the head
        :return:
        """
        self.partition_coeff = []
        f = open(KIN_DIR + filename)
        for line in f:
            vals = line.split(",")
            if (vals[0] is not ''):
                self.partition_coeff.append(np.array([float(x) for x in vals[1:]]))
        f.close()
        pass

    def load_bone_files(self, list_of_files, percentile = 0.5):
        """
        Purpose: Loads the bone files pecified in the list_of_files
        :param list_of_files:
        :return:
        """
        self.segments = dict()
        self.scaling_factors = dict()
        for seg in list_of_files:
            self.segments[seg] = mod.segment_from_file(seg, BONE_DIR + seg + ".csv")
            self.scaling_factors[seg] = self.segments[seg].scale_to_percentile(percentile)
            self.segments[seg].flat_scale(1/1000.0) # convert from mm to m
            pass
        pass

    def attach_model(self, list_of_segments, list_of_cor):
        """
        Purpose: Strings the models together so that the common point is the centers of rotation. It is very important
                 that the list of segments is from skull to T1
        :param list_of_segments:
        :param list_of_cor:
        :return:
        """
        n = len(list_of_segments)
        for i in range(0, n-1):
            self.segments[list_of_segments[i]].add_child_from_landmarks(self.segments[list_of_segments[i+1]], list_of_cor[i], list_of_cor[i])
            pass
        pass

    def load_muscles(self, filename):
        """
        Purpose: This loads in all of the muscles that go into the model.
        :param filename:
        :return:
        """
        self.muscles = []
        f = open(MUS_DIR + filename)
        for line in f:
            vals = line.split(",")
            name = vals[0]
            n = 1
            attach_points = []
            while (n < len(vals) and vals[n] != ''):
                seg = vals[n]
                pts = np.array([float(x) for x in vals[n+1:n+4]])/1000.0 # convert to meters
                n += 4
                if (seg in self.scaling_factors.keys()):
                    pts = pts.dot(np.diag(self.scaling_factors[seg]))
                attach_points.append((seg, pts))
            attach_points.append(attach_points.pop(1))
            muscle_right = mod.Muscle()
            muscle_left = mod.Muscle()
            muscle_right.name = "right " + name
            muscle_left.name = "left " + name
            for point in attach_points:
                muscle_right.add_point(self.segments[point[0]], point[1])
                muscle_left.add_point(self.segments[point[0]], point[1].dot(np.diag([-1, 1, 1])))
            self.muscles.append(muscle_left)
            self.muscles.append(muscle_right)
        f.close()

    def load_ligaments(self, filename):
        """
        Purpose: This loads in all of the muscles that go into the model.
        :param filename:
        :return:
        """
        self.ligaments = []
        f = open(LIG_DIR + filename)
        for line in f:
            vals = line.split(",")
            name = vals[0]
            n = 1
            attach_points = []
            while (n < len(vals) and vals[n] != ''):
                seg = vals[n]
                pts = np.array([float(x) for x in vals[n+1:n+4]])/1000.0 # convert to meters
                n += 4
                if (seg in self.scaling_factors.keys()):
                    pts = pts.dot(np.diag(self.scaling_factors[seg]))
                attach_points.append((seg, pts))
            attach_points.append(attach_points.pop(1))
            if (attach_points[0][1][0] == 0.0 or name == "transverse ligament"):
                lig = mod.Ligament()
                lig.name = name
                for point in attach_points:
                    lig.add_point(self.segments[point[0]], point[1])
                self.ligaments.append(lig)
            else:
                lig_right = mod.Ligament()
                lig_left = mod.Ligament()
                lig_right.name = "right " + name
                lig_left.name = "left " + name
                for point in attach_points:
                    lig_right.add_point(self.segments[point[0]], point[1])
                    lig_left.add_point(self.segments[point[0]], point[1].dot(np.diag([-1, 1, 1])))
                self.ligaments.append(lig_left)
                self.ligaments.append(lig_right)
        f.close()

    def flex_model(self, phi):
        self.flexion_angle += phi
        self.update_angles()

    def lateral_bend_model(self, phi):
        self.lateral_bend += phi
        self.update_angles()

    def axial_rotate_model(self, phi):
        self.axial_rotation += phi
        self.update_angles()


    def update_angles(self):
        """
        Purpose: Updates the angles in the model.
        :return:
        """
        segments = ["skull", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        n = 0
        f = lambda x: 90.0 * np.sin(2*np.pi*x*(0.5)) * np.pi / 180.0
        #print(self.flexion_angle, f(self.flexion_angle))
        for seg in segments:
            self.segments[seg].joint_euler_angles = np.array([self.partition_coeff[n][0]*f(self.flexion_angle),
                                                self.partition_coeff[n][2]*f(self.lateral_bend),
                                                self.partition_coeff[n][1]*f(self.axial_rotation)])
            n += 1

    def inverse_kinematics(self, landmark_database):
        """
        Purpose: Given a landmark database for a kinematic trial, this will calculate the joint angles which minimize
                 the objective function:
                        J(q) = \sum_{n=1}^{N} || x_approx(n) - x(n) ||^2 + 1/2 q.T K q
        :param landmark_database:
        :return:
        """
        pass

    def draw_landmarks(self):
        for seg in self.segments.keys():
            self.segments[seg].draw_landmarks()


    def update_model(self):
        self.segments["T1"].reset_R()
        self.segments["T1"].apply_joint_angles()
        self.segments["T1"].recompute_positions()

    def draw_muscles(self):
        for muscle in self.muscles:
            muscle.draw()


    def print_head_angle(self):
        print(self.segments['C2'].euler_angles*180.0/np.pi)

    def animate(self):
        window = view.GFXWindow()
        window.add_object(self.segments["T1"])

        for muscle in self.muscles:
            window.add_object(muscle)

        for ligament in self.ligaments:
            window.add_object(ligament)

        #window.add_update_function(lambda dt: self.draw_landmarks())
        window.add_update_function(lambda dt: self.update_model())

        # some test kinematic stuff:
        #window.add_update_function(lambda dt: self.lateral_bend_model(-2.0*dt))
        #window.add_update_function(lambda dt: self.flex_model(-0.5*dt))
        #window.add_update_function(lambda dt: self.draw_muscles())
        window.add_update_function(lambda dt: self.axial_rotate_model(-0.25*dt))

        #window.add_update_function(lambda dt: self.print_head_angle())

        window.run()











