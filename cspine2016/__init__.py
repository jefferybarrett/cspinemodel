"""
The Cervical Spine Model
Written by: Jeff M. Barrett, M.Sc. Candidate
            University of Waterloo

This package contains the information needed to produce the cervical spine model. It was initially programmed in 2016
by Jeff Barrett to fulfill the requirements for a Master's in Science in Kinesiology.

How to use this package:

"""
import numpy as np
import matplotlib.pyplot as plt
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

g = -9.806 # m/s/s





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

def make_list_of_joints():
    list_of_joints = ["C"+str(x)+"-C"+str(x+1) for x in range(1,7)]
    list_of_joints.insert(0,"C0-C1")
    list_of_joints.append("C7-T1")
    return list(reversed(list_of_joints))





class CSpine2016(object):


    def __init__(self, percentile = 0.5, sex = "male",
                 mus_filename = "final_muscle_coordinates.csv",
                 kin_filename = "angle_partition.csv",
                 lig_filename = "final_ligament_coordinates.csv"):
        """
        Returns an instance of the Cervical Spine Model
        :return:
        """

        self.sex = sex
        # begin by listing all of the segments that will be in the model
        # each of these needs to correspond to a .csv file that has been properly
        # formatted with a mesh and stuff.
        self.partition_coeff = None         # the coefficients for partitioning the angles
        self.segments = None                # a dictionary enumerating the segments themselves
        self.muscles = None                 # a list of Muscle objects
        self.scaling_factors = None         # a dictionary enumerating the scaling factors for each segment
        self.activation_map = None          # a mapping from EMG channel to muscles that will be used
        self.joints = None                  # a dictionary of joints


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
        self.load_ILS_ligaments(lig_filename, sex = sex)

        # some parameters that will be populated with trial data
        self.sampling_rate = None
        self.number_of_frames = None
        self.time = None
        self.flexion_angle = None
        self.axial_rotation = None
        self.lateral_bend = None



    def test_data(self):
        """
        Purpose: Populates the model with the following test data
        :return:
        """
        self.sampling_rate = 64     # Hz
        self.number_of_frames = 64 * 10
        self.time = np.linspace(0, self.number_of_frames/self.sampling_rate, self.number_of_frames)
        self.flexion_angle = (45.0 * np.sin(2*np.pi*self.time))
        self.axial_rotation = 0.0 * np.sin(2*np.pi*2*self.time)
        self.lateral_bend = 0.0*self.time

    ########################################################
    # LOADING METHODS
    ########################################################

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
        self.segments["global"] = mod.Segment(name = "global",
                                              orientation = np.eye(3))     # empty segment to represent global
        self.segments["global"].landmarks = {"origin": np.array([0.0, 0.0, 0.0])}
        self.scaling_factors = dict()
        for seg in list_of_files:
            self.segments[seg] = mod.segment_from_file(seg, BONE_DIR + seg + ".csv")
            self.scaling_factors[seg] = self.segments[seg].scale_to_percentile(percentile)
            self.segments[seg].flat_scale(1/1000.0) # convert from mm to m
            self.segments[seg].add_force((self.segments[seg].landmarks['center of mass'],
                                         self.segments[seg].mass * np.array([0.0, 0.0, g])))
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
        joint_names = make_list_of_joints()
        self.joints = dict()
        newJoint = mod.Joint(parent = self.segments["global"], child = self.segments["T1"],
                             landmark_in_child = "posterior-inferior vertebral body",
                             landmark_in_parent= "origin")
        self.joints["T1-global"] = newJoint
        #self.segments["global"].add_joint(newJoint)
        n = len(list_of_segments)
        for i in range(0, n-1):
            newJoint = mod.RotaryJoint(stiffness_matrix=np.eye(3),
                                       child = self.segments[list_of_segments[i+1]],
                                       parent = self.segments[list_of_segments[i]],
                                       landmark_in_parent = list_of_cor[i],
                                       landmark_in_child = list_of_cor[i])
            #self.segments[list_of_segments[i]].add_joint(newJoint)
            self.joints[joint_names[i]] = newJoint
            #self.segments[list_of_segments[i]].add_child_from_landmarks(self.segments[list_of_segments[i+1]], list_of_cor[i], list_of_cor[i])
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
            PCSA = float(vals[1])
            n = 2
            attach_points = []
            while (n < len(vals) and vals[n] != ''):
                seg = vals[n]
                pts = np.array([float(x) for x in vals[n+1:n+4]])/1000.0 # convert to meters
                n += 4
                if (seg in self.scaling_factors.keys()):
                    pts = pts.dot(np.diag(self.scaling_factors[seg]))
                attach_points.append((seg, pts))
            attach_points.append(attach_points.pop(1))
            muscle_right = mod.Muscle(PCSA = PCSA)
            muscle_left = mod.Muscle(PCSA = PCSA)
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

    def load_ILS_ligaments(self, filename, sex = "male"):
        """
        Purpose: This loads in all of the muscles that go into the model. These will also have parameters for the

        :param filename:
        :return:
        """
        self.ligaments = []
        f = open(LIG_DIR + filename)
        next(f) # skip the header
        for line in f:
            vals = line.split(",")
            name = vals[0]
            # name, k, sigma, mu, ff, df, n, mff, mdf, fmff, fmdf
            # 0,    1,   2,   3,  4,  5,  6,  7,  8,   9,    10
            k = float(vals[1])
            sigma = float(vals[2])
            mu = float(vals[3])
            n_elements = int(vals[6])
            if (sex == "male"):
                force_factor = float(vals[4]) * float(vals[7])
                disp_factor = float(vals[5]) * float(vals[8])
            else:
                force_factor = float(vals[4]) * float(vals[9])
                disp_factor = float(vals[5]) * float(vals[10])
            n = 11
            attach_points = []
            ID = ""
            while (n < len(vals) and vals[n] != ''):
                seg = vals[n]
                ID = seg + ID
                pts = np.array([float(x) for x in vals[n+1:n+4]])/1000.0 # convert to meters
                n += 4
                if (seg in self.scaling_factors.keys()):
                    pts = pts.dot(np.diag(self.scaling_factors[seg]))
                attach_points.append((seg, pts))
            attach_points.append(attach_points.pop(1))
            if (attach_points[0][1][0] == 0.0 or name == "transverse ligament"):
                lig = mod.ILS_Ligament(k = k, sigma = sigma, mu = mu, force_factor = force_factor, disp_factor = disp_factor, n = n_elements)
                lig.name = name
                for point in attach_points:
                    lig.add_point(self.segments[point[0]], point[1])
                lig.groupID = ID
                self.ligaments.append(lig)
            else:
                lig_right = mod.ILS_Ligament(k = k, sigma = sigma, mu = mu, force_factor = force_factor, disp_factor = disp_factor, n = n_elements)
                lig_left = mod.ILS_Ligament(k = k, sigma = sigma, mu = mu, force_factor = force_factor, disp_factor = disp_factor, n = n_elements)
                if (name == "PAOM" or name == "PAAM" or name == "AAAM" or name == "AAOM" or name == "tectoral membrane" or name == "LF"):
                    lig_right.name = name
                    lig_left.name = name
                else:
                    lig_right.name = "right " + name
                    lig_left.name = "left " + name
                lig_right.groupID = ID
                lig_left.groupID = ID
                for point in attach_points:
                    lig_right.add_point(self.segments[point[0]], point[1])
                    lig_left.add_point(self.segments[point[0]], point[1].dot(np.diag([-1, 1, 1])))
                self.ligaments.append(lig_left)
                self.ligaments.append(lig_right)
        f.close()

    def angles_from_file(self, filename, sampling_rate):
        """
        Purpose: This function loads some input neck angles from the specified file. It will expect the file
                 to have lateral bending angles, then axial rotation angles, then flexion-extension angles in
                 three adjacent columns.
        :param filename:
        :return:
        """
        angles = bm.csvread(filename)
        self.sampling_rate = sampling_rate
        self.number_of_frames = np.size(angles, 0)
        self.flexion_angle = angles[:,2]
        self.lateral_bend = angles[:,0]
        self.axial_rotation = angles[:,1]
        self.update_angles()


    def load_emg_from_file(self, filename):
        """

        :param filename:
        :return:
        """
        data = bm.csvread(filename) / 100.0
        for muscle in self.muscles:
            if (self.activation_map is not None and self.activation_map[muscle.name] != -1):
                muscle.activation = data[:,self.activation_map[muscle.name]]


    def set_activation_map(self, filename):
        """

        :param filename:
        :return:
        """
        self.activation_map = dict()
        f = open(filename)
        for line in f:
            vals = line.strip().split(",")
            if (vals[1] == ''):
                vals[1] = 0
            self.activation_map[vals[0]] = int(vals[1]) - 1
        f.close()


    ########################################################
    # COMPUTATION METHODS
    ########################################################

    def update_angles(self):
        """
        Purpose: Updates the angles in the model.
        :return:
        """
        self.time = np.linspace(0, self.number_of_frames/self.sampling_rate, self.number_of_frames)
        segments = ["skull", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1", "global"]
        number_of_frames = len(self.time)
        print(number_of_frames)
        for seg in segments:
            self.segments[seg].allocate_frames(number_of_frames)


        n = 0
        deg2rad = lambda x: x * np.pi / 180.0

        joint_list = make_list_of_joints()
        print(joint_list)
        for joint_name in joint_list:
            angles = np.array([self.partition_coeff[7-n][0]*deg2rad(self.flexion_angle),
                               self.partition_coeff[7-n][2]*deg2rad(self.lateral_bend),
                               self.partition_coeff[7-n][1]*deg2rad(self.axial_rotation)]).T
            self.joints[joint_name].joint_euler_angles = angles
            self.joints[joint_name].recompute_position()
            n += 1

        self.joints["T1-global"].joint_euler_angles = np.zeros_like(angles)
        self.joints["T1-global"].joint_displacement = np.zeros_like(angles)
        self.segments["global"].apply_joints()
        #self.segments["global"].recompute_positions()

        '''
        for seg in segments:
            angles = np.array([self.partition_coeff[n][0]*deg2rad(self.flexion_angle),
                               self.partition_coeff[n][2]*deg2rad(self.lateral_bend),
                               self.partition_coeff[n][1]*deg2rad(self.axial_rotation)]).T
            self.segments[seg].change_joint_angles(angles)
            n += 1
        self.segments["T1"].change_joint_angles(np.zeros_like(angles))
        self.segments["T1"].apply_joint_angles()
        self.segments["T1"].recompute_positions()
        '''

        for ligament in self.ligaments:
            ligament.compute_global_points(self.sampling_rate)

        for muscle in self.muscles:
            muscle.compute_global_points(self.sampling_rate)

        for seg in self.segments.keys():
            self.segments[seg].compute_global_landmarks()


    def zero_tissues(self):
        """
        Purpose: Zeros the ligament forces when the model is in anatomical position.
        """
        flex_angles = self.flexion_angle
        lateral_bend = self.lateral_bend
        axial_rotation = self.axial_rotation
        sampling_rate = self.sampling_rate
        number_of_frames = self.number_of_frames

        # use anatomical position to determine the resting lengths
        self.sampling_rate = 4     # Hz
        self.number_of_frames = self.sampling_rate * 2
        self.time = np.linspace(0, self.number_of_frames/self.sampling_rate, self.number_of_frames)
        self.flexion_angle = 0.0*self.time
        self.axial_rotation =  0.0*self.time
        self.lateral_bend = 0.0*self.time
        self.update_angles()

        # compute the resting lengths
        for ligament in self.ligaments:
            ligament.length0 = ligament.length[3]

        for muscle in self.muscles:
            muscle.length0 = muscle.length[3]

        # revert back to how it was before
        self.flexion_angle = flex_angles
        self.lateral_bend = lateral_bend
        self.axial_rotation = axial_rotation
        self.sampling_rate = sampling_rate
        self.number_of_frames = number_of_frames
        if (flex_angles is not None):
            self.update_angles()


    def inverse_dynamics(self):
        """

        :return:
        """
        self.update_angles()
        self.segments["T1"].inverse_dynamics(1.0/self.sampling_rate)
        #for segment in self.segments.keys():
        #    self.segments[segment].inverse_dynamics(1.0/self.sampling_rate)


    def emg_assisted_optimization(self):
        """
        Purpose: Performs the EMG assisted optimization routine for gaining up muscle forces to balance the
                 external moment.
        :return:
        """
        def plot_moments(moment):
            plt.plot(self.time, moment[:,0], 'r', label= "x")
            plt.plot(self.time, moment[:,1], 'g', label = "y")
            plt.plot(self.time, moment[:,2], 'b', label = "z")
        big_moment_vector = []
        plots = [331, 332, 333, 334, 335,336, 337, 338]
        seg_list = ["skull", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1"]
        n = 0
        plt.figure()
        for segment in seg_list:
            if (segment != "T1"):
                seg = self.segments[segment]
                M = np.copy(seg.joint_M)
                cor = seg.parent_landmark       # the center of rotation
                for ligament in self.ligaments:
                    M -= ligament.compute_moment(seg, seg.global_landmarks[cor])
                plt.subplot(plots[n])
                plot_moments(M)
                plt.title(seg_list[n] + "/" + seg_list[n+1])
                n += 1
        plt.subplot(334)
        plt.ylabel("Joint Moment (Nm)")
        plt.subplot(338)
        plt.xlabel("Time (sec)")
        plt.show(block = False)



    ########################################################
    # PLOTTING METHODS
    ########################################################

    def triple_plot(self, p1, p2, p3, lab1, lab2, lab3, ylab1, ylab2, ylab3, xlab):
        """

        :param p1:
        :param p2:
        :param p3:
        :param lab1:
        :param lab2:
        :param lab3:
        :return:
        """
        plt.figure()
        plt.subplot(311)
        plt.plot(self.time, p1[:,0], label= lab1)
        plt.plot(self.time, p1[:,1], label = lab2)
        plt.plot(self.time, p1[:,2], label = lab3)
        plt.ylabel(ylab1)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
        plt.subplot(312)
        plt.plot(self.time, p2[:,0], label= lab1)
        plt.plot(self.time, p2[:,1], label = lab2)
        plt.plot(self.time, p2[:,2], label = lab3)
        plt.ylabel(ylab2)
        plt.subplot(313)
        plt.plot(self.time, p3[:,0], label= lab1)
        plt.plot(self.time, p3[:,1], label = lab2)
        plt.plot(self.time, p3[:,2], label = lab3)
        plt.ylabel(ylab3)
        plt.xlabel(xlab)
        plt.show(block = False)


    def plot_segment_linear_kinematics(self, segment):
        """

        :param segment:
        :return:
        """
        self.triple_plot(self.segments[segment].x, self.segments[segment].v, self.segments[segment].a,
                         "x", "y", "z", "Position (m)", "Velocity (m/s)", "Acceleration (m/s/s)", "Time (sec)")


    def plot_segment_kinetics(self, segment):
        """

        :param segment:
        :return:
        """
        plt.figure()
        plt.subplot(211)
        plt.plot(self.time, self.segments[segment].joint_F[:,0], label= "x")
        plt.plot(self.time, self.segments[segment].joint_F[:,1], label = "y")
        plt.plot(self.time, self.segments[segment].joint_F[:,2], label = "z")
        plt.ylabel("Joint Reaction Force (N)")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
        plt.subplot(212)
        plt.plot(self.time, self.segments[segment].joint_M[:,0], label= "x")
        plt.plot(self.time, self.segments[segment].joint_M[:,1], label = "y")
        plt.plot(self.time, self.segments[segment].joint_M[:,2], label = "z")
        plt.ylabel("Joint Moment (Nm)")
        plt.xlabel("Time (sec)")
        plt.show(block = False)


    def plot_segment_angular_kinematics(self, segment):
        euler_angles = bm.rad2deg(bm.vdcm2angles(self.segments[segment].M))
        self.triple_plot(euler_angles, self.segments[segment].omega, self.segments[segment].alpha,
                         "x","y","z","Euler Angles (deg)", "Angular Velocity (deg/s)", "Acceleration (deg/s/s)",
                         "Time (sec)")


    def plot_ligament_lengths(self, names, level = None):
        plt.figure()
        for ligament in self.ligaments:
            if (ligament.name in names and (level == None or ligament.groupID == level)):
                plt.plot(self.time, (ligament.length - ligament.length0) * 1000.0)
        plt.xlabel("Time (sec)")
        plt.ylabel("Ligament Element Lengths (mm)")
        plt.show(block = False)


    def plot_total_ligament_force(self, names, level = None):
        plt.figure()
        force = np.zeros_like(self.time)
        for ligament in self.ligaments:
            if (ligament.name in names and (level == None or ligament.groupID == level)):
                force += ligament.force
                ligament.colour = np.array([0.0, 0.0, 1.0])
        plt.plot(self.time, force)
        plt.title(names)
        plt.xlabel("Time (sec)")
        plt.ylabel("Ligament Force (N)")
        plt.show(block = False)


    def plot_ligament_force(self, names, level = None):
        plt.figure()
        for ligament in self.ligaments:
            if (ligament.name in names and (level == None or ligament.groupID == level)):
                plt.plot(self.time, ligament.force)
        plt.xlabel("Time (sec)")
        plt.ylabel("Ligament Force (N)")
        plt.show(block = False)


    def plot_muscle_lengths(self, names):
        """
        Purpose: Plots the length fo the given muscles. Note this will plot the lengths of all the muscle
                 elements associated
        :param name:        name is a set containing the names of muscles which should be plotted.
        :return:
        """
        plt.figure()
        for muscle in self.muscles:
            if (muscle.name in names):
                plt.plot(self.time, muscle.length)
        plt.xlabel("Time (sec)")
        plt.ylabel("Muscle Element Lengths")
        plt.show(block = False)

    def plot_muscle_force(self, names):
        """
        Purpose: Plots the force in the muscles
        :param names:
        :return:
        """
        force = np.zeros_like(self.time)
        plt.figure()
        for muscle in self.muscles:
            if (muscle.name in names):
                force += muscle.force
        plt.plot(self.time, force)
        plt.xlabel("Time (sec)")
        plt.ylabel("Muscle Element Force (N)")
        plt.title(names)
        plt.show(block = False)

    def plot_muscle_activation(self, names):
        plt.figure()
        for muscle in self.muscles:
            if (muscle.name in names):
                plt.plot(self.time, muscle.activation)
        plt.xlabel("Time (sec)")
        plt.ylabel("Muscle Element Activation")
        plt.title(names)
        plt.show(block = False)


    def plot_net_joint_moments(self):

        def plot_segment_kinetics(joint):
            plt.plot(self.time, self.joints[joint].net_M[:,0], 'r', label= "Flexion-Extension")
            plt.plot(self.time, self.joints[joint].net_M[:,1], 'g', label = "Lateral Bend")
            plt.plot(self.time, self.joints[joint].net_M[:,2], 'b', label = "Axial Twist")
            plt.title(joint)

        plt.figure()
        plots = [331, 332, 333, 334, 335,336, 337, 338]
        n = 0
        for joint in list(reversed(make_list_of_joints())):
            plt.subplot(plots[n])
            n += 1
            plot_segment_kinetics(joint)
        plt.subplot(334)
        plt.ylabel("Joint Moment (Nm)")
        plt.subplot(338)
        plt.xlabel("Time (sec)")
        plt.subplot(338)
        plt.legend(bbox_to_anchor=(1.1, 0.2, 1.3, .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
        plt.show(block = False)


    def plot_net_joint_forces(self):

        def plot_segment_kinetics(joint):
            plt.plot(self.time, self.joints[joint].net_F[:,0], 'r', label= "x")
            plt.plot(self.time, self.joints[joint].net_F[:,1], 'g', label = "y")
            plt.plot(self.time, self.joints[joint].net_F[:,2], 'b', label = "z")
            plt.title(joint)

        plt.figure()
        plots = [331, 332, 333, 334, 335,336, 337, 338]
        n = 0
        for joint in list(reversed(make_list_of_joints())):
            plt.subplot(plots[n])
            n += 1
            plot_segment_kinetics(joint)
        plt.subplot(334)
        plt.ylabel("Joint Force (N)")
        plt.subplot(338)
        plt.xlabel("Time (sec)")
        plt.legend(bbox_to_anchor=(1.1, 0.0, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
        plt.show(block = False)

    ########################################################
    # ANIMATION METHODS
    ########################################################

    def animate(self):
        window = view.GFXWindow()
        window.add_object(self.segments["global"])

        for muscle in self.muscles:
            window.add_object(muscle)

        for ligament in self.ligaments:
            window.add_object(ligament)
        window.run()

















"""

        #window.add_update_function(lambda dt: self.draw_landmarks())
        #window.add_update_function(lambda dt: self.update_model())

        # some test kinematic stuff:
        #window.add_update_function(lambda dt: self.lateral_bend_model(-2.0*dt))
        #window.add_update_function(lambda dt: self.flex_model(-0.5*dt))
        #window.add_update_function(lambda dt: self.draw_muscles())
        #window.add_update_function(lambda dt: self.axial_rotate_model(-0.25*dt))

        #window.add_update_function(lambda dt: self.print_head_angle())
"""