"""
This is the main executable for the cervical spine model. It only contians the "main" function, which
instantiates a cervical spine model and iterates through trials to compute final values.


The user can optionally choose to animate the model, which should show a movie of what the cervical spine
is doing throughout the trial.
"""



from cspine2016 import *


SUB_DATA_DIR = "DATA/subject_data/"
MODEL_DATA_DIR = "DATA/model_data/"


def main():
    cspine = CSpine2016(percentile = 0.50, sex = "male")
    cspine.test_data()
    cspine.zero_tissues()

    # load the activation map which will tell it which EMG channels to attach to which muscles
    cspine.set_activation_map(MODEL_DATA_DIR + "activation_map.csv")

    # process an individual trial
    cspine.angles_from_file(SUB_DATA_DIR + "SUB1/007-hOnly-45rot30ext_neckangle.csv", sampling_rate = 10.0)
    cspine.load_emg_from_file(SUB_DATA_DIR + "SUB1/permve_001-hOnly-20lat.csv")


    cspine.inverse_dynamics()
    #cspine.emg_assisted_optimization()
    cspine.plot_net_joint_forces()
    cspine.plot_net_joint_moments()
    cspine.plot_segment_linear_kinematics("skull")
    #cspine.plot_segment_kinetics("C7")
    #cspine.plot_segment_angular_kinematics("skull")
    #cspine.plot_total_ligament_force("right CL", level = "C1C2")
    #cspine.plot_total_ligament_force("left CL", level = "C1C2")
    ##cspine.plot_ligament_lengths("right CL", level = "C1C2")
    #cspine.plot_muscle_force("right anterior scalene")
    #cspine.plot_muscle_force("right sternocleidomastoid")
    #cspine.plot_muscle_activation("right semispinalis capitis")
    #cspine.plot_segment_angular_kinematics("skull")



    cspine.animate()



    #cspine.update_angles()
    #cspine.flex_model(1.5)
    #cspine.lateral_bend_model(-20.0)
    #cspine.axial_rotate_model(-10.0)

    #cspine.flex_model(-70.0)
    #cspine.animate()
    #cspine.plot_muscle_lengths({'right splenius capitis', 'left splenius capitis'})
    pass



main()




