"""
This is the main executable for the cervical spine model. It only contians the "main" function, which
instantiates a cervical spine model and iterates through trials to compute final values.


The user can optionally choose to animate the model, which should show a movie of what the cervical spine
is doing throughout the trial.
"""



from cspine2016 import *


def main():
    cspine = CSpine2016(percentile = 0.50)
    #cspine.flex_model(1.5)
    print(cspine.partition_coeff)
    #cspine.lateral_bend_model(-20.0)
    #cspine.axial_rotate_model(-10.0)

    #cspine.flex_model(-70.0)
    cspine.animate()
    pass



main()




