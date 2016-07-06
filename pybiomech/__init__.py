



"""
the pybiomech package
A package for tools for processing and modelling in human biomechanics.

Written by: Jeff M. Barrett
            M.Sc. Candidate (Biomechanics)
            University of Waterloo, Waterloo, Ontario
            Winter/Spring 2016
"""

import numpy as np



"""
==============================================================================================================
                    KINEMATICS
==============================================================================================================
This is mainly a library of functions that are useful for processing kinematic data.
None of these are vectorized (especially the rotation matrix stuff)
"""

def norm(x):
    return np.sqrt(np.sum(x**2))





def rotx(theta):
    """
    Purpose: Returns the rotation matrix about the x-axis
    :param theta:
    :return:
    """
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), np.sin(theta)], [0.0, -np.sin(theta), np.cos(theta)]])




def roty(theta):
    """
    Purpose: Returns the rotation matrix for a rotation about the y-axis by angle theta
    :param theta:
    :return:
    """
    return np.array([[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0], [np.sin(theta), 0.0, np.cos(theta)]])




def rotz(theta):
    """
    Purpose: Returns the rotation matrix for a rotation by theta about the z-axis
    :param theta:
    :return:
    """
    return np.array([[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])



def quat_mult(q, p):
    """
    Purpose: Computes the product of the quaternions q and p
    :param q:
    :param p:
    :return:
    """
    q0 = q[0]
    p0 = p[0]
    qv = q[1:]
    pv = p[1:]
    return np.concatenate([[q0*p0 - qv.dot(pv)], q0*pv + p0*qv + np.cross(qv, pv)])


def vquat_mult(q, p):
    """
    Applies quaternion multiplication to both the elements in q and p
    :param q:
    :param p:
    :return:
    """
    if (len(q) == len(p)):
        qp = np.zeros_like(q)
        for i in range(0, len(qp)):
            qp[i] = quat_mult(q[i], p[i])
        return qp
    else:
        print("Error: Array dimension mismatch!")
        return np.zeros_like(q)



def quat_conj(q):
    """
    Purpose: Returns the conjugate of the provided quaternion.
    :param q:
    :return:
    """
    return np.concatenate([[q[0]], -q[1:]])


def vquat_conj(q):
    """
    Purpose: Computes the conjugate of the provided quaternion
    :param q:
    :return:
    """
    return np.concatenate(q[:, 0], -q[:, 1:])


def quat_mag(q):
    """
    Purpose: Computes the magnitude of the provided quaternion, q
    :param q:
    :return:
    """
    return np.sqrt(q.dot(q))


def quat_inv(q):
    """
    Purpose: Returns the multiplicative inverse of q
    :param q:
    :return:
    """
    Q = quat_mag(q)
    if (Q != 0.0):
        return quat_conj(q) / Q
    else:
        print("Warning: Quaternion's magnitude is zero")
        return np.zeros_like(q)



def angle2dcm(phi, theta, psi, sequence = "XYZ"):
    """
    Purpose: Constructs a rotation matrix for the angles phi, theta, and psi corresponding to the sequence in sequence
    :param phi:
    :param theta:
    :param psi:
    :param sequence:
    :return:
    """
    if (sequence == "XYX" or sequence == "121"):
        return rotx(phi).dot(roty(theta).dot(rotx(psi)))
    elif (sequence == "XYZ" or sequence == "123"):
        return rotx(phi).dot(roty(theta).dot(rotz(psi)))
    elif (sequence == "XZX" or sequence == "131"):
        return rotx(phi).dot(rotz(theta).dot(rotx(psi)))
    elif (sequence == "XZY" or sequence == "132"):
        return rotx(phi).dot(rotz(theta).dot(roty(psi)))
    elif (sequence == "YXY" or sequence == "212"):
        return roty(phi).dot(rotx(theta).dot(roty(psi)))
    elif (sequence == "YXZ" or sequence == "213"):
        return roty(phi).dot(rotx(theta).dot(rotz(psi)))
    elif (sequence == "YZX" or sequence == "231"):
        return roty(phi).dot(rotz(theta).dot(rotx(psi)))
    elif (sequence == "YZY" or sequence == "232"):
        return roty(phi).dot(rotz(theta).dot(roty(psi)))
    elif (sequence == "ZXY" or sequence == "312"):
        return rotz(phi).dot(rotx(theta).dot(roty(psi)))
    elif (sequence == "ZXZ" or sequence == "313"):
        return rotz(phi).dot(rotx(theta).dot(rotz(psi)))
    elif (sequence == "ZYX" or sequence == "321"):
        return rotz(phi).dot(roty(theta).dot(rotx(psi)))
    elif (sequence == "ZYZ" or sequence == "323"):
        return rotz(phi).dot(roty(theta).dot(rotz(psi)))
    else:
        print("unidentified sequence")
        return np.eye(3)



def dcm2angle(R, sequence = "XYZ"):
    """
    Purpose: Decomposes the rotation matrix, R, into Euler angles with the specified rotation
    :param R:
    :param sequence:            A string which specifies the sequence to be used.
                                Supported sequences:
                                    - XYX or 121 (about X, then about Y, then about X)
                                    - XYZ or 123 (about Z, then about Y, then about X)
                                    - XZX or 131 (about X, then about Z, then about X)
                                    - XZY or 132 (about Y, then about Z, then about X)
                                    - YXY or 121 (about Y, then about X, then about Y)
                                    - YXZ or 213 (about Z, then about X, then about Z)
                                    - YZX or 231 (about X, then about Z, then about Y)
                                    - YZY or 232 (about Y, then about Z, then about Y)
                                    - ZYX or 312 (about X, then about Y, then about Z)
                                    - ZXZ or 313 (about Z, then about X, then about Z)
                                    - ZYX or 321 (about X, then about Y, then about Z)
                                    - ZYZ or 323 (about Z, then about Y, then about Z)
    :return:
    """
    if (sequence == "XYX" or sequence == "121"):
        return np.arctan2(R[1,0], R[2,0]), np.arccos(R[0,0]), np.arctan2(R[0,1], -R[0,2])
    elif (sequence == "XYZ" or sequence == "123"):
        return np.arctan2(R[1,2], R[2,2]), -np.arcsin(R[0,2]), np.arctan2(R[0,1], R[0,0])
    elif (sequence == "XZX" or sequence == "131"):
        return np.arctan2(R[2,0], -R[1,0]), np.arccos(R[0,0]), np.arctan2(R[0,2], R[0,1])
    elif (sequence == "XZY" or sequence == "132"):
        return np.arctan2(-R[2,1],R[1,1]), np.arcsin(R[0,1]), np.arctan2(-R[0,2], R[0,0])
    elif (sequence == "YXY" or sequence == "212"):
        return np.arctan2(R[0,1], -R[2,1]), np.arccos(R[1,1]), np.arctan2(R[1,0], R[1,2])
    elif (sequence == "YXZ" or sequence == "213"):
        return np.arctan2(-R[0,2], R[2,2]), np.arcsin(R[1,2]), np.arctan2(-R[1,0], R[1,1])
    elif (sequence == "YZX" or sequence == "231"):
        return np.arctan2(R[2,0], R[0,0]), -np.arcsin(R[1,0]), np.arctan2(R[1,2], R[1,1])
    elif (sequence == "YZY" or sequence == "232"):
        return np.arctan2(R[2,1], R[0,1]), np.arccos(R[1,1]), np.arctan2(R[1,2], -R[1,0])
    elif (sequence == "ZXY" or sequence == "312"):
        return np.arctan2(R[0,1], R[1,1]), -np.arcsin(R[2,1]), np.arctan2(R[2,0], R[2,2])
    elif (sequence == "ZXZ" or sequence == "313"):
        return np.arctan2(R[0,2], R[1,2]), np.arccos(R[2,2]), np.arctan2(R[2,0], -R[2,1])
    elif (sequence == "ZYX" or sequence == "321"):
        return np.arctan2(-R[1,0], R[0,0]), np.arcsin(R[2,0]), np.arctan2(-R[2,1], R[2,2])
    elif (sequence == "ZYZ" or sequence == "323"):
        return np.arctan2(R[1,2], -R[0,2]), np.arccos(R[2,2]), np.arctan2(R[2,1], R[2,0])
    else:
        print("unidentified sequence")
        return 0.0, 0.0, 0.0



def vangles2dcm(phi = None, theta = None, psi = None, sequence = "XYZ"):
    """
    Returns a Nx3x3 array of direction-cosine-matrices corresponding to the sequence of angles specified
    with the rotation sequence
    :param phi:
    :param theta:
    :param psi:
    :param sequence:
    :return:
    """
    if (len(phi) == len(theta) == len(psi)):
        R = []
        for i in range(0, len(theta)):
            R.append(angle2dcm(phi[i], theta[i], phi[i], sequence))
        return np.array(R)
    else:
        print("Error: angle arrays of different lengths")
        return np.array([])



def dcm2quat(R):
    """
    Purpose: Converts the 3x3 rotation matrix into a quaternion (1x4-vector)
    :param R:
    :return:
    """
    if (R[1,1] > -R[2,2]) and (R[0,0] > -R[1,1]) and (R[0,0] > -R[2,2]):
        b = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
        return (0.5/b) * np.array([b*b, R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0]])
    elif (R[1,1] < R[2,2]) and (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        b = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        return (0.5/b) * np.array([R[1,2]-R[2,1], b*b, R[0,1]+R[1,0], R[2,0]+R[0,2]])
    elif (R[1,1] > R[2,2]) and (R[0,0] < R[1,1]) and (R[0,0] < -R[2,2]):
        b = np.sqrt(1 - R[0,0] + R[1,1] - R[2,2])
        return (0.5/b) * np.array([R[2,0]-R[0,2], R[0,1]+R[1,0], b*b, R[1,2]+R[2,1]])
    else:
        b = np.sqrt(1 - R[0,0] - R[1,1] + R[2,2])
        return (0.5/b) * np.array([R[0,1]-R[1,0], R[2,0]+R[0,2], R[1,2]+R[2,1], b*b])


def vdcm2quat(R):
    """
    Purpose: Converts a Nx3x3 rotation matrix array into an Nx4 array of quaternions
    :param R:
    :return:
    """
    q = []
    for r in R:
        q.append(dcm2quat(r))
    return np.array(q)



def quat2dcm(q):
    """
    Purpose: Converts the quaternion into a direction cosine matrix (first normalzies the quaternion)
    :param R:
    :return:
    """
    q = q/np.sqrt(np.dot(q,q))
    R = np.zeros([3,3])
    R[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
    R[1,1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
    R[2,2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
    R[0,1] = 2*q[1]*q[2] + 2*q[0]*q[3]
    R[1,0] = 2*q[1]*q[2] - 2*q[0]*q[3]
    R[0,2] = 2*q[1]*q[3] - 2*q[0]*q[2]
    R[2,0] = 2*q[1]*q[3] + 2*q[0]*q[2]
    R[1,2] = 2*q[2]*q[3] + 2*q[0]*q[1]
    R[2,1] = 2*q[2]*q[3] - 2*q[0]*q[1]
    return R


def vquat2dcm(Q):
    """
    Purpose: convets an Nx4 quaternion array into an array of rotation matrices
    :param q:
    :return:
    """
    R = []
    for q in Q:
        R.append(quat2dcm(q))
    return np.array(R)







def vdiff(v, dt):
    """
    Purpose: A vectorized computation of the centered difference derivative of a timeseries of
             vectors/matrices in v assuming evenly spaced samples. While preserving second-order
             accuracy on the boundaries

    :param v:           The vector-array for differentiation (Nx(size)) where N is the time-dimension
                        This will work for any provided array, granted that the first dimension is used for
                        time differentiation.
                                -> Future versions should aim to generalize the axis as an input to the
                                   function.
                                -> Future versions should also generalize this to take t as an input and
                                   compute the derivative for unevenly spaced samples as well
    :param dt:          the timestep
    :return:            returns the derivative of the matrix v
    """
    i = np.arange(np.size(v, axis = 0))
    dv = np.copy(v)
    dv[i] = (v[np.roll(i,-1)] - v[np.roll(i,1)]) / (2*dt)
    dv[0] = (-3*v[0] + 4*v[1] - v[2])/(2*dt)
    dv[-1] = (3*v[-1] - 4*v[-2] + v[-3])/(2*dt)
    return dv





def dcm_angular_velocity(R, dt):
    """
    Purpose: Computes the angular velocity of the rotation matrix, R, with timestep dt
    :param R:           an Nx3x3 array representing the instantaneous orientation
    :param dt:          the timespacing
    :return:
    """
    dR = vdiff(R, dt)
    omega_matrix = np.einsum('nij,nkj->nik', dR, R) # dR/dt * R.T
    omega = np.zeros([len(R), 3])
    omega[:, 0] = omega_matrix[:, 2, 1]           # omega_x
    omega[:, 1] = omega_matrix[:, 0, 2]           # omega_y
    omega[:, 2] = omega_matrix[:, 1, 0]           # omega_z
    return omega





def quat_angular_velocity(q, dt):
    """
    Purpose: Computes the angular velocity vector given the Nx4 array of quaternions
    :param q:
    :param dt:
    :return:
    """
    dq = vdiff(q, dt)
    omega = 2 * vquat_mult(vquat_conj(q), dq)
    return omega[:,1:]














'''

==============================================================================================================
                    TEST CASES FOR DCM2ANGLE and ANGLE2DCM
==============================================================================================================
- this seems to work up to gimble lock.

seq = "ZYZ"
phi = np.pi / 7
theta = np.pi / 3
psy = np.pi / 5
R = bm.angle2dcm(phi, theta, psy, seq)
alpha, beta, gamma = bm.dcm2angle(R, seq)
R2 = bm.angle2dcm(alpha, beta, gamma, seq)


print("Summary:")
print("dtheta = " + str(np.sum(np.abs(np.array([alpha, beta, gamma]) - np.array([phi, theta, psy])))))
print("dR     = " + str(np.sum(np.abs(R - R2))))

print("")
print("details:")
print([alpha, beta, gamma])
print([phi, theta, psy])

print(R)
print(R2)



==============================================================================================================
                    TEST CASES FOR QUATERNION STUFF
==============================================================================================================
- this seems to work up to gimble lock.


phi = 10*np.random.rand()
theta = 10*np.random.rand()
psi = 10*np.random.rand()


R = bm.angle2dcm(phi, theta, psi)
q = bm.dcm2quat(R)
R2 = bm.quat2dcm(q)

print(R)
print(R2)

print("summary:")

print(np.sum(np.abs(R-R2)))





==============================================================================================================
                    TEST CASES FILTERING AND DIFFERENTIATING
==============================================================================================================
- this seems to work up to gimble lock.


N = 1024
f = 10
t = np.linspace(0, 1, N)
dt = t[1] - t[0]

R = np.zeros([N,2,2])

R[:,0,0] = np.cos(2*np.pi*f*t)
R[:,1,1] = np.cos(2*np.pi*f*t)
R[:,0,1] = -np.sin(2*np.pi*f*t)
R[:,1,0] = np.sin(2*np.pi*f*t)


print(R[0])

dR = bm.vdiff(R, dt)

plt.plot(t, R[:,0,0])
plt.plot(t, dR[:,0,0])
plt.show()


leR = bm.linear_envelop(R, 4, 1024)
plt.plot(t, R[:,0,0])
plt.plot(t, leR[:,0,0])
plt.show()




v = np.array([np.sin(t), np.cos(t)]).transpose()

dv = bm.vdiff(v, dt)

plt.subplot(2,1,1)
plt.plot(t, v)
plt.subplot(2,1,2)
plt.plot(t, dv)
plt.show()


N = 256
t = np.linspace(0, 4, N)
dt = t[1]-t[0]
phi = 2.0*t#2*np.pi * np.sin(2*np.pi*t)
theta = 0.0*t#np.pi/2-0.2*t#np.cos(2*np.pi*t)
psi = 0.0*t#0.2*t#np.sin(2*np.pi*t)

R = bm.vangles2dcm(phi, theta, psi)

omega = bm.dcm_angular_velocity(R, dt)
print(omega)
plt.plot(t, omega)
plt.show()



'''






















