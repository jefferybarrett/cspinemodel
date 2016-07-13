

"""
The GUI-Module
    Jeff M. Barrett
    M.Sc Candidate | Biomechanics
    University of Waterloo

This module is support for the gfx objects so that they can be drawn to the screen
"""

import pyglet
from pyglet.gl import *
import numpy as np
import ctypes


# Constants
DEFAULT_MOUSE_SENSITIVITY = 1.0
DEFAULT_ROTATION_SENSITIVITY = 0.001
DEFAULT_SCROLL_SENSITIVITY = 0.01

DEFAULT_MAX_THROTTLE = 6.0
DEFAULT_MAX_TORQUE_THROTTLE = 6.0
DEFAULT_CAMERA_MASS = 10.0
DEFAULT_VISCOSITY = 20.0


def read_obj(filename):
    vertices = []
    normals = []
    faces = []
    deal = {"v": lambda x: vertices.append([float(y) for y in x]),
            "f": lambda x: faces.append([int(y)-1 for y in x]),
            "vn": lambda x: normals.append([float(y) for y in x])}
    f = open(filename)
    for line in f:
        vals = line.split(" ")
        if (vals[0] in deal.keys()):
            deal[vals[0]](vals[1:])
    f.close()


def vec(*args):
    return (GLfloat * len(args))(*args)


def draw_axes():
    # make the axes not shiny at all
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPointSize(10.0)
    glBegin(GL_POINTS)
    glVertex3f(0.0, 0.0, 0.0) # origin
    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)
    # x-axis
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)

    # y-axis
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)

    # z-axis
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 1.0)
    glEnd()
    glColor3f(0.0, 0.0, 0.0)


def draw_point(x, y, z, size = 10.0, colour = np.array([0.0, 0.0, 0.0])):
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(colour[0], colour[1], colour[2], 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex3f(x, y, z)
    glEnd()


def draw_coordinate_system(origin, R, c = 1.0, size = 10.0):
    x_p = origin + R.dot(np.array([c, 0.0, 0.0]))
    y_p = origin + R.dot(np.array([0.0, c, 0.0]))
    z_p = origin + R.dot(np.array([0.0, 0.0, c]))

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex3f(origin[0], origin[1], origin[2]) # origin
    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)
    # x-axis
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(x_p[0], x_p[1], x_p[2])

    # y-axis
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(y_p[0], y_p[1], y_p[2])

    # z-axis
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(z_p[0], z_p[1], z_p[2])
    glEnd()
    glColor3f(0.0, 0.0, 0.0)


def begin_line(colour = np.array([0.0, 0.0, 0.0])):
    """
    Purpose:
    :return:
    """
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPointSize(10.0)
    glColor3f(colour[0], colour[1], colour[2])
    glEnable(GL_LINE_SMOOTH)
    glLineWidth(4.0)
    glBegin(GL_LINE_STRIP)

def add_point(pt):
    """

    :param pt:
    :return:
    """
    glVertex3f(pt[0], pt[1], pt[2])

def end_line():
    glEnd(GL_LINE_STRIP)


def hextoint(i):
    """
    Purpose: Converts the provided hexidecimal integer to an integer
    :param i:
    :return:
    """
    if i > 255:
        i = 255
    return (1.0/255.0) * i




class Camera(object):

    def __init__(self, x = 1.0, y = 0.0, z = 0.0, xc = 0.0, yc = 0.0, zc = 0.0, upx = 0.0, upy = 0.0, upz = 1.0, alpha = 45.0):
        """
        Iitializes a camera object with an (x,y,z)-location and a (xc,yc,zc)-location of the center of attention.
        Finally, the up-direction is specified as a (upx,upy,upz)-vector

        By default the camera is one-unit along the x-axis, facing the origin, with positive-z pointing upward
        :param x:           the x-coordinate of the location of the camera
        :param y:           the y-              ""
        :param z:           the z-              ""
        :param xc:          the x-coordinate of the center of attention
        :param yc:          the y-coordinate            ""
        :param zc:          the z-coordinate            ""
        :param upx:         the x-component of the up-direction
        :param upy:         the y-component of the up-direction
        :param upz:         the z-component of the up-direction
        :return:
        """

        self.alpha = alpha

        def normalize(v):
            return v / np.sqrt(np.sum(v * v))


        self.x = np.array([x, y, z])
        self.xc = np.array([xc, yc, zc])
        up = normalize(np.array([upx, upy, upz]))

        # compute the forward direction vector
        forward = normalize(self.xc - self.x)

        # compute the left-direction
        left = normalize(np.cross(up, forward))

        # finally, recompute the up-direction
        up = normalize(np.cross(forward, left))

        # we can augment these vectors together to form a LCS:
        # this has the xyz unit vectors where:
        #       x: points forward
        #       y: points left
        #       z: points upward
        self.lcs = np.array([forward, left, up])

        # set-up for kinematics
        self.throttle = np.array([0.0, 0.0, 0.0])
        self.torque_throttle = np.array([0.0, 0.0, 0.0])
        self.max_throttle = DEFAULT_MAX_THROTTLE
        self.max_torque_throttle = DEFAULT_MAX_TORQUE_THROTTLE
        self.v = np.array([0.0,0.0,0.0])
        self.omega = np.array([0.0, 0.0, 0.0])
        self.mass = DEFAULT_CAMERA_MASS
        self.viscosity = DEFAULT_VISCOSITY




    def move_forward(self, amount):
        """
        Moves the observer foward. If amount is negative, this will move the observer backwards
        :param amount:
        :return:
        """
        self.x += amount * self.lcs[0]

    def move_backward(self, amount):
        """
        Convenience method
        :param amount:
        :return:
        """
        self.move_forward(-amount)

    def move_left(self, amount):
        """
        Moves the observer left. If the value is negative, it moves the observer right
        :param amount:
        :return:
        """
        self.x += amount * self.lcs[1]

    def move_right(self, amount):
        """
        Another convenience method
        :param amount:
        :return:
        """
        self.move_left(-amount)


    def move_upward(self, amount):
        """
        Moves the observer upward
        :param amount:
        :return:
        """
        self.x += amount * self.lcs[2]


    def move_down(self, amount):
        """
        Another convenience method
        :param amount:
        :return:
        """
        self.move_upward(-amount)


    def yaw(self, theta):
        """
        This performs a rotation by theta about the current up axis
        :param theta:
        :return:
        """
        #self.lcs = self.lcs.dot(np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]))
        self.lcs = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]).dot(self.lcs)


    def pitch(self, theta):
        """
        Increases the pitch of the observer (rotation about y-axis)
        :param theta:
        :return:
        """
        #self.lcs = self.lcs.dot(np.array([[np.cos(theta), 0.0, -np.sin(theta)],[0.0, 1.0, 0.0],[np.sin(theta),0.0,np.cos(theta)]]))
        self.lcs = np.array([[np.cos(theta), 0.0, -np.sin(theta)],[0.0, 1.0, 0.0],[np.sin(theta),0.0,np.cos(theta)]]).dot(self.lcs)


    def roll(self, theta):
        """
        Rolls the camera (about the current x-axis)
        :param theta:
        :return:
        """
        #self.lcs = self.lcs.dot(np.array([[1.0, 0.0, 0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]]))
        self.lcs = np.array([[1.0, 0.0, 0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]]).dot(self.lcs)


    def center_of_attention(self):
        """
        Convenience method for finding a worthwhile center of attention
        :return:
        """
        return self.x + self.lcs[0]


    def perspective(self, aspect_ratio = 1.0, near  = 0.01, far  = 100.0):
        """
        This is for setting up the projection to image-space.
        Input: aspect_ratio is the screen's aspect ratio (this doesn't seem to be working too well for some reason)
               near is the distance to the camera for the near clipping plane
               far is the distance to the camera of the far clipping plane
        :return: void
        """
        coa = self.center_of_attention() # get the center of attention
        up = self.lcs[2]
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.alpha, aspect_ratio, near, far)
        gluLookAt(self.x[0], self.x[1], self.x[2], coa[0], coa[1], coa[2], up[0], up[1], up[2])




    # methods for an engine-type maneouver
    def add_thrust(self, amount, i):
        """

        :param amount:
        :param i:
        :return:
        """
        dt = np.array([0.0, 0.0, 0.0])
        dt[i] = 1.0
        self.throttle += amount * dt
        if (np.abs(self.throttle[i]) > self.max_throttle):
            self.throttle[i] = np.sign(self.throttle[i]) * self.max_throttle


    def add_torque(self, amount, i):
        """

        :param amount:
        :param i:
        :return:
        """
        dt = np.array([0.0, 0.0, 0.0])
        dt[i] = 1.0
        self.torque_throttle += amount * dt
        if (np.abs(self.torque_throttle[i]) > self.max_torque_throttle):
            self.torque_throttle[i] = np.sign(self.torque_throttle[i]) * self.max_torque_throttle

    def throttle_forward(self, amount):
        """
        Increases the throttle by the specified amount in the forward direction
        :param amount:
        :return:
        """
        self.add_thrust(amount, 0)

    def throttle_backward(self, amount):
        """
        Convenience method
        :param amount:
        :return:
        """
        self.throttle_forward(-amount)


    def throttle_left_strife(self, amount):
        """
        Applies the amount to the throttle sideways
        :param amount:
        :return:
        """
        self.add_thrust(amount, 1)

    def throttle_right_strife(self, amount):
        """
        A convenience method
        :param amount:
        :return:
        """
        self.throttle_left_strife(-amount)

    def throttle_up(self, amount):
        """

        :param amount:
        :return:
        """
        self.add_thrust(amount, 2)

    def throttle_down(self, amount):
        """

        :param amount:
        :return:
        """
        self.throttle_up(-amount)


    def throttle_roll(self, amount):
        """

        :param amount:
        :return:
        """
        self.add_torque(amount, 0)

    def throttle_yaw(self, amount):
        self.add_torque(amount, 2)

    def throttle_pitch(self, amount):
        self.add_torque(amount, 1)

    def kill_throttle(self):
        self.throttle = np.array([0.0, 0.0, 0.0])
        self.torque_throttle = np.array([0.0, 0.0, 0.0])



    def update(self, dt):
        self.v += dt*(self.throttle.dot(self.lcs) - self.viscosity * self.v) / self.mass
        self.omega += dt * (self.lcs.dot(self.torque_throttle) - self.viscosity * self.omega) / self.mass
        self.x += dt * self.v
        self.lcs -= dt * np.cross(self.lcs, self.omega)
        self.torque_throttle -= dt * self.viscosity * self.torque_throttle



class GFXWindow(pyglet.window.Window):

    def __init__(self, width = 1200, height = 800, near = 0.01, far = 100.0, xloc = 400, yloc = 100):
        """
        Purpose: Initializer for the class GFXWindow
        :param width:
        :param height:
        :return:
        """
        # need to call the superclass' initializer because reasons
        super(GFXWindow, self).__init__(width, height,
                                        vsync=True,
                                        fullscreen = False,
                                        resizable=True,
                                        config=Config(sample_buffers=1, samples=4, depth_size=2.0, double_buffer=True,))

        self.colorscheme = {
            'background' : (hextoint(0), hextoint(0), hextoint(0), hextoint(255))
        }
        glClearColor(*self.colorscheme['background'])



        self.set_location(xloc,yloc)                            # note that these are wrt the top-left corner of the screen


        # optional parameters
        self.mouse_sensitivity = DEFAULT_MOUSE_SENSITIVITY      # this controls how quickly the mouse will rotate the screen
        self.scroll_sensitivity = DEFAULT_SCROLL_SENSITIVITY    # controls how quickly the mouse will speed things up
        self.rotation_sensitivity = DEFAULT_ROTATION_SENSITIVITY# the sensitivity from rotation
        self.aspect_ratio = width/height                        # the aspect ratio of the current window
        self.near = near                                        # distance to near clipping plane
        self.far = far                                          # distance to the far clipping plane


        # set up the camera
        self.camera = Camera()                                  # keep default parameters for the camera (for now)

        self.init_gl(width, height)                             # initialize the view

        # initialize the list of objects in the scene as well as the update functions
        self.list_of_objects = []                               # list of objects in the scene
        self.framerate = 1/32.0                                  # the desired framerate (1/ this)
        self.scene_update_fcns = []                             # functions the gui will call on update



    def add_object(self, obj):
        """
        Purpose: Adds a mesh object to the view. Note that the obj should implement a pyglet-friendly .draw() method
        :param obj:
        :return:
        """
        self.list_of_objects.append(obj)


    def add_update_function(self, f):
        self.scene_update_fcns.append(f)




    def init_gl(self, width, height):
        # Set clear color
        glClearColor(255/255, 255/255, 255/255, 0/255)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
        # but this is not the case on Linux or Mac, so remember to always
        # include it.
        # Define a simple function to create ctypes arrays of floats:

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(1.0, 1.0, 1.0, 0.0))
        glLightfv(GL_LIGHT1, GL_POSITION, vec(self.camera.x[0], self.camera.x[1], self.camera.x[2], 0.0))

        # Set viewport
        #print("resizing")
        glViewport(0, 0, width, height)
        self.camera.perspective(self.aspect_ratio, self.near, self.far)

        # One-time GL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)







    def on_resize(self, width, height):
        """
        Purpose: This is the event handler if the window gets resized
        :param width:
        :param height:
        :return:
        """

        # Set window values
        self.width = width
        self.height = height
        self.aspect_ratio = self.width / self.height
        # Initialize OpenGL context
        self.init_gl(width, height)



    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if (modifiers == 0):                                    # free moving
            self.camera.pitch(dy * self.rotation_sensitivity)
            self.camera.yaw(-dx * self.rotation_sensitivity)
        if (modifiers == 1):                                    # the shift key
            self.camera.roll(dy * self.rotation_sensitivity)
            #self.camera.yaw(-dx * self.rotation_sensitivity)
        if (modifiers == 64):                                   # the cmd key
            self.camera.move_left(dx * self.rotation_sensitivity)
            self.camera.move_down(dy * self.rotation_sensitivity)


    def on_key_press(self, symbol, modifiers):
        #print(symbol)
        if (symbol == 119): # w
            self.camera.throttle_forward(self.mouse_sensitivity)
        elif (symbol == 115): # s
            self.camera.throttle_backward(self.mouse_sensitivity)
        elif (symbol == 97): # a
            self.camera.throttle_left_strife(self.mouse_sensitivity)
        elif (symbol == 100): # d
            self.camera.throttle_right_strife(self.mouse_sensitivity)
        elif (symbol == 65361 or symbol == 101): # left arrow key
            self.camera.roll(self.rotation_sensitivity*10.0)
        elif (symbol == 65363 or symbol == 113): # right arrow key
            self.camera.roll(-self.rotation_sensitivity*10.0)
        elif (symbol == 114): # r
            self.camera.throttle_up(self.mouse_sensitivity)
        elif (symbol == 102): # d
            self.camera.throttle_down(self.mouse_sensitivity)
        elif (symbol == 32):
            self.camera.kill_throttle()
        else:
            pass





    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Purpose: When the mouse-wheel is scrolled, this will zoom out of the frame
        :param x:
        :param y:
        :param scroll_x:
        :param scroll_y:
        :return:
        """
        self.camera.move_forward(scroll_y * self.scroll_sensitivity)


    def on_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_axes()
        for obj in self.list_of_objects:
            obj.draw()

        self.camera.perspective(self.aspect_ratio, self.near, self.far)


    def update(self, dt):
        # camera relaxation of velocity
        # print(1/dt) # display the framerate
        # currently a stable 45 FPS (May 2 / 2016)
        #glLightfv(GL_LIGHT1, GL_POSITION, vec(self.camera.x[0], self.camera.x[1], self.camera.x[2], 0.0))
        #glEnable(GL_LIGHT1)

        glLightfv(GL_LIGHT1, GL_SPECULAR, vec(0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, vec(self.camera.xc[0], self.camera.xc[1], self.camera.xc[2], 0.0))

        self.camera.update(dt)
        for f in self.scene_update_fcns:
            f(dt)


    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        pyglet.app.run()




class Mesh(object):

    def __init__(self, vertices = np.array([]), normals = np.array([]), faces = np.array([]), meshtype = GL_TRIANGLES):
        """

        :param vertices:            a numpy-array of vertices (Nx3)
        :param normals:             a numpy array of normals (Nx3)
        :param faces:               an array of face-connectors (Nx3)
        :return:
        """

        self.vertices = np.array(vertices)
        self.normals = np.array(normals)
        self.faces = np.array(faces)
        self.type = meshtype
        #self.colour = np.tile(np.array([0.85, 0.76, 0.51]), len(self.vertices))#0.1 * np.ones([self.vertices.size])
        self.colour = np.tile(np.array([0.1, 0.09, 0.08]), len(self.vertices))#0.1 * np.ones([self.vertices.size])
        self.set_up_list()


        # properties needed for translation and rotation using OpenGL
        self.euler_angle = np.array([0.0, 0.0, 0.0])
        self.point_of_rotation = np.array([0.0, 0.0, 0.0]) # self.centroid()
        self.translation = np.array([0.0, 0.0, 0.0])




    def set_up_list(self):
        verts = self.vertices.flatten()
        vfaces = self.faces.flatten()
        vnorms = self.normals.flatten()
        vcols = self.colour.flatten()

        verts = (GLfloat * len(verts))(*verts)
        vnorms = (GLfloat * len(vnorms))(*vnorms)
        vfaces = (GLuint * len(vfaces))(*vfaces)
        vcols = (GLfloat * len(vcols))(*vcols)

        self.list = glGenLists(1)
        glNewList(self.list, GL_COMPILE)

        # set up the material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, vec(0.4, 0.4, 0.4, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, vec(0.1, 0.1, 0.1, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, vec(0.2, 0.2, 0.2, 1.0))

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glColorPointer(3, GL_FLOAT, 0, vcols)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, vnorms)
        glDrawElements(GL_TRIANGLES, len(vfaces), GL_UNSIGNED_INT, vfaces)
        glPopClientAttrib()
        glEndList()


    def apply_scale(self, alpha):
        """
        Purpose: Applies the scale, alpha, to the model
        :param alpha:
        :return:
        """
        self.vertices = alpha * self.vertices
        self.set_up_list()

    def apply_transformation_matrix(self, T):
        """
        Purpose: Applies the transformation matrix, T, to the triplets of vertices
        :param M:
        :return:
        """
        self.vertices = (self.vertices).dot(T)
        self.set_up_list()

    def translate_mesh(self, dx):
        """
        Purpose: Translates the model in 3D space by amounts dx = np.array([x,y,z])
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.translation += dx


    def translate_mesh_to(self, dx):
        """
        Purpose: Moves the mesh's origin to the point (x,y,z)
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.translation = dx




    def centroid(self):
        """
        Purpose: Returns the mesh's geometric centroid
        :return: Returns an nparray containing the mesh's geometric centroid
        """
        return np.mean(self.vertices, axis = 0)


    def reorient(self):
        """
        Orients the body before drawing it (optional) this also has the added bonus of reducing
        overhead
        :return:
        """
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(self.translation[0], self.translation[1], self.translation[2])
        #glTranslated(self.point_of_rotation[0], self.point_of_rotation[1], self.point_of_rotation[2])
        glRotated(self.euler_angle[2], 0.0, 0.0, 1.0)
        glRotated(self.euler_angle[1], 0.0, 1.0, 0.0)
        glRotated(self.euler_angle[0], 1.0, 0.0, 0.0)
        #glTranslated(-self.point_of_rotation[0], -self.point_of_rotation[1], -self.point_of_rotation[2])



    def orient(self):
        """
        Orients the body in 3D space (while preserving the previous orientation)
        :return:
        """
        glMatrixMode(GL_MODELVIEW)
        glTranslated(self.point_of_rotation[0], self.point_of_rotation[1], self.point_of_rotation[2])
        glRotated(self.euler_angle[0], 1.0, 0.0, 0.0)
        glRotated(self.euler_angle[1], 0.0, 1.0, 0.0)
        glRotated(self.euler_angle[2], 0.0, 0.0, 1.0)
        glTranslated(-self.point_of_rotation[0], -self.point_of_rotation[1], -self.point_of_rotation[2])
        glTranslated(self.translation[0], self.translation[1], self.translation[2])


    def deorient(self):
        """
        Undoes an orient call.
        :return:
        """
        glMatrixMode(GL_MODELVIEW)
        glTranslated(-self.translation[0], -self.translation[1], -self.translation[2])
        glTranslated(self.point_of_rotation[0], self.point_of_rotation[1], self.point_of_rotation[2])
        glRotated(-self.euler_angle[2], 0.0, 0.0, 1.0)
        glRotated(-self.euler_angle[1], 0.0, 1.0, 0.0)
        glRotated(-self.euler_angle[0], 1.0, 0.0, 0.0)
        glTranslated(-self.point_of_rotation[0], -self.point_of_rotation[1], -self.point_of_rotation[2])


    def draw(self):
        """
        Purpose: Draws the triangular mesh (needs a valid opengl context to work)
                 Note that this uses a ZYX-Euler Sequence
        :return:
        """
        glCallList(self.list)



class PointMesh():

    def __init__(self, point = np.array([0.0, 0.0, 0.0])):
        self.x = point


    def draw(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex3f(self.x[0], self.x[1], self.x[2])
        glEnd()












