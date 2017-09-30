import OpenGL
import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import OpenGL.GLE as GLE
import numpy as n
import imageio
from PIL import Image
from os import listdir
import sys
import math
import numpy

OpenGL.USE_FREEGLUT = True


class tickScaler:
    def __init__(self, minv, maxv):
        self.maxTicks = 6
        self.tickSpacing = 0
        self.lst = 10
        self.niceMin = 0
        self.niceMax = 0
        self.minPoint = minv
        self.maxPoint = maxv
        self.noTicks = 0
        self.calculate()

    def calculate(self):
        if not self.maxPoint == self.minPoint:
            self.lst = self.niceNum(self.maxPoint - self.minPoint, False)
            self.tickSpacing = self.niceNum(self.lst /
                                            (self.maxTicks - 1), True)
            self.niceMin = math.floor(self.minPoint /
                                      self.tickSpacing) * self.tickSpacing
            self.niceMax = math.ceil(self.maxPoint /
                                     self.tickSpacing) * self.tickSpacing
            self.noTicks = (self.niceMax - self.niceMin) / self.tickSpacing
        else:
            self.lst = 0
            self.tickSpacing = 0
            self.niceMin = 0
            self.niceMax = 0
            self.noTicks = 1

    def niceNum(self, lst, rround):
        self.lst = lst
        niceFraction = 0  # nice, rounded fraction */

        exponent = math.floor(math.log10(self.lst))
        fraction = self.lst / math.pow(10, exponent)

        if (self.lst):
            if (fraction < 1.5):
                niceFraction = 1
            elif (fraction < 3):
                niceFraction = 2
            elif (fraction < 7):
                niceFraction = 5
            else:
                niceFraction = 10
        else:
            if (fraction <= 1):
                niceFraction = 1
            elif (fraction <= 2):
                niceFraction = 2
            elif (fraction <= 5):
                niceFraction = 5
            else:
                niceFraction = 10

        return niceFraction * math.pow(10, exponent)

    def setMinMaxPoints(self, minPoint, maxPoint):
        self.minPoint = minPoint
        self.maxPoint = maxPoint
        self.calculate()

    def setMaxTicks(self, maxTicks):
        self.maxTicks = maxTicks
        self.calculate()


class camera:

    forward = False
    backward = False
    left = False
    right = False
    up = False
    down = False

    def update(self):
        rtheta = self.toRadian(self.theta)
        rphi = self.toRadian(self.phi)

        self.pos[0] = self.focus[0] + self.rad * n.cos(rtheta) * n.sin(rphi)
        self.pos[1] = self.focus[1] + self.rad * n.sin(rtheta)
        self.pos[2] = self.focus[2] + self.rad * n.cos(rtheta) * n.cos(rphi)

        self.moveFocus()

        GLU.gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                      self.focus[0], self.focus[1], self.focus[2],
                      0, 1, 0)

    def __init__(self, root):
        self.pos = [0, 0, 0]
        self.focus = root
        self.theta = 1.55
        self.phi = 1
        self.rad = 1
        self.movementspeed = 0.01

    def toRadian(self, angle):
        return angle * (n.pi / 180)

    def moveFocus(self):
        if(self.forward and not self.backward and not
           self.left and not self.right):
            self.pos[0] -= self.pos[0] * self.movementspeed
            self.pos[1] -= self.pos[1] * self.movementspeed
            self.pos[2] -= self.pos[2] * self.movementspeed

            self.focus[0] -= self.pos[0] * self.movementspeed
            self.focus[1] -= self.pos[1] * self.movementspeed
            self.focus[2] -= self.pos[2] * self.movementspeed
        if(not self.forward and self.backward and not
           self.left and not self.right):
            self.focus[0] += self.pos[0] * self.movementspeed
            self.focus[1] += self.pos[1] * self.movementspeed
            self.focus[2] += self.pos[2] * self.movementspeed
        if(not self.forward and not self.backward and
           self.left and not self.right):
            left = self.cross(self.pos, [0, 1, 0])
            self.focus[0] += left[0] * self.movementspeed
            self.focus[1] += left[1] * self.movementspeed
            self.focus[2] += left[2] * self.movementspeed
        if(not self.forward and not self.backward and not
           self.left and self.right):
            right = self.cross([0, 1, 0], self.pos)
            self.focus[0] += right[0] * self.movementspeed
            self.focus[1] += right[1] * self.movementspeed
            self.focus[2] += right[2] * self.movementspeed
        if(self.up and not self.down):
            left = self.cross(self.pos, [0, 1, 0])
            self.focus[1] += self.movementspeed
        if(not self.up and self.down):
            left = self.cross(self.pos, [0, 1, 0])
            self.focus[1] -= self.movementspeed

    def cross(self, a, b):
        cx = a[1]*b[2] - a[2]*b[1]
        cy = a[2]*b[0] - a[0]*b[2]
        cz = a[0]*b[1] - a[1]*b[0]
        return [cx, cy, cz]


class graphbuilder:
    Graph = None

    niceScaleArr = None

    fontchar = None
    fonti = None
    fonts = []
    w = []
    h = []

    textID = None

    textColour = [1, 1, 1]
    graphColour = textColour

    scalefactor = None

    def build(self, neuronobject, scalefactor=100):
        self.scalefactor = scalefactor
        minval, maxval = neuronobject.get_boundingbox()

        midpoint = self.vecSubtract(maxval,
                                    self.vecDiv(self.vecSubtract(maxval,
                                                                 minval),
                                                2))
        midpoint = [midpoint[0], midpoint[2], midpoint[1]]

        self.niceScaleArr = []
        for i in (0, 2, 1):
            self.niceScaleArr.append(tickScaler(minval[i], maxval[i]))

        self.buildGraph(self.niceScaleArr, self.graphColour)

        self.LoadFontImage()
        self.BuildFont()

        return self.Graph, self.scale3d(midpoint, self.scalefactor)

    def buildGraph(self, arr, lineColour):

        # Front
        self.Front = GL.glGenLists(1)
        GL.glNewList(self.Front, GL.GL_COMPILE)
        ticks = int(arr[0].noTicks + 1)
        for x in range(0, ticks):  # Left to Right
            p1 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMin,
                  arr[2].niceMax]
            p2 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMax,
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[1].noTicks)
        for y in range(0, ticks + 1):  # Bottom to Top
            p1 = [arr[0].niceMin,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMax]
            p2 = [arr[0].niceMax,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()
        # Back
        self.Back = GL.glGenLists(2)
        GL.glNewList(self.Back, GL.GL_COMPILE)
        ticks = int(arr[0].noTicks + 1)
        for x in range(0, ticks):  # Left to Right
            p1 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMin,
                  arr[2].niceMin]
            p2 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMax,
                  arr[2].niceMin]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[1].noTicks)
        for y in range(0, ticks + 1):  # Bottom to Top
            p1 = [arr[0].niceMin,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMin]
            p2 = [arr[0].niceMax,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMin]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()
        # Top
        self.Top = GL.glGenLists(3)
        GL.glNewList(self.Top, GL.GL_COMPILE)
        ticks = int(arr[0].noTicks + 1)
        for x in range(0, ticks):  # Left to Right
            p1 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMax,
                  arr[2].niceMin]
            p2 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMax,
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[2].noTicks)
        for z in range(0, ticks + 1):  # Front to Back
            p1 = [arr[0].niceMin,
                  arr[1].niceMax,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            p2 = [arr[0].niceMax,
                  arr[1].niceMax,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()
        # Bottom
        self.Bottom = GL.glGenLists(4)  # Left to Right
        GL.glNewList(self.Bottom, GL.GL_COMPILE)
        ticks = int(arr[0].noTicks + 1)
        for x in range(0, ticks):
            p1 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMin,
                  arr[2].niceMin]
            p2 = [arr[0].niceMin + (arr[0].tickSpacing * x),
                  arr[1].niceMin,
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[2].noTicks)
        for z in range(0, ticks + 1):  # Front to Back
            p1 = [arr[0].niceMin,
                  arr[1].niceMin,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            p2 = [arr[0].niceMax,
                  arr[1].niceMin,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()
        # Left
        self.Left = GL.glGenLists(5)
        GL.glNewList(self.Left, GL.GL_COMPILE)
        ticks = int(arr[2].noTicks)
        for z in range(0, ticks + 1):  # Front to Back
            p1 = [arr[0].niceMin,
                  arr[1].niceMin,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            p2 = [arr[0].niceMin,
                  arr[1].niceMax,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[1].noTicks)
        for y in range(0, ticks + 1):  # Bottom to Top
            p1 = [arr[0].niceMin,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMin]
            p2 = [arr[0].niceMin,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()
        # Right
        self.Right = GL.glGenLists(6)
        GL.glNewList(self.Right, GL.GL_COMPILE)
        ticks = int(arr[2].noTicks)
        for z in range(0, ticks + 1):  # Front to Back
            p1 = [arr[0].niceMax,
                  arr[1].niceMin,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            p2 = [arr[0].niceMax,
                  arr[1].niceMax,
                  arr[2].niceMin + (arr[2].tickSpacing * z)]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        ticks = int(arr[1].noTicks)
        for y in range(0, ticks + 1):  # Bottom to Top
            p1 = [arr[0].niceMax,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMin]
            p2 = [arr[0].niceMax,
                  arr[1].niceMin + (arr[1].tickSpacing * y),
                  arr[2].niceMax]
            self.drawLine(p1, p2, color=lineColour,
                          scalefactor=self.scalefactor)
        GL.glEndList()

        self.Graph = []
        self.Graph.append(self.Front)
        self.Graph.append(self.Back)
        self.Graph.append(self.Top)
        self.Graph.append(self.Bottom)
        self.Graph.append(self.Left)
        self.Graph.append(self.Right)

    def drawLine(self, p1, p2, color=[0, 0, 0],
                 linewidth=0.5, scalefactor=100):
        p1 = self.scale3d(p1, scalefactor)
        p2 = self.scale3d(p2, scalefactor)
        GL.glLineWidth(linewidth)
        GL.glColor3f(color[0], color[1], color[2])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(p1[0], p1[1], p1[2])
        GL.glVertex3f(p2[0], p2[1], p2[2])
        GL.glEnd()

    def determineVisGraphPlanes(self, phi, theta):
        '''
        Front 0
        Back 1
        Top 2
        Bottom 3
        Left 4
        Right 5
        '''
        planes = [0, 1, 2, 3, 4, 5]
        if phi >= -90 and phi < 90:
            planes.remove(0)
        if phi >= 90 or phi < -90:
            planes.remove(1)
        if phi >= -180 and phi < 0:
            planes.remove(4)
        if phi >= 0 and phi < 180:
            planes.remove(5)
        if theta >= 0:
            planes.remove(2)
        else:
            planes.remove(3)
        for i in planes:
            GL.glCallList(self.Graph[i])
        self.drawAxisLabels(planes)

    def drawAxisLabels(self, planes):

        edges = self.findEdges(planes)

        s = [0, 0, 0]
        for e in edges:
            pos = self.getPos(e)
            i = 0
            if e is 0 or e is 1 or e is 4 or e is 5:
                i = 0
            elif e is 2 or e is 3 or e is 6 or e is 7:
                i = 1
            else:
                i = 2
            pos = [item[i] for item in pos]

            s[i] = numpy.std(pos)

        s = max(self.scale3d(s, self.scalefactor)) * 0.3

        check = []
        corner = None
        first = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        last = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for e in edges:
            quadpos = self.getPos(e)
            for i in quadpos:
                message = None
                if e is 0 or e is 1 or e is 4 or e is 5:
                    message = i[0]
                elif e is 2 or e is 3 or e is 6 or e is 7:
                    message = i[1]
                else:
                    message = i[2]
                strarr = str(message)
                i1, i2, i3 = self.scale3d(i, self.scalefactor)
                if [i1, i2, i3] not in check:
                    wOff = float()
                    for ch in strarr:

                        ind = ord(ch)
                        index = self.fontchar.index(str(ind))

                        self.billboardLabel([i1, i2, i3], index, wOff, s)

                        wOff = wOff + (float(self.w[index]) * 0.01)
                        check.append([i1, i2, i3])
                else:
                    corner = [i1, i2, i3]
                if quadpos.index(i) is 0:
                    first[edges.index(e)] = [i1, i2, i3]
            last[edges.index(e)] = [i1, i2, i3]

        s = s + 0.5
        xyzOff = [self.scale(self.niceScaleArr[0].tickSpacing,
                  self.scalefactor) / 2,
                  self.scale(self.niceScaleArr[1].tickSpacing,
                  self.scalefactor) / 2,
                  self.scale(self.niceScaleArr[2].tickSpacing,
                  self.scalefactor) / 2]

        if not first[0] == corner:
            self.billboardLabel([first[0][0] - xyzOff[0],
                                first[0][1],
                                first[0][2]], 39, 0, s)
        if not first[1] == corner:
            self.billboardLabel([first[1][0],
                                first[1][1] - xyzOff[1],
                                first[1][2]], 41, 0, s)
        if not first[2] == corner:
            self.billboardLabel([first[2][0],
                                 first[2][1],
                                 first[2][2] - xyzOff[2]], 40, 0, s)

        if not last[0] == corner:
            self.billboardLabel([last[0][0] + xyzOff[1],
                                 last[0][1],
                                 last[0][2]], 39, 0, s)
        if not last[1] == corner:
            self.billboardLabel([last[1][0],
                                 last[1][1] + xyzOff[1],
                                 last[1][2]], 41, 0, s)
        if not last[2] == corner:
            self.billboardLabel([last[2][0],
                                 last[2][1],
                                 last[2][2] + xyzOff[2]], 40, 0, s)

    def scale3d(self, value, factor):
        return [(value[0] / factor), (value[1] / factor), (value[2] / factor)]

    def scale(self, value, factor):
        return value / factor

    def billboardLabel(self, objPos, character, offset, scale):

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glColor4f(self.textColour[0], self.textColour[1],
                     self.textColour[2], 1.0)

        GL.glEnable(GL.GL_TEXTURE_2D)

        GL.glPushMatrix()

        GL.glTranslate(objPos[0], objPos[1], objPos[2])

        modelview = (GL.GLfloat * 16)()
        mvm = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX, modelview)

        for i in (0, 1, 2):
            for j in (0, 1, 2):
                if i is j:
                    modelview[i * 4 + j] = 1
                else:
                    modelview[i * 4 + j] = 0

        GL.glLoadMatrixf(mvm)

        GL.glScalef(scale, scale, scale)
        GL.glTranslate(offset, 0, 0)

        GL.glCallList(self.fonts[character])

        GL.glPopMatrix()
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_BLEND)

    def toRadian(self, angle):
        return angle * (n.pi / 180)

    def LoadFontImage(self):
        im = Image.open("res/font_0.png").convert("RGBA")
        newData = []
        data = im.getdata()
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
        im.putdata(newData)

        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw",
                                                           "RGBA",
                                                           0, -1)

        self.textID = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textID)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_S,
                           GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_T,
                           GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexEnvf(GL.GL_TEXTURE_ENV,
                     GL.GL_TEXTURE_ENV_MODE,
                     GL.GL_BLEND)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, ix, iy, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image)

    def BuildFont(self):
        cx = None
        cy = None

        self.fontchar = []
        x = []
        y = []
        self.w = []
        self.h = []
        with open("res/font.fnt") as f:
            lines = f.readlines()
            for line in lines:
                words = line.split(' ')
                if words[0] == str('char'):
                    for word in words:
                        if 'id=' in word:
                            self.fontchar.append(word.split('=')[1])
                        if 'x=' in word:
                            x.append(word.split('=')[1])
                        if 'y=' in word:
                            y.append(word.split('=')[1])
                        if 'width=' in word:
                            self.w.append(word.split('=')[1])
                        if 'height=' in word:
                            self.h.append(word.split('=')[1])

        self.fonti = GL.glGenLists(69)
        self.fonts = []

        from decimal import Decimal, getcontext

        for loop in range(0, 69):

            getcontext().prec = 6
            c = (Decimal(1) / Decimal(256))
            cx = c * Decimal(int(x[loop]))
            cy = c * Decimal(int(y[loop]))
            cxw = cx + (c * Decimal(int(self.w[loop])))
            cyw = cy + (c * Decimal(int(self.h[loop])))
            s = 0.01
            cw = (float(self.w[loop]))*s
            ch = (float(self.h[loop]))*s

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.textID)
            GL.glNewList(self.fonti+loop, GL.GL_COMPILE)
            GL.glBegin(GL.GL_QUADS)

            GL.glTexCoord2f(float(cx), 1 - float(cyw))
            GL.glVertex2f(0, 0)
            GL.glTexCoord2f(float(cxw), 1 - float(cyw))
            GL.glVertex2f(cw, 0)
            GL.glTexCoord2f(float(cxw), 1 - float(cy))
            GL.glVertex2f(cw, ch)
            GL.glTexCoord2f(float(cx), 1 - float(cy))
            GL.glVertex2f(0, ch)

            GL.glEnd()

            GL.glEndList()

            self.fonts.append(self.fonti + loop)

    def getPos(self, edges):
        quadspos = []
        if edges is 0:  # FR_T
            ticks = int(self.niceScaleArr[0].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin +
                                 (self.niceScaleArr[0].tickSpacing * i),
                                 self.niceScaleArr[1].niceMax,
                                 self.niceScaleArr[2].niceMax])
        if edges is 1:  # FR_B
            ticks = int(self.niceScaleArr[0].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin +
                                 (self.niceScaleArr[0].tickSpacing * i),
                                 self.niceScaleArr[1].niceMin,
                                 self.niceScaleArr[2].niceMax])
        if edges is 2:  # FR_L
            ticks = int(self.niceScaleArr[1].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin,
                                 self.niceScaleArr[1].niceMin +
                                 (self.niceScaleArr[1].tickSpacing * i),
                                 self.niceScaleArr[2].niceMax])
        if edges is 3:  # FR_R
            ticks = int(self.niceScaleArr[1].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMax,
                                 self.niceScaleArr[1].niceMin +
                                 (self.niceScaleArr[1].tickSpacing * i),
                                 self.niceScaleArr[2].niceMax])
        if edges is 4:  # BA_T
            ticks = int(self.niceScaleArr[0].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin +
                                 (self.niceScaleArr[0].tickSpacing * i),
                                 self.niceScaleArr[1].niceMax,
                                 self.niceScaleArr[2].niceMin])
        if edges is 5:  # BA_B
            ticks = int(self.niceScaleArr[0].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin +
                                 (self.niceScaleArr[0].tickSpacing * i),
                                 self.niceScaleArr[1].niceMin,
                                 self.niceScaleArr[2].niceMin])
        if edges is 6:  # BA_L
            ticks = int(self.niceScaleArr[1].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin,
                                 self.niceScaleArr[1].niceMin +
                                 (self.niceScaleArr[1].tickSpacing * i),
                                 self.niceScaleArr[2].niceMin])
        if edges is 7:  # BA_R
            ticks = int(self.niceScaleArr[1].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMax,
                                 self.niceScaleArr[1].niceMin +
                                 (self.niceScaleArr[1].tickSpacing * i),
                                 self.niceScaleArr[2].niceMin])
        if edges is 8:  # TO_L
            ticks = int(self.niceScaleArr[2].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin,
                                 self.niceScaleArr[1].niceMax,
                                 self.niceScaleArr[2].niceMin +
                                 (self.niceScaleArr[2].tickSpacing * i)])
        if edges is 9:  # TO_R
            ticks = int(self.niceScaleArr[0].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMax,
                                 self.niceScaleArr[1].niceMax,
                                 self.niceScaleArr[2].niceMin +
                                 (self.niceScaleArr[2].tickSpacing * i)])
        if edges is 10:  # BO_L
            ticks = int(self.niceScaleArr[2].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMin,
                                 self.niceScaleArr[1].niceMin,
                                 self.niceScaleArr[2].niceMin +
                                 (self.niceScaleArr[2].tickSpacing * i)])
        if edges is 11:  # BO_R
            ticks = int(self.niceScaleArr[2].noTicks + 1)
            for i in range(0, ticks):
                quadspos.append([self.niceScaleArr[0].niceMax,
                                 self.niceScaleArr[1].niceMin,
                                 self.niceScaleArr[2].niceMin +
                                 (self.niceScaleArr[2].tickSpacing * i)])
        return quadspos

    def findEdges(self, planes):
        edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        '''
        FR_T 0
        FR_B 1
        FR_L 2
        FR_R 3
        BA_T 4
        BA_B 5
        BA_L 6
        BA_R 7
        TO_L 8
        TO_R 9
        BO_L 10
        BO_R 11
        '''
        if 0 not in planes:
            if 0 in edges:
                edges.remove(0)
            if 1 in edges:
                edges.remove(1)
            if 2 in edges:
                edges.remove(2)
            if 3 in edges:
                edges.remove(3)
        elif 1 not in planes:
            if 4 in edges:
                edges.remove(4)
            if 5 in edges:
                edges.remove(5)
            if 6 in edges:
                edges.remove(6)
            if 7 in edges:
                edges.remove(7)
        if 2 not in planes:
            if 0 in edges:
                edges.remove(0)
            if 4 in edges:
                edges.remove(4)
            if 8 in edges:
                edges.remove(8)
            if 9 in edges:
                edges.remove(9)
        elif 3 not in planes:
            if 1 in edges:
                edges.remove(1)
            if 5 in edges:
                edges.remove(5)
            if 10 in edges:
                edges.remove(10)
            if 11 in edges:
                edges.remove(11)
        if 4 not in planes:
            if 2 in edges:
                edges.remove(2)
            if 6 in edges:
                edges.remove(6)
            if 8 in edges:
                edges.remove(8)
            if 10 in edges:
                edges.remove(10)
        elif 5 not in planes:
            if 3 in edges:
                edges.remove(3)
            if 7 in edges:
                edges.remove(7)
            if 9 in edges:
                edges.remove(9)
            if 11 in edges:
                edges.remove(11)

        return edges

    def vecSubtract(self, a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def vecDiv(self, a, b):
        return [a[0] / b, a[1] / b, a[2] / b]


class modelbuilder:

    index = None
    lists = []
    root = [0, 0, 0]
    vertices_gl = []
    colors_gl = []
    scalefactor = None
    Neuron = None

    def build(self, neuronobject, poly=True, scalefactor=100, fast=False):
        self.scalefactor = scalefactor
        if poly:
            self.LoadPolyObject(neuronobject, fast)
        else:
            self.LoadLineObject(neuronobject)
        return self.root, self.vertices_gl, self.colors_gl, self.Neuron

    def LoadLineObject(self, neuron):
        from btstructs import NeuronMorphology
        from btstructs import ForestStructure

        if isinstance(neuron, NeuronMorphology):
            result = self.LoadLineNeuron(neuron)
            self.vertices_gl = numpy.array.vbo.VBO(result[0])
            self.colors_gl = numpy.array.vbo.VBO(result[1])

        elif isinstance(neuron, ForestStructure):
            vertices = None
            colors = None

            for item in neuron.neurons:
                v, c = self.LoadLineNeuron(item[0], offset=item[1])
                if vertices is None:
                    vertices = v
                else:
                    vertices = n.append(vertices, v)
                if colors is None:
                    colors = c
                else:
                    colors = n.append(colors, c)
            self.vertices_gl = numpy.array.vbo.VBO(vertices)
            self.colors_gl = numpy.array.vbo.VBO(colors)

    def LoadPolyObject(self, neuron, fast=False):
        from btstructs import NeuronMorphology
        from btstructs import ForestStructure

        if isinstance(neuron, NeuronMorphology):
            self.Neuron = GL.glGenLists(1)
            GL.glNewList(self.Neuron, GL.GL_COMPILE)
            self.LoadPolyNeuron(neuron, fast)
            GL.glEndList()
            self.lists.append(self.Neuron)
        elif isinstance(neuron, ForestStructure):
            self.Neuron = GL.glGenLists(len(neuron.neurons))
            GL.glNewList(self.Neuron, GL.GL_COMPILE)
            i = 0
            for item in neuron.neurons:
                self.LoadPolyNeuron(item, i, fast)
                self.lists.append(self.Neuron + i)
                i += 1
            GL.glEndList()

    def LoadLineNeuron(self, neuron, offset=[0, 0, 0]):
        my_color_list = [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]
        vertices = []
        colors = []

        self.root = self.scale(neuron.tree.get_root().content['p3d'].xyz,
                               self.scalefactor)
        index = 0
        for node in neuron.tree:
            # not ordered but that has little importance here
            # draw a poly segment from parent to current point
            # print 'index: ', index, ' -> ', current_SWC
            c_x = self.scale(node.content['p3d'].xyz[0] + offset[0],
                             self.scalefactor)
            c_y = self.scale(node.content['p3d'].xyz[2] + offset[2],
                             self.scalefactor)
            c_z = self.scale(node.content['p3d'].xyz[1] + offset[1],
                             self.scalefactor)
            if index <= 3:
                index += 1
                print 'do not draw the soma and its CNG,\
                       !!! 2 !!! point descriptions'
            else:
                parent = node.parent
                p_x = self.scale(parent.content['p3d'].xyz[0] + offset[0],
                                 self.scalefactor)
                p_y = self.scale(parent.content['p3d'].xyz[2] + offset[2],
                                 self.scalefactor)
                p_z = self.scale(parent.content['p3d'].xyz[1] + offset[1],
                                 self.scalefactor)
                print 'index:', index

                vertices.append([p_x, p_z, p_y])
                vertices.append([c_x, c_z, c_y])

                colors.append(my_color_list[node.content['p3d'].segtype-1])
                colors.append(my_color_list[node.content['p3d'].segtype-1])

        return n.array(vertices, dtype=n.float32), \
            n.array(colors, dtype=n.float32)

    def LoadPolyNeuron(self, neuron, index=0, fast=False):
        from btstructs import NeuronMorphology

        offset = [0, 0, 0]
        my_color_list = [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 1.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]
        if not isinstance(neuron, NeuronMorphology):
            offset = neuron[1]
            neuron = neuron[0]
        if index == 0:
            self.root = self.scale(neuron.tree.get_root().content['p3d'].xyz +
                                   offset, self.scalefactor)

        GLE.gleSetJoinStyle(GLE.TUBE_NORM_MASK | GLE.TUBE_JN_ROUND |
                            GLE.TUBE_JN_CAP)
        segments = []
        if fast:
            segments = neuron.tree.get_segments_fast()
        elif not fast:
            segments = neuron.tree.get_segments()
        for s in segments:
            point_array = []
            colour_array = []
            radius_array = []

            for node in s:
                point_array.append([self.scale(node.content['p3d'].xyz[0] +
                                               offset[0], self.scalefactor),
                                   self.scale(node.content['p3d'].xyz[2] +
                                              offset[1], self.scalefactor),
                                   self.scale(node.content['p3d'].xyz[1] +
                                              offset[2], self.scalefactor)])
                colour_array.append(my_color_list[
                                              node.content['p3d'].segtype-1])
                radius_array.append(self.scale(node.content['p3d'].radius,
                                               self.scalefactor))
            if len(point_array) >= 2:
                point_array.insert(0, point_array[0])
                point_array.append(point_array[len(point_array) - 1])
                colour_array.insert(0, colour_array[0])
                colour_array.append(colour_array[len(colour_array) - 1])
                radius_array.insert(0, radius_array[0])
                radius_array.append(radius_array[len(radius_array) - 1])
                GLE.glePolyCone(point_array, colour_array, radius_array)

    def scale(self, value, factor):
        return value / factor


class btvizGL:
    running = True

    leftMouseDown = False
    rotate = False
    mouseXY = [0, 0]
    camera = None

    index = GL.GLuint()
    lists = []

    poly = False
    scalefactor = 100

    root = [0, 0, 0]
    displaysize = (0, 0)

    animate = False
    imagecount = 0
    filename = None
    axis = 'z'

    rotation = 0

    showStats = True

    graph = False

    backgroundColour = [0, 0, 0]

    Neuron = None
    Graph = None

    gb = None

    def drawLine(self, parent, child, color, linewidth):
        GL.glLineWidth(linewidth)
        GL.glColor3f(color[0], color[1], color[2])
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(parent[0], parent[1], parent[2])
        GL.glVertex3f(child[0], child[1], child[2])
        GL.glEnd()

    def RenderText(self, text, x, y):
        GL.glColor3f(1.0, 0.0, 0.0)
        GL.glWindowPos2f(x, y)
        GLUT.glutBitmapString(GLUT.GLUT_BITMAP_HELVETICA_10, text)

    def initGL(self, multisample):
        GL.glShadeModel(GL.GL_SMOOTH)

        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        GL.glClearColor(self.backgroundColour[0], self.backgroundColour[1],
                        self.backgroundColour[2], 1)

        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        if multisample:
            GL.glEnable(GL.GL_MULTISAMPLE)

        self.camera = camera(self.root)

    def display(self):
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()

        self.camera.update()

        self.drawObjects()

        if self.showStats:
            self.drawText()
        GLUT.glutSwapBuffers()

        if self.Animate:
                self.SaveImage(self.imagecount)
                self.imagecount += 1
                self.camera.phi += 1
                if self.imagecount == 360:

                    filenames = listdir('captures/images/')
                    filenames = [int(x.split(".")[0]) for x in filenames]
                    filenames.sort()
                    filenames = [str(x) + ".bmp" for x in filenames]

                    with imageio.get_writer('captures/animations/' +
                                            self.filename +
                                            '.gif', mode='I',
                                            fps=30) as writer:
                        from os import remove
                        for filename in filenames:
                            image = imageio.imread('captures/images/' +
                                                   filename)
                            writer.append_data(image)
                            remove('captures/images/' + filename)
                        GLUT.glutSetOption(GLUT.
                                           GLUT_ACTION_ON_WINDOW_CLOSE,
                                           GLUT.
                                           GLUT_ACTION_GLUTMAINLOOP_RETURNS)
                        self.running = False
                GLUT.glutPostRedisplay()

    def drawObjects(self):
        if self.poly:
            if self.graph:
                self.gb.determineVisGraphPlanes(self.camera.phi,
                                                self.camera.theta)

            if self.animate:
                if self.axis is 'x':
                    GL.glRotate(self.rotation, 1, 0, 0)
                if self.axis is 'y':
                    GL.glRotate(self.rotation, 0, 0, 1)
                if self.axis is 'z':
                    GL.glRotate(self.rotation, 0, 1, 0)
                self.rotation += 1

            GL.glCallList(self.Neuron)

        else:
            if len(self.vertices_gl) > 0:
                GL.glPushMatrix()
                self.vertices_gl.bind()

                GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                GL.glVertexPointer(3, GLUT.GL_FLOAT, 0, self.vertices_gl)

                self.colors_gl.bind()
                GL.glEnableClientState(GL.GL_COLOR_ARRAY)
                GL.glColorPointer(3, GLUT.GL_FLOAT, 0, self.colors_gl)

                GL.glDrawArrays(GL.GL_LINES, 0, len(self.vertices_gl))
                GL.glPopMatrix()

    def drawText(self):
        self.RenderText(('CamPos %f,%f,%f') % (self.camera.pos[0],
                                               self.camera.pos[1],
                                               self.camera.pos[2]),
                        10, 575)
        self.RenderText(('theta %f, phi %f') % (self.camera.theta,
                                                self.camera.phi),
                        10, 555)
        self.RenderText(('radius %f') % (self.camera.rad), 10, 495)
        self.RenderText(('CentrePos %f,%f,%f') % (self.camera.focus[0],
                                                  self.camera.focus[1],
                                                  self.camera.focus[2]),
                        10, 515)
        self.RenderText(('MousePos %f,%f') % (self.mouseXY[0],
                                              self.mouseXY[1]),
                        10, 535)
        self.RenderText(('rotate ' + self.rotate.__str__()),
                        10, 475)

    def reshape(self, width, height):

        if height == 0:
            height = 1
        aspect = width / height

        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(90, aspect, 0.01, 100)

    def mouse(self, button, state, x, y):
        self.rotate = False
        if button == GLUT.GLUT_LEFT_BUTTON and state == GLUT.GLUT_DOWN:
            self.leftMouseDown = True
            self.mouseXY = [x, y]
            self.rotate = True
        elif button == GLUT.GLUT_LEFT_BUTTON and state == GLUT.GLUT_UP:
            self.leftMouseDown = False

    def mouseMove(self, x, y):

        if(self.rotate):
            self.camera.phi += (x - self.mouseXY[0]) * 0.1

            if self.camera.phi > 180:
                self.camera.phi = self.camera.phi - 360
            elif self.camera.phi < -180:
                self.camera.phi = self.camera.phi + 360

            self.camera.theta = self.clamp(self.camera.theta +
                                           (y - self.mouseXY[1]) *
                                           0.1, -89.999, 89.999)
        self.mouseXY = [x, y]
        GLUT.glutPostRedisplay()

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def mouseWheel(self, wheel, direction, x, y):
        if direction > 0:
            self.camera.rad = max(0.1, self.camera.rad - 0.1)
        elif direction < 0:
            self.camera.rad = min(100, self.camera.rad + 0.1)

        GLUT.glutPostRedisplay()

    def keyDown(self, key, x, y):
        '''
        if key == 'w':
            self.camera.forward = True

        if key == 's':
            self.camera.backward = True

        if key == 'a':
            self.camera.left = True

        if key == 'd':
            self.camera.right = True
        '''
        if key == 'q':
            self.camera.up = True

        if key == 'e':
            self.camera.down = True

        GLUT.glutPostRedisplay()

    def keyUp(self, key, x, y):
        '''
        if key == 'w':
            self.camera.forward = False

        if key == 's':
            self.camera.backward = False

        if key == 'a':
            self.camera.left = False

        if key == 'd':
            self.camera.right = False
        '''
        if key == 'q':
            self.camera.up = False

        if key == 'e':
            self.camera.down = False

        if key == 'r':
            self.camera.focus = self.root
            self.camera.phi = 0
            self.camera.theta = 0

        if key == 'p':
            self.SaveImage(0)

        if key == 'h':
            self.showStats = not self.showStats

        if key == 'g':
            self.graph = not self.graph

        if key == 'q':
            self.running = False

        GLUT.glutPostRedisplay()

    def SaveImage(self, index):

        buff = GL.glReadPixels(0, 0, self.displaysize[0],
                               self.displaysize[1],
                               GL.GL_RGB,
                               GLUT.GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB",
                                size=(self.displaysize[0],
                                      self.displaysize[1]),
                                data=buff)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save('captures/images/' + str(index) + '.bmp')

    def Animate(self, neuronObject, filename,
                displaysize=(800, 600), radius=5,
                poly=True, axis='z', graph=False):

        self.Animate = True
        self.filename = filename
        self.axis = axis
        self.showStats = False
        self.main(neuronObject, displaysize, radius, poly, False, True, graph)

    def Plot(self, neuronObject, displaysize=(800, 600), radius=5, poly=True,
             fast=False, multisample=True, graph=True):
        self.Animate = False
        self.showStats = True
        self.main(neuronObject, displaysize, radius,
                  poly, fast, multisample, graph)

    def main(self, neuronObject, displaysize=(800, 600), radius=5, poly=True,
             fast=False, multisample=True, graph=True):

        self.poly = poly
        self.displaysize = displaysize
        self.graph = graph

        title = 'btmorph OpenGL Viewer'

        GLUT.glutInit(sys.argv)
        if multisample:
            GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE)
        else:
            GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                     GLUT.GLUT_DOUBLE |
                                     GLUT.GLUT_MULTISAMPLE)
        GLUT.glutInitWindowSize(self.displaysize[0], self.displaysize[1])
        GLUT.glutInitWindowPosition(50, 50)
        GLUT.glutCreateWindow(title)
        GLUT.glutDisplayFunc(self.display)
        GLUT.glutReshapeFunc(self.reshape)
        GLUT.glutMouseFunc(self.mouse)
        GLUT.glutMotionFunc(self.mouseMove)
        GLUT.glutMouseWheelFunc(self.mouseWheel)
        GLUT.glutKeyboardFunc(self.keyDown)
        GLUT.glutKeyboardUpFunc(self.keyUp)

        mb = modelbuilder()
        self.root, self.vertices_gl, self.colors_gl, self.Neuron = \
            mb.build(neuronObject, self.poly, 100, fast)
        if graph:
            self.gb = graphbuilder()
            self.Graph, mid = \
                self.gb.build(neuronObject, scalefactor=100)

        self.initGL(multisample)
        self.camera.rad = radius
        self.camera.focus = mid

        while self.running:
            GLUT.glutMainLoopEvent()

        GLUT.glutDestroyWindow(GLUT.glutGetWindow())
