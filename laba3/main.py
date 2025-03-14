from PIL import Image, ImageOps
import numpy as np
import math
import random

def bresenham_line(self, image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
            derror += dy
        if (derror > (x1 -x0)):
            derror -= 2 * (x1 -x0)
            y += y_update

#задание 7
def bar_kor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def otris_treyg2(img_mat, x0, y0,z0, x1, y1,z1, x2, y2,z2, zb, color,a):
    p_x0 = a*5000*4 * x0/z0 + 2000
    p_x1 = a*5000*4 * x1/z1 + 2000
    p_x2 = a*5000*4 * x2/z2 + 2000
    p_y0 = a*5000*4 * y0/z0 + 2000
    p_y1 = a*5000*4 * y1/z1 + 2000
    p_y2 = a*5000*4 * y2/z2 + 2000
    xmin = min(p_x0, p_x1, p_x2)
    ymin = min(p_y0, p_y1, p_y2)
    if (xmin < 0):
        xmin = 0
    if (ymin < 0):
        ymin = 0
    xmax = max(p_x0, p_x1, p_x2)
    ymax = max(p_y0, p_y1, p_y2)
    for x in range(int(xmin),int(xmax)):
        for y in range (int(ymin), int(ymax)):
            l0, l1, l2 = bar_kor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
            if (l0 >= 0 and l1 >=0 and l2 >= 0):
                z_coord = l0*z0+l1*z1+l2*z2
                if (z_coord<zb[x][y]):
                    img_mat[y, x] = color
                    zb[x][y] = z_coord


# Задание 11
def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    nx=(y1-y2)*(z1-z0)-(z1-z2)*(y1-y0)
    ny =(z1-z2)*(x1-x0)-(x1-x2)*(z1-z0)
    nz =(x1-x2)*(y1-y0)-(y1-y2)*(x1-x0)
    return [nx, ny,nz]

#Задание 12 косинус
def cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0,0,1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    return np.dot(l, n)/ (np.linalg.norm(l)*np.linalg.norm(n))

#Задание 15
def rotate(mass_ver,alfa,beta,gamma,tx,ty,tz):
    rx = np.array([[1,0,0],[0,math.cos(alfa),math.sin(alfa)],[0,-math.sin(alfa),math.cos(alfa)]])
    ry = np.array([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    rz = np.array([[math.cos(gamma), math.sin(gamma),0],[-math.sin(gamma), math.cos(gamma), 0],[0,0,1]])
    r = np.dot(np.dot(rx,ry), rz)
    mass_rot = []
    for i in range(len(mass_ver)):
        mass_rot.append(np.dot(r,mas_ver[i])+[tx,ty,tz])
    return mass_rot

f = open('model_1.obj')
mas_ver =[]
mass_pol = []
for s in f:
    splitted = s.split()
    if (splitted[0] == 'v'):
        mas_ver.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
    if (splitted[0] == 'f'):
        xx0= splitted[1].split('/')
        yy0 = splitted[2].split('/')
        zz0 = splitted[3].split('/')
        mass_pol.append([int(xx0[0]), int(yy0[0]), int(zz0[0])])
f.close()



img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,1)
for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0),1)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img1.png')

#Задание 16
img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,0.7)
for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0),0.7)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img2.png')

img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,0.4)
for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0),0.4)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img3.png')

img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,0.3)
for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0),0.3)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img4.png')

img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,0.2)
for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0),0.2)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img5.png')