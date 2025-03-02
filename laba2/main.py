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


# Задание 8
def otris_treyg(img_mat, x0, y0, x1, y1, x2, y2, color):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    if (xmin < 0):
        xmin = 0
    if (ymin < 0):
        ymin = 0
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)
    for x in range(int(xmin),int(xmax)):
        for y in range (int(ymin), int(ymax)):
            l2, l1, l0 = bar_kor(x, y, x0, y0, x1, y1, x2, y2)
            if (l0 >= 0 and l1 >=0 and l2 >= 0):
                img_mat[y, x] = color

def otris_treyg2(img_mat, x0, y0,z0, x1, y1,z1, x2, y2,z2, zb, color):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    if (xmin < 0):
        xmin = 0
    if (ymin < 0):
        ymin = 0
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)

    for x in range(int(xmin),int(xmax)):
        for y in range (int(ymin), int(ymax)):
            l0, l1, l2 = bar_kor(x, y, x0, y0, x1, y1, x2, y2)
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


#задание 9
img_mat = np.zeros((200, 200, 3), dtype = np.uint8)
otris_treyg(img_mat, 4, 4, 5, 45, 23, 86, 255)
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img1.png')


f = open('model_1.obj')
mas =[]
mass = []
for s in f:
    splitted = s.split()
    if (splitted[0] == 'v'):
        mas.append([float(splitted[1]), float(splitted[2]),float(splitted[3])])
    if (splitted[0] == 'f'):
        xx0= splitted[1].split('/')
        yy0 = splitted[2].split('/')
        zz0 = splitted[3].split('/')
        mass.append([int(xx0[0]), int(yy0[0]), int(zz0[0])])
f.close()

img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
for i in range (len(mass)):
    random_number = random.randint(0, 255)
    x0 = mas[mass[i][0]-1][0]*5000*4+2000
    y0 = mas[mass[i][0]-1][1]*5000*4+2000
    x1 = mas[mass[i][1]-1][0]*5000*4+2000
    y1 = mas[mass[i][1]-1][1]*5000*4+2000
    x2 = mas[mass[i][2]-1][0]*5000*4+2000
    y2 = mas[mass[i][2]-1][1]*5000*4+2000
    otris_treyg(img_mat, x0, y0, x1, y1, x2, y2, random_number)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img2.png')


f = open('model_1.obj')
mas =[]
mass = []
for s in f:
    splitted = s.split()
    if (splitted[0] == 'v'):
        mas.append([float(splitted[1]), float(splitted[2]),float(splitted[3])])
    if (splitted[0] == 'f'):
        xx0= splitted[1].split('/')
        yy0 = splitted[2].split('/')
        zz0 = splitted[3].split('/')
        mass.append([int(xx0[0]), int(yy0[0]), int(zz0[0])])
f.close()

img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
for i in range (len(mass)):
    random_number = random.randint(0, 255)
    x0 = mas[mass[i][0]-1][0]*5000*4+2000
    y0 = mas[mass[i][0]-1][1]*5000*4+2000
    x1 = mas[mass[i][1]-1][0]*5000*4+2000
    y1 = mas[mass[i][1]-1][1]*5000*4+2000
    x2 = mas[mass[i][2]-1][0]*5000*4+2000
    y2 = mas[mass[i][2]-1][1]*5000*4+2000
    z0= mas[mass[i][0]-1][2]*5000*4+2000
    z1= mas[mass[i][1]-1][2]*5000*4+2000
    z2= mas[mass[i][2]-1][2]*5000*4+2000
    n1, n2, n3 = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    otris_treyg(img_mat, x0, y0, x1, y1, x2, y2, random_number)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img2.png')

#задание 12
img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
for i in range (len(mass)):
    random_number = random.randint(0, 255)
    x0 = mas[mass[i][0]-1][0]*5000*4+2000
    y0 = mas[mass[i][0]-1][1]*5000*4+2000
    x1 = mas[mass[i][1]-1][0]*5000*4+2000
    y1 = mas[mass[i][1]-1][1]*5000*4+2000
    x2 = mas[mass[i][2]-1][0]*5000*4+2000
    y2 = mas[mass[i][2]-1][1]*5000*4+2000
    z0= mas[mass[i][0]-1][2]*5000*4+2000
    z1= mas[mass[i][1]-1][2]*5000*4+2000
    z2= mas[mass[i][2]-1][2]*5000*4+2000
    if (cosinus(x0,y0,z0,x1,y1,z1,x2,y2,z2)<0):
        otris_treyg(img_mat, x0, y0, x1, y1, x2, y2, random_number)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img3.png')

#Задание 13
img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
for i in range (len(mass)):
    random_number = random.randint(0, 255)
    x0 = mas[mass[i][0]-1][0]*5000*4+2000
    y0 = mas[mass[i][0]-1][1]*5000*4+2000
    x1 = mas[mass[i][1]-1][0]*5000*4+2000
    y1 = mas[mass[i][1]-1][1]*5000*4+2000
    x2 = mas[mass[i][2]-1][0]*5000*4+2000
    y2 = mas[mass[i][2]-1][1]*5000*4+2000
    z0= mas[mass[i][0]-1][2]*5000*4+2000
    z1= mas[mass[i][1]-1][2]*5000*4+2000
    z2= mas[mass[i][2]-1][2]*5000*4+2000
    if (cosinus(x0,y0,z0,x1,y1,z1,x2,y2,z2)<0):
        otris_treyg(img_mat, x0, y0, x1, y1, x2, y2, (-255 * cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2), 0, 0))


img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img4.png')

#Задание 14
img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
for i in range (len(mass)):
    random_number = random.randint(0, 255)
    x0 = mas[mass[i][0]-1][0]*5000*4+2000
    y0 = mas[mass[i][0]-1][1]*5000*4+2000
    x1 = mas[mass[i][1]-1][0]*5000*4+2000
    y1 = mas[mass[i][1]-1][1]*5000*4+2000
    x2 = mas[mass[i][2]-1][0]*5000*4+2000
    y2 = mas[mass[i][2]-1][1]*5000*4+2000
    z0= mas[mass[i][0]-1][2]*5000*4+2000
    z1= mas[mass[i][1]-1][2]*5000*4+2000
    z2= mas[mass[i][2]-1][2]*5000*4+2000
    if (cosinus(x0,y0,z0,x1,y1,z1,x2,y2,z2)<0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,(-255*cosinus(x0,y0,z0,x1,y1,z1,x2,y2,z2),0,0))

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img5.png')
