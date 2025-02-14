import math
import numpy as np
from PIL import Image, ImageOps

def draw_line(self, image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round ((1.0-t)*x0 + t*x1)
        y = round ((1.0-t)*y0 + t*y1)
        image[y, x] = color

def draw_line2(self, image, x0, y0, x1, y1, color):
    count =math.sqrt((x0 -x1)**2 + (y0 -y1)**2)
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round ((1.0-t)*x0 + t*x1)
        y = round ((1.0-t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line(self, image, x0, y0, x1, y1, color):
    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        image[y, x] = color

def x_loop_linefix1(self, image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        image[y, x] = color

def x_loop_linefix2(self, image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0- x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange= True

    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_linev2(self, image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def x_loop_line_no_calc_y(self, image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1- y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror+= dy
        if (derror > 0.5):
            derror-= 1.0
            y += y_update

def x_loop_line_no_y_calc2v(self, image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0 * (x1 -x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range (x0, x1):
        t = (x-x0)/(x1 -x0)
        y = round ((1.0-t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
            derror+= dy
        if (derror > 2.0 * (x1 -x0) * 0.5):
            derror-= 2.0 * (x1 -x0) * 1.0
            y += y_update

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

#1 задание
img_mat = np.zeros((600,800), dtype = np.uint8)
img = Image.fromarray(img_mat, mode = 'L')
img.save('img.png')

img_mat1 = np.zeros((600,800), dtype = np.uint8)
img_mat1[0:600, 0:800] = 255
img1 = Image.fromarray(img_mat1, mode = 'L')
img1.save('img1.png')

img_mat2 = np.zeros((600,800,3), dtype = np.uint8)
img_mat2[0:600, 0:800,0] = 255
img2 = Image.fromarray(img_mat2, mode = 'RGB')
img2.save('img2.png')

img_mat3 = np.zeros((600,800,3), dtype = np.uint8)
for i in range(600):
    for j in range(800):
        img_mat3[i,j] = i+j
img3 = Image.fromarray(img_mat3, mode = 'RGB')
img3.save('img3.png')


#задание 2
img_mat4 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    draw_line(0, img_mat4, x0,y0,x1,y1, 20, 255)

img4 = Image.fromarray(img_mat4, mode = 'RGB')
img4.save('img4.png')

img_mat5 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    draw_line2(0, img_mat5, x0,y0,x1,y1, 255)

img5 = Image.fromarray(img_mat5, mode = 'RGB')
img5.save('img5.png')

img_mat6 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_line(0, img_mat6, x0,y0,x1,y1, 255)

img6 = Image.fromarray(img_mat6, mode = 'RGB')
img6.save('img6.png')

img_mat7 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_linefix1(0, img_mat7, x0,y0,x1,y1, 255)

img7 = Image.fromarray(img_mat7, mode = 'RGB')
img7.save('img7.png')

img_mat8 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_linefix2(0, img_mat8, x0,y0,x1,y1, 255)

img8 = Image.fromarray(img_mat8, mode = 'RGB')
img8.save('img8.png')

img_mat9 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_linev2(0, img_mat9, x0,y0,x1,y1, 255)

img9 = Image.fromarray(img_mat9, mode = 'RGB')
img9.save('img9.png')

img_mat10 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_line_no_calc_y(0, img_mat10, x0,y0,x1,y1, 255)

img10 = Image.fromarray(img_mat10, mode = 'RGB')
img10.save('img10.png')

img_mat11 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    x_loop_line_no_y_calc2v(0, img_mat11, x0,y0,x1,y1, 255)

img11 = Image.fromarray(img_mat11, mode = 'RGB')
img11.save('img11.png')

img_mat12 = np.zeros((200,200,3), dtype = np.uint8)

for i in range(13):
    x0=100
    y0=100
    x1 = int(100+95*math.cos(i*2*math.pi/13))
    y1 = int(100+95*math.sin(i*2*math.pi/13))
    bresenham_line(0, img_mat12, x0,y0,x1,y1, 255)

img12 = Image.fromarray(img_mat12, mode = 'RGB')
img12.save('img12.png')

#Задание 3
f = open("model_1.obj")
coord = []
for s in f:
    splitted = s.split()
    if splitted[0] == 'v':
        coord.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
f.close()

#Задание 4

img_mat13 = np.zeros((1000,1000), dtype = np.uint8)
for i in coord:
    x = i[0]*5000+500
    y = i[1]*5000+500
    img_mat13[int(y), int(x)] = 255

img13 = Image.fromarray(img_mat13, mode = 'L')
img13 = ImageOps.flip(img13)
img13.save('img13.png')

f = open("model_1.obj")
coord_pal = []
for s in f:
    splitted = s.split()
    if splitted[0] == 'f':
        splittedx = splitted[1].split('/')
        splittedy = splitted[2].split('/')
        splittedz = splitted[3].split('/')
        coord_pal.append([int(splittedx[0]),int(splittedy[0]),int(splittedz[0])])
f.close()

#Задание 6
img_mat14 = np.zeros((4000,4000), dtype = np.uint8)
for i in range(len(coord_pal)):
    x0 = int(coord[coord_pal[i][0]-1][0]*20000 +2000)
    y0 = int(coord[coord_pal[i][0]-1][1]*20000+2000)
    x1 = int(coord[coord_pal[i][1]-1][0]*20000+2000)
    y1 = int(coord[coord_pal[i][1]-1][1]*20000+2000)
    x2 = int(coord[coord_pal[i][2]-1][0]*20000+2000)
    y2 = int(coord[coord_pal[i][2]-1][1]*20000+2000)
    bresenham_line(0, img_mat14, x0, y0, x1, y1,255)
    bresenham_line(0, img_mat14, x0, y0, x2, y2, 255)
    bresenham_line(0, img_mat14, x1, y1, x2, y2, 255)

img14 = Image.fromarray(img_mat14, mode = 'L')
img14 = ImageOps.flip(img14)
img14.save('img14.png')
