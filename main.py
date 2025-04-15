from PIL import Image, ImageOps
import numpy as np
import math

#задание 7
def bar_kor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def otris_treyg2(img_mat, x0, y0,z0, x1, y1,z1, x2, y2,z2, zb, I0,I1,I2,mas_text,text_mat,a,b):
    p_x0 = a * x0/z0 + b
    p_x1 = a * x1/z1 + b
    p_x2 = a * x2/z2 + b
    p_y0 = a * y0/z0 + b
    p_y1 = a * y1/z1 + b
    p_y2 = a * y2/z2 + b
    xmin = math.floor(min(p_x0, p_x1, p_x2))
    ymin = math.floor(min(p_y0, p_y1, p_y2))
    if (xmin < 0):
        xmin = 0
    if (ymin < 0):
        ymin = 0
    xmax = math.ceil(max(p_x0, p_x1, p_x2))
    ymax = math.ceil(max(p_y0, p_y1, p_y2))
    for x in range(xmin,xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bar_kor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
            if (l0 >= 0 and l1 >=0 and l2 >= 0):
                z_coord = l0*z0+l1*z1+l2*z2
                if (z_coord<zb[x][y]):
                    intensivity = -(I0*l0+I1*l1+I2*l2)
                    if (intensivity<0):
                        intensivity = 0
                    img_mat[y,x] = intensivity*text_mat[int(1024*( l0*mas_text[0][1]+l1*mas_text[1][1]+l2*mas_text[2][1])),int(1024*(l0*mas_text[0][0]+l1*mas_text[1][0]+l2*mas_text[2][0]))]
                    #img_mat[y, x] = (intensivity*color[0],intensivity*color[1],intensivity*color[2])
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
mass_n_text=[]
mas_text = []
l=[0,0,1]
for s in f:
    if len(s)==0:
        continue
    splitted = s.split()
    if (splitted[0] == 'v'):
        mas_ver.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
    if (splitted[0] == 'f'):
        xx0= splitted[1].split('/')
        yy0 = splitted[2].split('/')
        zz0 = splitted[3].split('/')
        mass_pol.append([int(xx0[0]), int(yy0[0]), int(zz0[0])])
        mass_n_text.append([int(xx0[1]), int(yy0[1]), int(zz0[1])])
    if (splitted[0] == 'vt'):
        mas_text.append([float(splitted[1]), float(splitted[2])])
f.close()

rotate_mas = rotate(mas_ver,math.pi,math.pi,2,0.045,-0.01,1)

vn_calc=np.full((len(mas_ver),3),0,dtype=np.float32)

for i in range(len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0] - 1][0]
    y0 = rotate_mas[mass_pol[i][0] - 1][1]
    x1 = rotate_mas[mass_pol[i][1] - 1][0]
    y1 = rotate_mas[mass_pol[i][1] - 1][1]
    x2 = rotate_mas[mass_pol[i][2] - 1][0]
    y2 = rotate_mas[mass_pol[i][2] - 1][1]
    z0 = rotate_mas[mass_pol[i][0] - 1][2]
    z1 = rotate_mas[mass_pol[i][1] - 1][2]
    z2 = rotate_mas[mass_pol[i][2] - 1][2]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    v0 = mass_pol[i][0]-1
    v1 = mass_pol[i][1]-1
    v2 = mass_pol[i][2]-1
    vn_calc[v0] += n/np.linalg.norm(n)
    vn_calc[v1] += n/np.linalg.norm(n)
    vn_calc[v2] += n/np.linalg.norm(n)

for i in range(len(vn_calc)):
    vn_calc[i]/=np.linalg.norm(vn_calc[i])



#18 задание
img_mat = np.zeros((4000, 4000,3), dtype = np.uint8)
zb = np.full((4000,4000),np.inf,dtype=np.float32)
tex_mat = np.array(ImageOps.flip(Image.open("bunny-atlas.jpg")))

n = tex_mat.shape

for i in range(len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0] - 1][0]
    y0 = rotate_mas[mass_pol[i][0] - 1][1]
    x1 = rotate_mas[mass_pol[i][1] - 1][0]
    y1 = rotate_mas[mass_pol[i][1] - 1][1]
    x2 = rotate_mas[mass_pol[i][2] - 1][0]
    y2 = rotate_mas[mass_pol[i][2] - 1][1]
    z0 = rotate_mas[mass_pol[i][0] - 1][2]
    z1 = rotate_mas[mass_pol[i][1] - 1][2]
    z2 = rotate_mas[mass_pol[i][2] - 1][2]

    n0 = vn_calc[mass_pol[i][0] - 1]
    n1 = vn_calc[mass_pol[i][1] - 1]
    n2 = vn_calc[mass_pol[i][2] - 1]

    I0 = np.dot(n0, l)
    I1 = np.dot(n1, l)
    I2 = np.dot(n2, l)

    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        mas_uv=[mas_text[mass_n_text[i][0]-1],mas_text[mass_n_text[i][1]-1],mas_text[mass_n_text[i][2]-1]]
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb, I0, I1, I2, mas_uv, tex_mat,20000, 2000)
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img2.png')
print("end")

#Задание 17
'''for i in range (len(mass_pol)):
    x0 = rotate_mas[mass_pol[i][0]-1][0]
    y0 = rotate_mas[mass_pol[i][0]-1][1]
    x1 = rotate_mas[mass_pol[i][1]-1][0]
    y1 = rotate_mas[mass_pol[i][1]-1][1]
    x2 = rotate_mas[mass_pol[i][2]-1][0]
    y2 = rotate_mas[mass_pol[i][2]-1][1]
    z0 = rotate_mas[mass_pol[i][0]-1][2]
    z1 = rotate_mas[mass_pol[i][1]-1][2]
    z2 = rotate_mas[mass_pol[i][2]-1][2]

    n0 = vn_calc[mass_pol[i][0] - 1]
    n1 = vn_calc[mass_pol[i][1] - 1]
    n2 = vn_calc[mass_pol[i][2] - 1]

    I0 = np.dot(n0, l)
    I1 = np.dot(n1, l)
    I2 = np.dot(n2, l)

    
    if (cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2) < 0):
        otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb,I0,I1,I2,(215,161,227),20000,2000)
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img1.png')
print("end")'''



