from PIL import Image, ImageOps
import numpy as np
import math

#задание 7
def bar_kor(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    nx=(y1-y2)*(z1-z0)-(z1-z2)*(y1-y0)
    ny =(z1-z2)*(x1-x0)-(x1-x2)*(z1-z0)
    nz =(x1-x2)*(y1-y0)-(y1-y2)*(x1-x0)
    return [nx, ny,nz]

def cosinus(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0,0,1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    return np.dot(l, n)/ (np.linalg.norm(l)*np.linalg.norm(n))
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
    if (xmax > img_mat.shape[1]): xmax = img_mat.shape[1]
    if (ymax > img_mat.shape[0]): ymax = img_mat.shape[0]
    for x in range(xmin,xmax):
        for y in range (ymin, ymax):
            l0, l1, l2 = bar_kor(x, y, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2)
            if (l0 >= 0 and l1 >=0 and l2 >= 0):
                z_coord = l0*z0+l1*z1+l2*z2
                if (z_coord<zb[y][x]):
                    intensivity = -(I0*l0+I1*l1+I2*l2)
                    if (intensivity<0):
                        intensivity = 0
                    img_mat[y,x] = intensivity*text_mat[int(1024*( l0*mas_text[0][1]+l1*mas_text[1][1]+l2*mas_text[2][1])),int(1024*(l0*mas_text[0][0]+l1*mas_text[1][0]+l2*mas_text[2][0]))]
                    #img_mat[y, x] = (intensivity*color[0],intensivity*color[1],intensivity*color[2])
                    zb[y][x] = z_coord

def parser(name_file):
    mas_ver = []
    mas_pol = []
    mas_n_text=[]
    mas_text=[]
    f = open(name_file)
    for s in f:
        if len(s) == 1 or len(s)==0:
            continue
        splitted = s.split()
        if (splitted[0] == 'v'):
            mas_ver.append([float(x) for x in splitted[1:]])
        if (splitted[0] == 'f'):
            n = len(splitted)
            mas_p = []
            mas_t=[]
            for i in range(1,n):
                element = splitted[i].split('/')
                mas_p.append(int(element[0]))
                mas_t.append(int(element[1]))
            mas_pol.append(mas_p)
            mas_n_text.append(mas_t)
        if (splitted[0] == 'vt'):
            mas_text.append([float(splitted[1]), float(splitted[2])])
    f.close()
    return mas_ver,mas_pol,mas_text,mas_n_text

def normals_vn(mass_pol, mas_ver, rotate_mas):
    vn_calc = np.full((len(mas_ver), 3), 0, dtype=np.float32)

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
        v0 = mass_pol[i][0] - 1
        v1 = mass_pol[i][1] - 1
        v2 = mass_pol[i][2] - 1
        vn_calc[v0] += n / np.linalg.norm(n)
        vn_calc[v1] += n / np.linalg.norm(n)
        vn_calc[v2] += n / np.linalg.norm(n)

    for i in range(len(vn_calc)):
        vn_calc[i] /= np.linalg.norm(vn_calc[i])
    return vn_calc



def rotate(mas_pol,mas_ver,alfa,beta,gamma,tx,ty,tz):
    rx = np.array([[1,0,0],[0,math.cos(alfa),math.sin(alfa)],[0,-math.sin(alfa),math.cos(alfa)]])
    ry = np.array([[math.cos(beta), 0, math.sin(beta)], [0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    rz = np.array([[math.cos(gamma), math.sin(gamma),0],[-math.sin(gamma), math.cos(gamma), 0],[0,0,1]])
    r = np.dot(np.dot(rx,ry), rz)
    mass_rot = []
    for i in range(len(mas_pol)):
        mass_rot.append(np.dot(r,mas_ver[i])+[tx,ty,tz])
    return mass_rot

def mult_quaternion(quat1, quat2):
    a1, b1, c1, d1 = quat1
    a2, b2, c2, d2 = quat2
    a = a1*a2-b1*b2-c1*c2-d1*d2
    b = a1*b2+b1*a2+c1*d2-c2*d1
    c = a1*c2-b1*d2+c1*a2-d1*b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + a2 * d1
    return [a,b,c,d]

def rotate_quaternion(u,theta, coord):
    cost = np.cos(theta/2)
    sint = np.sin(theta/2)
    a,b,c = np.dot(u,sint)
    quat = np.array([cost,a,b,c])
    rotate = mult_quaternion(mult_quaternion(quat,coord),np.array([cost,-a,-b,-c]))
    return rotate[1:]


def rotate_Euler(alfa, beta, gamma,coord):
    q1 = [math.cos(alfa/2),math.sin(alfa/2),0,0]
    q2 = [math.cos(beta/2),0,math.sin(beta/2),0]
    q3 = [math.cos(gamma / 2),0,0, math.sin(gamma / 2)]
    a,b,c,d = mult_quaternion(mult_quaternion(q1,q2),q3)
    r = np.array([[a**2+b**2-c**2-d**2,2*b*c-2*a*d,2*b*d+2*a*c],[2*b*c+2*a*d,a**2-b**2+c**2-d**2,2*c*d-2*a*b],[2*b*d-2*a*c,2*c*d+2*a*b,a**2-b**2-c**2+d**2]])
    rotate = np.dot(r,coord)
    return rotate


def render(file,texture,img_mat,zb, rot,tx,ty,tz,a,b,theta,angle,u):
    mas_ver, mass_pol, mas_text, mass_n_text = parser(file)
    l = [0, 0, 1]
    tex_mat = np.array(ImageOps.flip(Image.open(texture)))
    rotate_mas = []

    if (rot == 'q'):
        for coord in mas_ver:
            coord[0], coord[1], coord[2] = rotate_quaternion(u, theta, np.array([0, coord[0], coord[1], coord[2]]))
            rotate_mas.append([coord[0]+tx, coord[1]+ty, coord[2]+tz])
    elif (rot == 'e'):
        for coord in mas_ver:
            coord[0], coord[1], coord[2] = rotate_Euler(angle[0],angle[1],angle[2], np.array([coord[0], coord[1], coord[2]]))
            rotate_mas.append([coord[0]+tx, coord[1]+ty, coord[2]+tz])

    vn_calc = normals_vn(mass_pol, mas_ver, rotate_mas)
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
            mas_uv = [mas_text[mass_n_text[i][0] - 1], mas_text[mass_n_text[i][1] - 1], mas_text[mass_n_text[i][2] - 1]]
            otris_treyg2(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, zb, I0, I1, I2, mas_uv, tex_mat, a, b)




img_mat = np.zeros((2000, 3000, 3), dtype=np.uint8)
zb = np.full((2000, 3000), np.inf, dtype=np.float32)
render("model_1.obj","bunny-atlas.jpg",img_mat,zb,"q", 0.05,-0.1,1, 15000,2000,math.pi/3,[],[0,1,0])
render("model_1.obj","bunny-atlas.jpg",img_mat,zb,"e", 0.05,-0.05,1.1, 15000,3000,0,[0,math.pi/2,0],[0,0,1])
render("model_1.obj","bunny-atlas.jpg",img_mat,zb,"q", 0.01,-0.01,1, 20000,2000,-math.pi/3,[],[1,0,0])
render("12221_Cat_v1_l3.obj","Cat_diffuse.jpg",img_mat,zb,"q", -20,30,3, 20000,2000,-math.pi/3,[],[0,1,0])
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img7.png')
print("end")
