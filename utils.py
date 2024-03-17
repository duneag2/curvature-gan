from scipy import misc
import os, cv2, torch
import copy
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from scipy import ndimage, spatial
from scipy.linalg import lstsq
from skimage import color, feature
import sympy as sy
from sympy import symbols

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

np.random.seed(1234)


# def load_test_data(image_path, size=256):
#     img = misc.imread(image_path, mode='RGB')
#     img = misc.imresize(img, [size, size])
#     img = np.expand_dims(img, axis=0)
#     img = preprocessing(img)

#     return img

# def preprocessing(x):
#     x = x/127.5 - 1 # -1 ~ 1
#     return x

# def save_images(images, size, image_path):
#     return imsave(inverse_transform(images), size, image_path)

# def inverse_transform(images):
#     return (images+1.) / 2

# def imsave(images, size, path):
#     return misc.imsave(path, merge(images, size))

# def merge(images, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[h*j:h*(j+1), w*i:w*(i+1), :] = image

#     return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
#     cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

# def imagenet_norm(x):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.299, 0.224, 0.225]
#     mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
#     std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
#     return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def RGB2GRAY(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def gaussian_high_pass_filter(x):
    data = np.array(x, dtype=float)
    lowpass = ndimage.gaussian_filter(data, 3)
    hp = data - lowpass

    return hp


def _canny_edge_point(x, dim=2):
    x_gray = color.rgb2gray(x)
#     print(x_gray.shape)
#     x_gray = cv2.resize(x, (256, 256))
#     rand_int = np.random.randint(int(x.shape[0]-50))
#     x_gray = x_gray[rand_int:rand_int+50, rand_int:rand_int+50]
    x_gray = x_gray[(0+85):(256-85),(0+85):(256-85)]
    canny_div = feature.canny(x_gray, sigma=2.5)
    canny_arr = np.transpose(np.ndarray.nonzero(canny_div))

    if dim == 2:
        result = canny_arr

    elif dim == 3:
        result = canny_div
        
    size = x_gray.shape[0]

    return result, size

def canny_ablation(x, dim=2):
    x_gray = color.rgb2gray(x)
    x_gray = x_gray[(0+85):(256-85),(0+85):(256-85)]
    canny_div = feature.canny(x_gray, sigma=2.5)
    
    if dim == 2:
        result = canny_div
    
    return torch.Tensor(result)

def img2curvature_2d(x):
    points, num = _canny_edge_point(x, 2)
    curvature = np.zeros((num, num))

    for pt in points:
        pt1 = points[spatial.KDTree(points).query(pt, k=3)[1]][1]
        pt2 = points[spatial.KDTree(points).query(pt, k=3)[1]][2]
        df = pd.DataFrame({'x': [pt[0], pt1[0], pt2[0]], 'y': [pt[1], pt1[1], pt2[1]]})
        
        model = np.poly1d(np.polyfit(df.x, df.y, 3))
        kappa2 = model.deriv(2)(pt[0]) / (1 + (model.deriv(1)(pt[1]))**2)**(3/2)
        curvature[pt[0], pt[1]] = kappa2
#     np.save('/home/koreagen/koreagen/curvature-2d/c.npy', curvature)
    curvature = torch.Tensor(curvature)
#     print(curvature.shape)
    
    
    return curvature


def img2curvature_3d(x):
    points, num = _canny_edge_point(x, 3)
    curvature = np.zeros((num, num))
    X = []
    Y = []
    Z = []
    for xx in range(num):
        for yy in range(num):
            X.append(xx)
            Y.append(yy)
            Z.append(points[xx, yy])
            
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    Z = np.expand_dims(Z, axis=1)
    
    order = 5
    e = [(x,y) for n in range(0,order+1) for y in range(0,n+1) for x in range(0,n+1) if x+y==n]
    eX = np.asarray([[x] for x,_ in e]).T
    eY = np.asarray([[y] for _,y in e]).T

    A = (X ** eX) * (Y ** eY)
    C, resid ,_ , _ = lstsq(A, Z)
#     print(resid)
    r2 = 1 - resid[0] / (Z.size * Z.var())

    assert len(C) == len(e)

    x, y = symbols('x, y')
    poly = 0

    for i in range(len(e)):
        poly = poly + (C[i]) * (x**e[i][0]) * (y**e[i][1])

#     print("##### Details of Approximated Surface #####")
#     print(f'Polynomial =\n{poly}')
#     print(f'R2 = {r2}')

    poly = sy.simplify(poly)


    dh_x = sy.diff(poly, x)[0]
    dh_y = sy.diff(poly, y)[0]
    dh_x_x = sy.diff(dh_x, x)
    dh_x_y = sy.diff(dh_x, y)
    dh_y_y = sy.diff(dh_y, y)

    K = (dh_x_x * dh_y_y - dh_x_y**2) / (1 + dh_x**2 + dh_y**2)**2
    H = (dh_x_x * (1 + dh_y**2) - 2 * dh_x * dh_y * dh_x_y + dh_y_y * (1 + dh_x**2)) / (2 * (1 + dh_x_x**2 + dh_y_y**2)**(3/2))

    for xxx in range(num):
        for yyy in range(num):
            curvature[xxx, yyy] = H.subs({x:xxx, y:yyy})
    curvature = torch.Tensor(curvature)
    
    return curvature