import numpy as np
from scipy.signal import convolve2d


def naiveComposite(bg, fg, mask, y, x):
    boolmask = mask.astype(np.bool)
    out = bg.copy()
    (h,w,_) = fg.shape
    out[y:y+h,x:x+w][boolmask] = fg[boolmask]
    return out 

def laplacian_op(img):
    op = np.array([[0., -1., 0. ],[-1. , 4., -1.],[0., -1., 0.]], np.float64)
    out = np.zeros_like(img)
    for ch in range(img.shape[2]):
        out[...,ch] = convolve2d(img[...,ch], op, mode='same')
    return out

def poisson(bg, fg, mask, niter):
    if not (bg.shape == mask.shape and mask.shape == fg.shape):
        print(bg.shape)
        print(fg.shape)
        print(mask.shape)
        raise Exception("All shapes should be of the same size")

    x = bg.astype(np.float64)
    boolmask = mask.astype(np.bool)
    constant_img = laplacian_op(fg.astype(np.float64))*boolmask

    x[boolmask] = 0.0
    for _ in range(niter):
        Ax = laplacian_op(x)
        r = (constant_img - Ax) * boolmask
        Arf = laplacian_op(r).flatten()
        rf = r.flatten()
        alpha = np.dot(rf, rf) / np.dot(rf, Arf)
        x += ((r*boolmask) * alpha)

    return x.clip(0,255).astype(bg.dtype)


