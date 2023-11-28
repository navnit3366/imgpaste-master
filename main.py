import numpy as np
from PIL import Image
from scipy.misc import imresize
from poisson import *


def read_img(path):
    return np.asarray(Image.open(path))


def resizeTo(target, base):
    t = np.zeros_like(target)
    t[:base.shape[0],:base.shape[1]] = base
    return t

def resizeToSize(img, pair):
    return imresize(img, pair)

def testNaiveComposite():
    print("Naive test...")
    bg = read_img('./data/waterpool.png')
    fg = read_img('./data/bear.png')
    mask = read_img('./data/mask.png')
    y = 10
    x = 20
    res = naiveComposite(bg, fg, mask, y, x)
    Image.fromarray(res).save('./results/naive.png')
    print("Naive test done...")


def testRamp():
    print("Ramp test...")
    bg = read_img('./data/ramp.png')
    fg = read_img('./data/fg.png')
    mask = read_img('./data/mask3.png')[:,:,:3]
    
    # need to resize the others
    fg_full = resizeTo(bg, fg)
    mask_full = resizeTo(bg, mask)
    weights = [1,50,100,200]
    results = []
    for w in weights:
        results.append(poisson(bg, fg_full, mask_full, w))
    
    it = 0
    for every in results:
        Image.fromarray(resizeToSize(every, (86, 160))).save('./results/ramp_' + str(weights[it]) + '.png')
        it += 1
    print("Ramp test done...")

def testPoisson():
    print("Poisson test...")
    bg = read_img('./data/waterpool.png')
    fg = read_img('./data/bear.png')
    mask = read_img('./data/mask.png')[:,:,:3]
    
    # need to resize the others
    fg_full = resizeTo(bg, fg)
    mask_full = resizeTo(bg, mask)
    weights = [0, 50, 500, 800, 1200, 1600, 2000, 3200] 
    results = []
    for w in weights:
        results.append(poisson(bg, fg_full, mask_full, w))

    it = 0
    for every in results:
        Image.fromarray(every).save('./results/poisson_' + str(weights[it]) + '.png')
        it += 1
    print("Poisson test done...")


testNaiveComposite()
testRamp()
testPoisson()
