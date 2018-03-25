import os
import torch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

#img = io.imread('Sample Image.bmp')
#print(img.astype)
#
#tensor = torch.from_numpy(img).type(torch.FloatTensor)
#gray = tensor[:, :, 0] * 0.2989 + tensor[:, :, 1] * 0.5870 + tensor[:, :, 2] * 0.1140
#gray = tensor[:, :, 2]
#print(tensor)


#plt.imshow(gray.numpy())
#plt.show()
#from torchvision.datasets import folder 

a = np.array([[1,2], [3, 4]])
b = torch.from_numpy(a)
b = b.type(torch.FloatTensor)
print(b)
print(torch.norm(b))
print(b/torch.norm(b))
#traindir = '/media/gukyeong/HardDisk/data/CURE-TSR/bmp/RealChallengeFree'
#flistdir = '/media/gukyeong/HardDisk/data/CURE-TSR/bmp/flist/RealChallengeFree.txt'
#
#img = []
#with open(flistdir) as f:
#    flist = f.readlines()
#
#for line in flist:
#    target = line[2:4]
#    path = os.path.join(traindir, line[5:-1])
#    item = (path, target)
#    img.append(item)
#    break
#
#print (img)

#loader = folder.default_loader

#os.mkdir('testfolder')
