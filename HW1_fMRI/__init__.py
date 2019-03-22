
import scipy.io as scio
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

ROOT_DIR = '/Users/liaoyuanda/Desktop/PNE/hw1/'

if __name__=="__main__":
    dataFile = '/Users/liaoyuanda/Desktop/PNE/hw1/twobrains.mat'
    data = scio.loadmat(dataFile)
    brain = data['brain1']
    voxels = (brain == 2024)

    down_sample = downsample_mat(voxels, down_sample_rate=3)
    
    print("down_sample finis")

    # plt.imshow(brain[:,120,:])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(down_sample)

    plt.show()

    '''

    '''
    #注：第二个维度为纵切法向
    '''
    def get_label_dict(brain):
        label_dict = dict()
        for hem in range(256):
            for v in range(256):
                for h in range(256):
                    val = brain[hem, v, h]
                    if val not in label_dict:
                        label_dict[val] = 1
                    else:
                        label_dict[val] += 1

        return label_dict

    b1 = data['brain1']
    b2 = data['brain2']

    b1_labels = get_label_dict(b1)
    b2_labels = get_label_dict(b2)
    print(len(b1_labels))
    print(len(b2_labels))
    
    inter = 0
    for label in b1_labels.keys():
        if label in b2_labels.keys():
            inter += 1
    print(inter)

    print(b1_labels.keys())

    # 共183个label
    
    '''