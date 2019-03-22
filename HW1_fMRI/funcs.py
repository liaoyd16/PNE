
import __init__
from __init__ import ROOT_DIR

import nibabel as nib
import scipy.io as scio
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import LUT
import hippo_LUT

import json

from ShowProcess import ShowProcess

from numba import jit
import copy

## 1
# def _is_ctx_and_lobe_is(label):
#     if (0 <= label - 1000) and (label - 1000 <= 35):
#         return LUT.lut[label - 1000]
#     elif (0 <= label - 2000) and (label - 2000 <= 35):
#         return LUT.lut[label - 2000]
#     else:
#         return None

# def _is_grey_matter_and_in_region(label, region):
#     if region == 'transverse-temporal':
#         return label==1034 or label==2034
#     elif region == 'pars-opercularis':
#         return label==1019 or label==2019


# a)
def calc_surface_ratios(brain):
    surfaces = {'frontal':0, 'temporal':0, 'parietal':0, 'occipital':0}
    total = 0

    process_bar = ShowProcess(36, 'ratio calc complete')

    filt = copy.deepcopy(brain)
    filt[filt < 1000] = 0
    filt[(1035 < filt) & (filt < 2000)] = 0
    filt[2035 < filt] = 0
    filt[filt <= 1035] -= 1000
    filt[2000 <= filt] -= 2000

    for i in range(1,36):
        process_bar.show_process()
        area = np.sum(np.where(filt == i, 1, 0))
        if not LUT.lut[i] == None:
            surfaces[LUT.lut[i]] += area
        total += area

    return surfaces['frontal'] / total, \
           surfaces['temporal'] / total, \
           surfaces['parietal'] / total, \
           surfaces['occipital'] / total


# b)
def calc_asymmetry(brain, region):
    L = 0
    R = 0

    if region=='transverse-temporal':
        L = float(np.sum(brain[brain == 1034]))
        R = float(np.sum(brain[brain == 2034]))
    else:
        L = float(np.sum(brain[brain == 1019]))
        R = float(np.sum(brain[brain == 2019]))

    return (L-R)/(L+R) * 2

# c)
def _downsample_mat(voxels, down_sample_rate):
    down_sample = np.array(
        np.zeros(
            (256//down_sample_rate, 256//down_sample_rate, 256//down_sample_rate)
        ), 
        dtype=bool
    )
    for  i in range(down_sample_rate):
        try:
            down_sample = down_sample | (voxels[np.arange(i, 256, down_sample_rate),:,:][:,np.arange(i, 256, down_sample_rate),:][:,:,np.arange(i, 256, down_sample_rate)])
        except:
            down_sample = down_sample | (voxels[np.arange(i, 256, down_sample_rate)[:-1],:,:][:,np.arange(i, 256, down_sample_rate)[:-1],:][:,:,np.arange(i, 256, down_sample_rate)[:-1]])

    return down_sample

def hippocampus_3d_show(brain, png_name):

    voxels = np.array(np.zeros((256,256,256)), dtype=bool)

    for label in hippo_LUT.lut:
        voxels = voxels | (brain == label)

    voxels = _downsample_mat(voxels, 4)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels)

    fig.savefig(ROOT_DIR + png_name)
    plt.show()


## 2
# 1)
''' matlab implementation'''

# 2)
def load_img_array_from_nii(nii):
    img = nib.load(nii)

    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)

    return img_arr


design_matrix = np.zeros((105, 2))
on = True
for k in range(15):
    if on:
        design_matrix[7*k : 7*k + 7, 0] = 1
        on = False
    else:
        design_matrix[7*k : 7*k + 7, 0] = 0
        on = True
design_matrix[:,1] = 1

hrf = np.array([0.00000,0.35802,0.56986,0.20415,0.00240,-0.05375,-0.04565,-0.02329,-0.00862,-0.00251,-0.00061])

def conv(design_matrix, hrf):
    ans = copy.deepcopy(design_matrix)
    ans[:, 0] = np.convolve(design_matrix[:, 0], hrf)[:105]
    design_matrix[:, 1] = 1
    return ans


def find_most_relevant(nii, X):
    # nii: [right-left, front-back, top-down, time]
    RL = nii.shape[0]
    FB = nii.shape[1]
    TD = nii.shape[2]

    corr_list = np.zeros((10, 4))
    cnt = 0

    XTX_1 = np.linalg.inv( np.matmul(X.transpose(), X) )
    mat = np.matmul(XTX_1, X.transpose())

    process_bar = ShowProcess(RL * FB * TD, 'ratio calc complete')

    corr_max = 0

    for rl in range(RL):
        for fb in range(FB):
            for td in range(TD):
                process_bar.show_process()

                activation = nii[rl, fb, td]
                corr = np.matmul(mat, activation)[0]

                if cnt < 10:
                    corr_list[cnt] = np.array([corr, rl, fb, td])
                    cnt += 1
                else:
                    i = np.argmin(corr_list[:,0].reshape(10))
                    if corr > corr_list[i,0]:
                        corr_list[i] = [corr, rl, fb, td]

    return np.array(corr_list)

def convert_coord(coord_80_80_47, vs = 2):
    origin = [45, 63, 36]
    ans = np.array(
          [int((origin[0] - coord_80_80_47[0]) * vs),
           int((coord_80_80_47[1] - origin[1]) * vs),
           int((coord_80_80_47[2] - origin[2]) * vs)])
    return ans

if __name__ == '__main__':
    
    dataFile = __init__.ROOT_DIR + 'twobrains.mat'
    brain1 = scio.loadmat(dataFile)['brain1']
    brain2 = scio.loadmat(dataFile)['brain2']
    
    ## 1.
    # a)
    ###
    print("1.a)")
    print("for brain1:")
    frontal, temporal, parietal, occipital = calc_surface_ratios(brain1)
    print("ratios: frontal={}, temporal={}, parietal={}, occipital={}".format(frontal, temporal, parietal, occipital))

    print("for brain2:")
    frontal, temporal, parietal, occipital = calc_surface_ratios(brain2)
    print("ratios: frontal={}, temporal={}, parietal={}, occipital={}".format(frontal, temporal, parietal, occipital))

    # b)
    print("1.b)")
    print("for brain1:")
    sym_tt = calc_asymmetry(brain1, 'transverse-temporal')
    sym_po = calc_asymmetry(brain1, 'pars-opercularis')
    print('transverse-temporal index: {}, pars-opercularis index: {}'.format(sym_tt, sym_po))
    
    print("for brain2:")
    sym_tt = calc_asymmetry(brain2, 'transverse-temporal')
    sym_po = calc_asymmetry(brain2, 'pars-opercularis')
    print('transverse-temporal index: {}, pars-opercularis index: {}'.format(sym_tt, sym_po))

    # c)
    print("1.c)")
    print("brain1")
    hippocampus_3d_show(brain1, "brain1.png")
    print("brain2")
    hippocampus_3d_show(brain2, "brain2.png")
    

    ## 2.
    # 1)
    ###

    # 2)
    nii = load_img_array_from_nii(ROOT_DIR + 'bold_y.nii')
    output_nii = load_img_array_from_nii(ROOT_DIR + 't1_y.nii')

    def plot_and_save(corr_list, output_nii, nii_dump_name):
        output_nii_glm = copy.deepcopy(output_nii)
        # for num_coord in range(10):
        #     coord_80_80_47 = corr_list[num_coord][1:4]
        #     print(corr_list[num_coord])
        #     coord_256_256_180 = convert_coord(coord_80_80_47)
        #     output_nii_glm[coord_256_256_180[0],coord_256_256_180[1],coord_256_256_180[2]] = 5000

        # for num_coord in range(10):
        #     coord_80_80_47 = corr_list[num_coord][1:4]
        #     coord_256_256_180 = convert_coord(coord_80_80_47)
        #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        #     ax1.imshow(output_nii_glm[coord_256_256_180[0],:,:])
        #     ax2.imshow(output_nii_glm[:,coord_256_256_180[1],:])
        #     ax3.imshow(output_nii_glm[:,:,coord_256_256_180[2]])

        ## test validity of coords: ok
        for num_coord in range(10):
            coord_80_80_47 = np.array(corr_list[num_coord][1:4], dtype=int)
            output_nii_glm[coord_80_80_47[0],coord_80_80_47[1],coord_80_80_47[2]] = 4000

        for num_coord in range(10):
            coord_80_80_47 = np.array(corr_list[num_coord][1:4], dtype=int)
            fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
            ax1.imshow(output_nii_glm[coord_80_80_47[0],:,:])
            ax2.imshow(output_nii_glm[:,coord_80_80_47[1],:])
            ax3.imshow(output_nii_glm[:,:,coord_80_80_47[2]])

            plt.show()
        
        # print("dumping: ", nii_dump_name)
        # json.dump(output_nii_glm.tolist(), open(nii_dump_name, 'w'))

    # print("GLM:")
    # corr_list = find_most_relevant(nii, design_matrix)
    # plot_and_save(corr_list, output_nii, ROOT_DIR + "output_nii_GLM.json")

    # print("GLM with HRF")
    # conv_design_matrix = conv(design_matrix, hrf)
    # corr_list = find_most_relevant(nii, conv_design_matrix)
    # plot_and_save(corr_list, output_nii, ROOT_DIR + "output_nii_HRF.json")

    print("GLM:")
    corr_list = find_most_relevant(nii, design_matrix)
    plot_and_save(corr_list, np.mean(nii, axis=3), ROOT_DIR + "output_nii_GLM.json")

    print("GLM with HRF")
    conv_design_matrix = conv(design_matrix, hrf)
    corr_list = find_most_relevant(nii, conv_design_matrix)
    plot_and_save(corr_list, np.mean(nii, axis=3), ROOT_DIR + "output_nii_HRF.json")
