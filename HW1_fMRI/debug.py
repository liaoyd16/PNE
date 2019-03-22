import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# img_arr格式: [right-left, front-back, top-down, time]
def load_img_array_from_nii(nii):
    img = nib.load(nii)

    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)

    return img_arr

# img = load_img_array_from_nii('../t1_y.nii')
# plt.imshow(img[:,:,128])
# plt.show()
img = load_img_array_from_nii('../bold_y.nii')
# plt.plot(np.arange(105), img[35,33,0])

plt.imshow(img[40,:,:,0])
plt.colorbar()
plt.show()
# print(img.shape)