""" script to demonstrate how to use tool.py """

##
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import utils




##
""" ========== 1 make data easy to access (run once after data download) ========== """

# ---- 1.1. make sure you have downloaded the zip files to ./data. the data urls are listed below
data_urls = [
    'https://www.kaggle.com/c/8089/download/stage1_train.zip',
    'https://www.kaggle.com/c/8089/download/stage1_train_labels.csv.zip',
    'https://www.kaggle.com/c/8089/download/stage1_test.zip',
    'https://www.kaggle.com/c/8089/download/stage1_sample_submission.csv.zip'
    ]

# ---- 1.2. unzip data
utils.unzip_data()

# ---- 1.3. create a mapping of image id and file path of image and masks
# returns a dictionary of {image_id: {'images': list_of_image_file_path, 'masks': list_of_mask_file_path}}
dict_id_path_train = utils.gen_id_path_dict('./data/stage1_train')
dict_id_path_test  = utils.gen_id_path_dict('./data/stage1_test')

# ---- 1.4. create a data structure:
"""
store data in a single file:
dict_data = {img_id: {'image': np.array of shape (M, N, 3), 'mask': np.array of shape (N, N) }}
where image is a (M,N,3) RGB array, with dtype=uint8,
and mask is a (M,N) array, where value 0 represents background and value 1 to num_nuclei represents every label
"""
utils.create_data_file(dict_id_path_train, 'data_train.pickle')
utils.create_data_file(dict_id_path_test,  'data_test.pickle')


"""" ========== 2. load data and visualize data ========== """

# ----- 2.1 load data
data_tr = utils.load_data('data_train.pickle')
data_tt = utils.load_data('data_test.pickle')

# ----- 2.2 plot data from the original file directory
# plot a random image with masks
plt.figure()
utils.plot_img_and_mask_from_file(dict_id_path_train)

# plot a particular image
plt.figure()
id_to_plot = list(dict_id_path_train.keys())[0]
utils.plot_img_and_mask_from_file(dict_id_path_train, id_to_plot)

# plot a random test image
plt.figure()
utils.plot_img_and_mask_from_file(dict_id_path_test)

# ----- 2.3 plot data from our data structure
# plot a random image with masks
plt.figure()
utils.plot_img_and_mask_from_dict(data_tr)
# plot a particular image
id_to_plot = list(data_tr.keys())[0]
plt.figure()
utils.plot_img_and_mask_from_dict(data_tr, id_to_plot)

# plot a random test image
plt.figure()
utils.plot_img_and_mask_from_dict(data_tt)



""" ========== 3. performance evalutation ========== """

""" ----- 3.1 mask segmentation """
id_eg = 'd0f2a00d3155c243048bc48944aef93fb08e2258d1fa5f9ccadd9140082bc22f'
image = data_tr[id_eg]['image']
mask_true = data_tr[id_eg]['mask']
# merge masks
mask_bin = np.where(mask_true > 0, 1, 0)
# re-segment masks
mask_seg = utils.segment_mask(mask_bin)

h_fig, h_axes = plt.subplots(2, 2, sharex='all', sharey='all')
plt.axes(h_axes[0, 0])
plt.imshow(image)
plt.title('image')
plt.axis('off')
plt.axes(h_axes[0, 1])
plt.imshow(mask_true)
plt.title('true labels')
plt.axis('off')
plt.axes(h_axes[1, 0])
plt.imshow(mask_bin)
plt.title('merged')
plt.axis('off')
plt.axes(h_axes[1, 1])
plt.imshow(mask_seg)
plt.title('re-segmented')
plt.axis('off')

""" ----- 3.2 compute IOU and score """
IOU = utils.cal_prediction_IOU(mask_true, mask_seg)
score = utils.cal_score_from_IOU(IOU)
print(score)


""" ========== 4. image split and stitch ========== """

# ----- 4.1 image split
id_eg = np.random.choice(list(data_tr.keys()))
image = data_tr[id_eg]['image']
mask_true = data_tr[id_eg]['mask']

# full image
plt.figure()
utils.plot_img_and_mask_from_dict(data_tr, id_eg)

# split image segments
img_seg = utils.img_split(image, size_seg=128, overlap=0.2)
img_seg_start = np.array(list(img_seg.keys()))
rs_start = np.unique(img_seg_start[:, 0])
cs_start = np.unique(img_seg_start[:, 1])
h_fig, h_axes = plt.subplots(len(rs_start), len(cs_start))
for i_r, r in enumerate(rs_start):
    for i_c, c in enumerate(cs_start):
        plt.axes(h_axes[i_r, i_c])
        plt.imshow(img_seg[(r, c)])
        plt.axis('off')
        plt.title((r, c), fontsize='x-small')

mask_seg = utils.img_split(mask_true, size_seg=128, overlap=0.5)
mask_seg = {key: utils.mask_2Dto3D(mask_seg[key]) for key in mask_seg}


# ----- 4.2 image stitch
importlib.reload(utils)
img_full = utils.img_stitch(img_seg)
plt.figure()
plt.imshow(img_full)

# ----- 4.3 mask stitch
importlib.reload(utils)

mask_full = utils.img_stitch(mask_seg, mode='mask')
plt.imshow(utils.mask_3Dto2D(mask_full[0]), cmap='Paired')




##
""" image rescale according to nuc size """

ratio_size_nuc_ideal = 16
id_img = random.choice(list(data_tr.keys()))


img_cur = data_tr[id_img]['image']
mask_cur = data_tr[id_img]['mask']

image_rescale, mask_rescale, rescale_factor = utils.resize_image_mask(img_cur, mask_cur, r_size_img_nuc_ideal=12)

plt.subplot(2, 2, 1)
plt.imshow(img_cur)
plt.subplot(2, 2, 2)
utils.plot_mask2D(mask_cur)
plt.subplot(2, 2, 3)
plt.imshow(image_rescale)
plt.title(rescale_factor)
plt.subplot(2, 2, 4)
utils.plot_mask2D(mask_rescale)

##
""" image load random crop """
id_img = random.choice(list(data_tr.keys()))

img_cur = data_tr[id_img]['image']
mask_cur = data_tr[id_img]['mask']

image_crop, mask_crop_3D = utils.load_image_mask_with_random_crop(img_cur, mask_cur, cropsize=256)


plt.subplot(2, 2, 1)
plt.imshow(img_cur)
plt.subplot(2, 2, 2)
utils.plot_mask2D(mask_cur)
plt.subplot(2, 2, 3)
plt.imshow(image_crop)
plt.title('crop')
plt.subplot(2, 2, 4)
utils.plot_mask2D(utils.mask_3Dto2D(mask_crop_3D))

##



##

