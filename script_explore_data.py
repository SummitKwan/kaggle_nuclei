""" script to understand data stucrure """

import os
import warnings
import urllib
import zipfile
import pickle
import skimage

from tqdm import tqdm
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt



""" download data """
data_urls = [
    'https://www.kaggle.com/c/8089/download/stage1_train.zip',
    'https://www.kaggle.com/c/8089/download/stage1_train_labels.csv.zip',
    'https://www.kaggle.com/c/8089/download/stage1_test.zip',
    'https://www.kaggle.com/c/8089/download/stage1_sample_submission.csv.zip'
    ]

path_data = './data'
if not os.path.isdir(path_data):
    os.makedirs(path_data)


for data_url in data_urls:
    data_file_zip = os.path.basename(data_url)
    if data_file_zip not in os.listdir(path_data):
        warnings.warn('please download data file {} manually to {}'.format(data_url, path_data))
        break
        # the line below does not work, since kaggle needs username and password
        # urllib.request.urlretrieve(data_url, os.path.join(path_data, data_file_zip))
    data_file_unzip = data_file_zip[:-4]
    if '.' in data_file_unzip:
        path_extract = path_data
    else:
        path_extract = os.path.join(path_data, data_file_unzip)
    if data_file_zip in os.listdir(path_data) and data_file_unzip not in os.listdir(path_data):
        zipfile.ZipFile(os.path.join(path_data, data_file_zip), 'r')\
            .extractall(path_extract)

""" understand the data folder structure """
# get image names
path_data = './data/stage1_train'
# path_data = './data/stage1_test'
# path_data = './data/Amit_Sethi_processed'

ids_img = os.listdir(path_data)
ids_img = [id_img for id_img in ids_img
       if os.path.isdir(os.path.join(path_data, id_img))
       and 'images' in os.listdir(os.path.join(path_data, id_img))]
path_ids_img = [os.path.join(path_data, id_img) for id_img in ids_img]
print('number of images in {} is {}'.format(path_data, len(ids_img)))

num_img_per_id = [len([file for file in os.listdir(os.path.join(path_id_img, 'images')) if file[-3:] == 'png'])
                  for path_id_img in path_ids_img]
# it seems that every image id only contains a single image of the same name


""" create a dictionary of {image_id: {'images': list_of_image_file_path, 'masks': list_of_mask_file_path}} """

def gen_id_path_dict(path_data):
    """ generate a dictionary that stores the file location of every image """
    ids_img = os.listdir(path_data)
    ids_img = [id_img for id_img in ids_img
               if os.path.isdir(os.path.join(path_data, id_img))
               and 'images' in os.listdir(os.path.join(path_data, id_img))]

    dict_ids = {}
    for id_img in ids_img:

        list_img_names = os.listdir(os.path.join(path_data, id_img, 'images'))
        list_img_path = [os.path.join(path_data, id_img, 'images', img_name)
                         for img_name in list_img_names if img_name[-3:] in ('png', 'PNG', 'tif') ]
        if list_img_path:
            dict_ids[id_img] = {'images': list_img_path}
        else:
            continue
        list_mask_names = os.listdir(os.path.join(path_data, id_img, 'masks'))
        list_mask_path = [os.path.join(path_data, id_img, 'masks', img_name)
                          for img_name in list_mask_names if img_name[-3:] == 'png']
        if list_mask_path:
            dict_ids[id_img]['masks'] = list_mask_path

    return dict_ids


# a dictionary of the file path for every id
dict_ids = gen_id_path_dict(path_data)


"""
store data in a single file:
dict_data = {img_id: {'image': np.array of shape (M, N, 4), 'mask': np.array of shape (N, N) }}
where image is a (M,N,4) RGBA array, with dtype=uint8, 
and mask is a (M,N) array, where value 0 represents background and value 1 to num_nuclei represents every label
"""
dict_data = {}
for id_img in tqdm(dict_ids):
    img = ndimage.imread(dict_ids[id_img]['images'][0])
    list_masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_img]['masks']]
    num_masks = len(list_masks)
    array_masks = np.zeros(img.shape[:2], dtype='uint16')
    for i_mask, mask in enumerate(list_masks):
        array_masks[mask > 0] = i_mask+1
    dict_data[id_img] = {'image': img, 'mask': array_masks}

# save to disk
path_dict_data = os.path.join('./data', 'data_train.pickle')
with open(path_dict_data, 'wb') as f:
    pickle.dump(dict_data, f)

# load from disk
with open(path_dict_data, 'rb') as f:
    dict_data = pickle.load(f)



""" use external dataset: data fix """

dict_ids = gen_id_path_dict('./data/stage1_train_fix')

dict_data = {}
for id_img in tqdm(dict_ids):
    img = ndimage.imread(dict_ids[id_img]['images'][0])[:, :, :3]
    list_masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_img]['masks']]
    num_masks = len(list_masks)
    array_masks = np.zeros(img.shape[:2], dtype='uint16')
    for i_mask, mask in enumerate(list_masks):
        array_masks[mask > 0] = i_mask+1
    dict_data[id_img] = {'image': img, 'mask': array_masks}

path_dict_data = os.path.join('./data', 'data_train_fix.pickle')
with open(path_dict_data, 'wb') as f:
    pickle.dump(dict_data, f)

# load from disk
with open(path_dict_data, 'rb') as f:
    dict_data = pickle.load(f)


""" use external dataset: Amit """
dict_ids = gen_id_path_dict('./data/Amit_Sethi_processed')


dict_data = {}
for id_img in tqdm(dict_ids):
    img = skimage.io.imread(dict_ids[id_img]['images'][0])[:, :, :3]
    list_masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_img]['masks']]
    num_masks = len(list_masks)
    array_masks = np.zeros(img.shape[:2], dtype='uint16')
    for i_mask, mask in enumerate(list_masks):
        array_masks[mask > 0] = i_mask+1
    dict_data[id_img] = {'image': img, 'mask': array_masks}

path_dict_data = os.path.join('./data', 'data_train_Amit.pickle')
with open(path_dict_data, 'wb') as f:
    pickle.dump(dict_data, f)

# load from disk
with open(path_dict_data, 'rb') as f:
    dict_data = pickle.load(f)


""" use external dataset """
dict_ids = {}
path_data = './data/TNBC_NucleiSegmentation'
folders = os.listdir(path_data)
folders_slide = [folder for folder in folders if folder[:5]=='Slide']
folders_gt = [folder for folder in folders if folder[:2]=='GT']
for folder in folders_slide:
    folder_mask = 'GT'+folder[5:]
    img_names = [item for item in os.listdir(os.path.join(path_data, folder)) if item[-3:]=='png']
    for img_name in img_names:
        id_img = img_name[:-4]
        dict_ids[id_img] = {'images': os.path.join(path_data, folder, img_name),
                            'masks': os.path.join(path_data, folder_mask, img_name)
                            }


dict_data = {}
for id_img in tqdm(dict_ids):
    img = ndimage.imread(dict_ids[id_img]['images'])[:, :, :3]
    mask2D = ndimage.label(ndimage.imread(dict_ids[id_img]['masks'], flatten=True))[0].astype('uint16')
    dict_data[id_img] = {'image': img, 'mask': mask2D}

path_dict_data = os.path.join('./data', 'data_train_TNBC.pickle')
with open(path_dict_data, 'wb') as f:
    pickle.dump(dict_data, f)

# load from disk
with open(path_dict_data, 'rb') as f:
    dict_data = pickle.load(f)


""" stage2 final test data """
dict_id_path_test_stage2  = utils.gen_id_path_dict('./data/stage2_test')
utils.create_data_file(dict_id_path_test_stage2,  'data_test_stage2.pickle')

plt.figure()
id_img = random.choice(list(data_tt2.keys()))
utils.plot_img_and_mask_from_dict(data_tt2, id_img)
print(id_img)
os.mkdir('./data/stage2_test_images_no_subfolder')
dst = './data/stage2_test_images_no_subfolder'
import shutil
for img_id in dict_id_path_test_stage2:
    shutil.copy(dict_id_path_test_stage2[img_id]['images'][0], dst)


""" visualize example data """
id_to_plot = np.random.choice(list(dict_ids.keys()))


def add_sub_axes(h_axes=None, loc='top', size=0.25, gap=0.02, sub_rect=None):
    """
    tool funciton to add an axes around the existing axis
    :param h_axes: the current axes handle, default to None, use the gca
    :param loc:    location of the newly added sub-axes: one of ['top', 'bottom', 'left', 'right', 'custom'], default to 'top'
                    - if one of ['top', 'bottom', 'left', 'right'], the size of sub axes is determined by size and gap parameter;
                    - if set to 'custom', the location and size if specifited by sub_rect parameter
    :param size:   size of the sub-axes, with respect to the origial axes, default to 0.25
    :param gap:    gap between the original axes and and the newly added sub-axes
    :param sub_rect: the rect of custom sub-axes, rect = [x_left, y_bottom, ]
    :return:       handle of sub axes
    """
    if h_axes is None:
        h_axes = plt.gca()
    # get axes position
    axes_rect = h_axes.get_position()
    x0, y0, width, height = axes_rect.x0, axes_rect.y0, axes_rect.width, axes_rect.height

    # set modefied axes and new sub-axes position
    if sub_rect is not None:
        loc = 'custom'
    if loc == 'top':
        x0_new, y0_new, width_new, height_new = x0, y0, width, height * (1 - size - gap)
        x0_sub, y0_sub, width_sub, height_sub = x0, y0+height * (1 - size), width, height * size
        sharex, sharey = h_axes, None
    elif loc == 'bottom':
        x0_new, y0_new, width_new, height_new = x0, y0 + height * (size + gap), width, height * (1 - size - gap)
        x0_sub, y0_sub, width_sub, height_sub = x0, y0, width, height * size
        sharex, sharey = h_axes, None
    elif loc == 'left':
        x0_new, y0_new, width_new, height_new = x0 + width * (size + gap), y0, width * (1 - size - gap), height
        x0_sub, y0_sub, width_sub, height_sub = x0, y0, width * size, height
        sharex, sharey = None, h_axes
    elif loc == 'right':
        x0_new, y0_new, width_new, height_new = x0, y0, width * (1 - size - gap), height
        x0_sub, y0_sub, width_sub, height_sub = x0 + width * (1 - size), y0, width * size, height
        sharex, sharey = None, h_axes
    elif loc == 'custom':
        x0_rel, y0_rel, width_rel, height_rel = sub_rect
        x0_new, y0_new, width_new, height_new = x0, y0, width, height
        x0_sub, y0_sub, width_sub, height_sub = x0 + x0_rel * width, y0 + y0_rel * height, width * width_rel, height * height_rel
        sharex, sharey = None, None
    else:
        warnings.warn('loc has to be one of "top", "bottom", "left", "right", or "custom"')
        return None

    # make the curretn axes smaller
    h_axes.set_position([x0_new, y0_new, width_new, height_new])
    # add a new axes
    h_subaxes = h_axes.figure.add_axes([x0_sub, y0_sub, width_sub, height_sub])

    return h_subaxes


""" plot by reading image form file """
def plot_img_and_mask_from_file(id_to_plot, dict_ids=dict_ids):
    if id_to_plot not in dict_ids:
        warnings.warn('given id does not exist in the data, id={}'.format(id_to_plot))
        return None
    img = ndimage.imread(dict_ids[id_to_plot]['images'][0])
    masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_to_plot]['masks']]
    h_ax = plt.gca()
    h_ax_sub = add_sub_axes(h_axes=h_ax, loc='right', size=0.5, gap=0)
    plt.axes(h_ax)
    plt.imshow(img)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(img.shape, fontsize='x-small')
    plt.axes(h_ax_sub)
    plt.imshow(img)
    list_mask_colors = np.random.rand(len(masks), 3)
    list_mask_colors = np.append(list_mask_colors, [[0.9]]*len(masks), axis=1)
    for i in range(len(masks)):
        mask = masks[i]
        mask_to_plot = (mask[:, :, None]*list_mask_colors[i][None, None, :]).astype('uint8')
        plt.imshow(mask_to_plot)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(len(masks), fontsize='x-small')

plot_img_and_mask_from_file(id_to_plot)

# plot random training samples
h_fig, h_axes = plt.subplots(4, 2, squeeze=False, figsize=[8, 8])
h_axes = np.ravel(h_axes)
for h_ax in h_axes:
    plt.axes(h_ax)
    id_to_plot = np.random.choice(list(dict_ids.keys()))
    plot_img_and_mask_from_file(id_to_plot)


""" plot using the saved dict_data """


def plot_img_and_mask_from_dict(id_to_plot, dict_data=dict_data):

    if id_to_plot not in dict_ids:
        warnings.warn('given id does not exist in the data, id={}'.format(id_to_plot))
        return None
    img = dict_data[id_to_plot]['image']
    masks = dict_data[id_to_plot]['mask']
    h_ax = plt.gca()
    h_ax_sub = add_sub_axes(h_axes=h_ax, loc='right', size=0.5, gap=0)
    plt.axes(h_ax)
    plt.imshow(img)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(img.shape, fontsize='x-small')
    plt.axes(h_ax_sub)
    plt.imshow(img)
    list_mask_colors = np.random.rand(len(masks), 3)
    list_mask_colors = np.append(list_mask_colors, [[0.9]]*len(masks), axis=1)
    for i in range(masks.max()):
        mask = (masks == i+1)*255
        mask_to_plot = (mask[:, :, None]*list_mask_colors[i][None, None, :]).astype('uint8')
        plt.imshow(mask_to_plot)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(len(masks), fontsize='x-small')

plot_img_and_mask_from_dict(id_to_plot)


# plot random training samples
h_fig, h_axes = plt.subplots(4, 2, squeeze=False, figsize=[8, 8])
h_axes = np.ravel(h_axes)
for h_ax in h_axes:
    plt.axes(h_ax)
    id_to_plot = np.random.choice(list(dict_ids.keys()))
    plot_img_and_mask_from_dict(id_to_plot)


"""  visualizing one image  """
id_to_show = 'd0f2a00d3155c243048bc48944aef93fb08e2258d1fa5f9ccadd9140082bc22f'
img = ndimage.imread(dict_ids[id_to_show]['images'][0])
masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_to_show]['masks']]




""" segmenting masks """
def segment_mask(mask_unlabeled):
    return ndimage.label(mask_unlabeled)[0]

id_eg = 'd0f2a00d3155c243048bc48944aef93fb08e2258d1fa5f9ccadd9140082bc22f'
image = dict_data[id_eg]['image']
mask_true =  dict_data[id_eg]['mask']
mask_bin = np.where(mask_true > 0, 1, 0)
mask_seg = segment_mask(mask_bin)

h_fig, h_axes = plt.subplots(2, 2)
plt.axes(h_axes[0, 0])
plt.imshow(image)
plt.axes(h_axes[0, 1])
plt.imshow(mask_true)
plt.axes(h_axes[1, 0])
plt.imshow(mask_bin)
plt.axes(h_axes[1, 1])
plt.imshow(mask_seg)


""" IOU """
def cal_prediction_IOU(mask_true, mask_pred):
    """
    calculate the IOU values of every pair of (true_mask, mask_pred)

    :param mask_true: np.array(shape=[H, W], dtype='int'), where every number is the label of that pixel
    :param mask_pred: np.array(shape=[H, W], dtype='int'), where every number is the label of that pixel
    :return: np.array(shape=[num_unique_labels_true, num_unique_labels_pred], dtype='float')
    """
    mask_true = mask_true.astype('int')
    mask_pred = mask_pred.astype('int')
    n_true = mask_true.max()
    n_pred = mask_pred.max()
    IOU_all = np.zeros([n_true, n_pred])
    for i in range(n_true):
        for j in range(n_pred):
            mask_true_cur = mask_true == i + 1
            mask_pred_cur = mask_pred == j + 1
            IOU_all[i, j] = np.sum((mask_true_cur & mask_pred_cur)).astype('float') \
                            / np.sum((mask_true_cur | mask_pred_cur))
    return IOU_all


IOU = cal_prediction_IOU(mask_true, mask_seg)

def cal_score_from_IOU(IOU):

    list_thrh = np.arange(0.5, 1.0, 0.05)
    list_score = np.zeros(list_thrh.shape)
    for i, thrh in enumerate(list_thrh):
        TP = np.max(IOU, axis=1) > thrh
        TN = np.logical_not(TP)
        FP = np.max(IOU, axis=0) <= thrh
        nTP = np.sum(TP)
        nTN = np.sum(TN)
        nFP = np.sum(FP)
        score = 1.0*nTP / (nTP + nTN + nFP)
        list_score[i] = score
    return np.mean(list_score), list_score

cal_score_from_IOU(IOU)


""" runline encoding """


def rle_encoding(mask):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    non_zeros = np.where(mask.T.flatten() > 0)[0]    # .T sets Fortran order down-then-right
    run_lengths = []
    for i in range(len(non_zeros)):
        if i==0 or non_zeros[i]-non_zeros[i-1] > 1:
            start = non_zeros[i]
            count = 1
            run_lengths.append(start)
            run_lengths.append(count)
        elif non_zeros[i]-non_zeros[i-1] == 1:
            run_lengths[-1] += 1
    return run_lengths

rle_encoding(mask)




""" test image blur """
import importlib
importlib.reload(utils)

list_pow_l = []
list_pow_h = []
for image_id in data_test:
    image = data_test[image_id]['image']
    pow_l, pow_h = utils.noise_detect(image)
    list_pow_l.append(pow_l)
    list_pow_h.append(pow_h)

plt.scatter(list_pow_l, list_pow_h, s=1)
plt.plot([0, 20], [0, 20*0.75])
##
image_id = random.choice(list(data_test.keys()))
image = data_test[image_id]['image']
pow_l, pow_h = utils.noise_detect(image)
print('{:.2f}, {:.2f}, {:.2f}'.format(pow_l, pow_h, pow_h/pow_l))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(utils.noise_blur(image))
