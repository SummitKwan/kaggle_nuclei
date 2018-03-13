""" a list of tools for the competition """

import os
import warnings
import zipfile
import pickle

from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


""" ========== data processing: unzip, get id, store to a single file, load ========== """

path_data = './data'
data_urls = [
        'https://www.kaggle.com/c/8089/download/stage1_train.zip',
        'https://www.kaggle.com/c/8089/download/stage1_train_labels.csv.zip',
        'https://www.kaggle.com/c/8089/download/stage1_test.zip',
        'https://www.kaggle.com/c/8089/download/stage1_sample_submission.csv.zip'
    ]


def unzip_data(data_urls=data_urls, path_data=path_data):
    """ unzip data files downloaded from Kaggle, stored at ./data """

    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    for data_url in data_urls:
        data_file_zip = os.path.basename(data_url)
        if data_file_zip not in os.listdir(path_data):
            warnings.warn('please download data file {} manually to {}'.format(data_url, path_data))
            break
            # the line below does not work, since kaggle needs username and password to download files
            # urllib.request.urlretrieve(data_url, os.path.join(path_data, data_file_zip))
        data_file_unzip = data_file_zip[:-4]
        if '.' in data_file_unzip:
            path_extract = path_data
        else:
            path_extract = os.path.join(path_data, data_file_unzip)
        if data_file_unzip in os.listdir(path_data):
            print('{} exsists in {}, no need to unzip'.format(data_file_zip, path_data))
        else:
            print('unzipping {} to {}'.format(data_file_zip, path_data))
            zipfile.ZipFile(os.path.join(path_data, data_file_zip), 'r') \
                .extractall(path_extract)


def gen_id_path_dict(path_data=path_data):
    """
    get mapping of id and image/mask file paths

    returns a dictionary of {image_id: {'images': list_of_image_file_path, 'masks': list_of_mask_file_path}}
    """
    ids_img = os.listdir(path_data)
    ids_img = [id_img for id_img in ids_img
               if os.path.isdir(os.path.join(path_data, id_img))
               and 'images' in os.listdir(os.path.join(path_data, id_img))]
    dict_ids = {}
    for id_img in ids_img:
        list_img_names = os.listdir(os.path.join(path_data, id_img, 'images'))
        list_img_path = [os.path.join(path_data, id_img, 'images', img_name)
                         for img_name in list_img_names if img_name[-3:] == 'png']
        if list_img_path:
            dict_ids[id_img] = {'images': list_img_path}
        if 'masks' in os.listdir(os.path.join(path_data, id_img)):
            list_mask_names = os.listdir(os.path.join(path_data, id_img, 'masks'))
            list_mask_path = [os.path.join(path_data, id_img, 'masks', img_name)
                              for img_name in list_mask_names if img_name[-3:] == 'png']
            if list_mask_path:
                dict_ids[id_img]['masks'] = list_mask_path

    return dict_ids


def create_data_file(dict_id_path, filename, filepath=path_data):
    """
    store data in a single file:
    dict_data = {img_id: {'image': np.array of shape (M, N, 3), 'mask': np.array of shape (N, N) }}
    where image is a (M,N,3) RGB array, with dtype=uint8,
    and mask is a (M,N) array, where value 0 represents background and value 1 to num_nuclei represents every label
    """
    if os.path.isfile(os.path.join(filepath, filename)):
        print('data file {} already exists in {}, no need to do that'.format(filename, filepath))
        return None

    dict_data = {}
    for id_img in tqdm(dict_id_path):
        img = ndimage.imread(dict_id_path[id_img]['images'][0])[:, :, :3]
        if 'masks' in dict_id_path[id_img]:
            list_masks = [ndimage.imread(mask_file) for mask_file in dict_id_path[id_img]['masks']]
            num_masks = len(list_masks)
            array_masks = np.zeros(img.shape[:2], dtype='uint16')
            for i_mask, mask in enumerate(list_masks):
                array_masks[mask > 0] = i_mask+1
        else:
            array_masks = np.array([])
        dict_data[id_img] = {'image': img, 'mask': array_masks}

    # save to disk
    path_dict_data = os.path.join(filepath, filename)
    with open(path_dict_data, 'wb') as f:
        pickle.dump(dict_data, f)


def load_data(filename, filepath=path_data):
    """ load data saved using create_data_file() """
    with open(os.path.join(filepath, filename), 'rb') as f:
        dict_data = pickle.load(f)
    return dict_data


""" ========== data visualization ========== """

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


def plot_img_and_mask_from_file(dict_ids, id_to_plot=None):
    """ plot by reading image form file """
    if id_to_plot is None:
        id_to_plot = np.random.choice(list(dict_ids.keys()))
    elif id_to_plot not in dict_ids:
        warnings.warn('given id does not exist in the data, id={}'.format(id_to_plot))
        return None
    img = ndimage.imread(dict_ids[id_to_plot]['images'][0])
    bool_mask = ('masks' in dict_ids[id_to_plot])
    h_ax = plt.gca()
    if bool_mask:
        masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_to_plot]['masks']]
        h_ax_sub = add_sub_axes(h_axes=h_ax, loc='right', size=0.5, gap=0)
    plt.axes(h_ax)
    plt.imshow(img)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(img.shape, fontsize='x-small')

    if bool_mask:
        plt.axes(h_ax_sub)
        # plt.imshow(img)
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


def plot_img_and_mask_from_dict(dict_data, id_to_plot=None):
    """ plot data from the defined data structure """

    if id_to_plot is None:
        id_to_plot = np.random.choice(list(dict_data.keys()))
    elif id_to_plot not in dict_data:
        warnings.warn('given id does not exist in the data, id={}'.format(id_to_plot))
        return None
    img = dict_data[id_to_plot]['image']
    h_ax = plt.gca()
    plt.axes(h_ax)
    plt.imshow(img)
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title(img.shape, fontsize='x-small')
    bool_mask = ('mask' in dict_data[id_to_plot]) and (dict_data[id_to_plot]['mask'].size > 0)
    if bool_mask:
        masks = dict_data[id_to_plot]['mask']
        h_ax_sub = add_sub_axes(h_axes=h_ax, loc='right', size=0.5, gap=0)
        plt.axes(h_ax_sub)
        # plt.imshow(img)
        list_mask_colors = np.random.rand(len(masks), 3)
        list_mask_colors = np.append(list_mask_colors, [[0.9]]*len(masks), axis=1)
        for i in range(masks.max()):
            mask = (masks == i+1)*255
            mask_to_plot = (mask[:, :, None]*list_mask_colors[i][None, None, :]).astype('uint8')
            plt.imshow(mask_to_plot)
        plt.axis('image')
        plt.xticks([])
        plt.yticks([])
        plt.title(masks.max(), fontsize='x-small')


""" ========== performance evaluation ========== """

def segment_mask(mask_unlabeled):
    """
     segmenting masks

    :param mask_unlabeled: unlabled masks, binary array
    :return: labeled mask, int array, where 0 is background, and 1,2,3... is labels
    """
    return ndimage.label(mask_unlabeled)[0]


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


def cal_score_from_IOU(IOU):
    """
    calculate scores at various threshold as defined by the competition

    :param IOU: result from cal_prediction_IOU()
    :return: list of scores at every threshold level
    """

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
    return {'ave': np.mean(list_score), 'all':list_score}

