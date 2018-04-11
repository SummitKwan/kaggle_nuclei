""" a list of tools for the competition """

import os
import warnings
import zipfile
import pickle
import random

from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndimage
import matplotlib as mpl
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


def plot_mask2D(mask2D):

    num_mask = mask2D.max()
    cmap = np.random.rand(num_mask+1, 3)*0.8 + 0.2
    cmap[0, :] = 0
    plt.imshow(mask2D, cmap=mpl.colors.ListedColormap(cmap))



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
        id_to_plot = random.choice(list(dict_data.keys()))
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
        plot_mask2D(masks)
        # list_mask_colors = np.random.rand(len(masks), 3)
        # list_mask_colors = np.append(list_mask_colors, [[0.9]]*len(masks), axis=1)
        # for i in range(masks.max()):
        #     mask = (masks == i+1)*255
        #     mask_to_plot = (mask[:, :, None]*list_mask_colors[i][None, None, :]).astype('uint8')
        #     plt.imshow(mask_to_plot)
        plt.axis('image')
        plt.xticks([])
        plt.yticks([])
        plt.title(len(np.unique(masks))-1, fontsize='x-small')


def get_contour(mask3D, edge=3):
    """ generate 2D array of mask contour """
    mask_contour = np.zeros(mask3D.shape[:2])
    for i in range(mask3D.shape[2]):
        mask_contour += mask3D[:, :, i] - ndimage.binary_erosion(mask3D[:, :, i], iterations=edge)
    return mask_contour


def gen_mask_contour(mask_pred=None, mask_true=None, image=None):

    img_shape = [0, 0]
    if mask_pred is None:
        img_shape = mask_true.shape[:2]
    else:
        img_shape = mask_pred.shape[:2]

    if image is None:
        contour_compare = np.zeros(img_shape+(3, ))
    else:
        contour_compare = image*0.5

    if mask_pred is not None:   # pred in red
        contour_compare[:, :, 0] += get_contour(mask_pred) * 255
        contour_compare[:, :, 2] -= get_contour(mask_pred) * 255
    if mask_true is not None:   # true in green
        contour_compare[:, :, 1] += get_contour(mask_true) * 255
        contour_compare[:, :, 2] -= get_contour(mask_true) * 255

    contour_compare = np.clip(contour_compare, 0, 255).astype('uint8')

    return contour_compare


""" ========== image manipulation  ========== """

def segment_mask(mask_unlabeled):
    """
     segmenting masks

    :param mask_unlabeled: unlabled masks, binary array
    :return: labeled mask, int array, where 0 is background, and 1,2,3... is labels
    """
    return ndimage.label(mask_unlabeled)[0]


def mask_2Dto3D(mask_2D):
    """ represent masks in np.array(shape=(N, M, num_masks)), where values are either 0 or 1 """
    labels_mask = np.unique(mask_2D)
    labels_mask = labels_mask[labels_mask>0]
    n, m = mask_2D.shape
    k = len(labels_mask)
    mask_3D = np.zeros(shape=(n, m, k), dtype='uint8')
    for i_label, label in enumerate(labels_mask):
        mask_3D[:, :, i_label] = (mask_2D == label)
    return mask_3D


def mask_3Dto2D(mask_3D, labels=None):
    """ represent masks in np.array(shape=(N, M)), where values range from 0 to number_masks, coding labels """

    """ in case of overlapping masks, the mask with small label index overwrites the one with larger label index """
    n, m, k = mask_3D.shape
    mask_2D = np.zeros(shape=(n, m), dtype='uint16')
    if labels is None:
        labels = range(k)
    i_labels = np.argsort(labels)
    for i in i_labels[::-1]:
        mask_2D[mask_3D[:, :, i]>0] = i+1
    return mask_2D



""" ========== performance evaluation ========== """


def cal_prediction_IOU(mask_true, mask_pred):
    """
    calculate the IOU values of every pair of (true_mask, mask_pred)

    :param mask_true: np.array(shape=[H, W], dtype='int'), where every number is the label of that pixel
    :param mask_pred: np.array(shape=[H, W], dtype='int'), where every number is the label of that pixel
    :return: np.array(shape=[num_unique_labels_true, num_unique_labels_pred], dtype='float')
    """
    mask_true = mask_true.astype('int').ravel()
    mask_pred = mask_pred.astype('int').ravel()
    mask_true_unq = np.unique(mask_true[mask_true>0])
    mask_pred_unq = np.unique(mask_pred[mask_pred>0])
    indx_mask_true = [np.where(mask_true == label)[0] for label in mask_true_unq]
    indx_mask_pred = [np.where(mask_pred == label)[0] for label in mask_pred_unq]
    IOU_all = np.zeros([len(indx_mask_true), len(indx_mask_pred)], dtype='float')
    for i, indx_true in enumerate(indx_mask_true):
        for j, indx_pred in enumerate(indx_mask_pred):
            if indx_true.min()>indx_pred.max() or indx_true.max()<indx_pred.min():
                IOU_all[i, j] = 0.0
            else:
                IOU_all[i, j] = np.intersect1d(indx_true, indx_pred).size / np.union1d(indx_true, indx_pred).size

    # n_true = mask_true.max()
    # n_pred = mask_pred.max()
    # IOU_all = np.zeros([n_true, n_pred])
    # for i in range(n_true):
    #     for j in range(n_pred):
    #         mask_true_cur = mask_true == i + 1
    #         mask_pred_cur = mask_pred == j + 1
    #         IOU_all[i, j] = np.sum((mask_true_cur & mask_pred_cur)).astype('float') \
    #                         / np.sum((mask_true_cur | mask_pred_cur))
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
    return {'ave': np.mean(list_score), 'all': list_score}



def rle_encoding(mask):
    '''
    reline encoding of masks
    mask: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    non_zeros = np.where(mask.T.flatten() > 0)[0]  # .T sets Fortran order down-then-right
    non_zeros = non_zeros + 1   # the official site asks the result to be 1-indexed
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


""" ========== image split and stitch ========== """


def floor_pow2(n, yn_half=True):
    res = 2 ** np.floor(np.log2(n))
    if yn_half:
        if n >= res*1.5:
            res = res*1.5
    return int(res)


def cal_img_split_start_index(img, size_seg=128, overlap=0.2):
    """
     split image to small segments, returns t

    :param img:      input image (shape=(m, m, k)) or size of image (m, n)
    :param size_seg: size of small segments, (length of square in pixels)
    :param overlap:  minimal proportion of overlap between neighboring segments (in 1D)
    :return:         (starting_indexes_of_rows, starting_indexes_of_columns)
    """

    if np.array(img).size <= 3:
        m, n = img[:2]
    else:
        m, n = img.shape[:2]

    def cal_split_start_index(size_img, size_seg):
        """ cal 1D split, returns the starting index of every segment """
        if size_img < size_seg:
            res_split = np.array([0])
        else:
            num_split_r, rem = divmod((size_img-size_seg), (int(size_seg*(1-overlap))))
            num_split_r = num_split_r + (rem > 0)
            res_split = np.linspace(0, size_img-size_seg, num_split_r+1).astype('int')
        return res_split

    r_split = cal_split_start_index(m, size_seg)
    c_split = cal_split_start_index(n, size_seg)

    return r_split, c_split


def img_split(img, size_seg=128, overlap=0.2):
    """
    split image to small segments of size<=size_seg and overlapping_proportion>=overlap

    :param img:      input image (shape=(m, m, k)) or size of image (m, n)
    :param size_seg: size of small segments, (length of square in pixels)
    :param overlap:  minimal proportion of overlap between neighboring segments (in 1D)
    :return:         [list_of_images]
    """

    r_split, c_split = cal_img_split_start_index(img, size_seg, overlap)
    segment_starting_index = []
    segment_img = []
    for r in r_split:
        for c in c_split:
            segment_starting_index.append((r, c))
            segment_img.append(img[r:r+size_seg, c:c+size_seg])
    return dict(zip(segment_starting_index, segment_img))


def img_stitch(segment_img, mode='image', info_mask_dict=None):
    """
    stitch images together

    :param segment_img: dict((starting_row, starting_col): np.array of image data )
    :param mode:  'image' or 'mask'
    :return: np.array of the full image
    """

    rc_split = list(segment_img.keys())
    r_split, c_split = zip(*rc_split)
    r_split = np.sort(np.unique(r_split))
    c_split = np.sort(np.unique(c_split))
    img_shape = segment_img[(r_split[0], c_split[0])].shape
    r_size_seg = img_shape[0]
    c_size_seg = img_shape[1]
    r_size_full = r_split[-1] + r_size_seg
    c_size_full = c_split[-1] + c_size_seg

    if mode == 'image':
        img_full_shape = (r_size_full, c_size_full) + img_shape[2:]
        img_full = np.zeros(shape=img_full_shape, dtype=segment_img[(r_split[0], c_split[0])].dtype)

        for r in r_split:
            for c in c_split:
                img_full[r:r+r_size_seg, c:c+c_size_seg] = segment_img[(r, c)]
        return img_full

    elif mode == 'mask':
        # gather all masks in the full image coordinate
        dtype = segment_img[(r_split[0], c_split[0])].dtype
        mask_full_list = []    # mask filled in the full image coordinate,
        mask_fseg_list = []    # mask form which segment, segment indexed using (r,c), as the key of input
        mask_info_list = []    # mask info, like score
        tf_keep_list = []      # true or false to keep the mask

        if len(r_split) <= 1:
            r_edge = 0
        else:
            r_edge = int((r_size_seg-r_split[1])/2)
        if len(c_split) <= 1:
            c_edge = 0
        else:
            c_edge = int((c_size_seg-c_split[1])/2)

        for i_r, r in enumerate(r_split):
            for i_c, c in enumerate(c_split):
                mask_cur = segment_img[(r, c)]

                mask_colloapse_r = np.sum(mask_cur, axis=1)
                mask_colloapse_c = np.sum(mask_cur, axis=0)
                mask_size = np.sum(mask_colloapse_r, axis=0).astype('int')

                mask_size[mask_size == 0] = -1   # prevent zero division error
                tf_keep_mask = (mask_size > 0)

                mask_center_r = np.sum(np.arange(r_size_seg)[:, None] * mask_colloapse_r, axis=0) / mask_size
                mask_center_c = np.sum(np.arange(c_size_seg)[:, None] * mask_colloapse_c, axis=0) / mask_size

                if i_r > 0:
                    tf_keep_mask = tf_keep_mask & (mask_center_r >= r_edge)
                if i_r < len(r_split) - 1:
                    tf_keep_mask = tf_keep_mask & (mask_center_r < r_size_seg - r_edge)
                if i_c > 0:
                    tf_keep_mask = tf_keep_mask & (mask_center_c >= c_edge)
                if i_c < len(c_split) - 1:
                    tf_keep_mask = tf_keep_mask & (mask_center_c < c_size_seg - c_edge)
                mask_cur_keep = mask_cur[:, :, tf_keep_mask]
                mask_full_cur = np.zeros(shape=(r_size_full, c_size_full, np.sum(tf_keep_mask)), dtype=dtype)
                mask_full_cur[r:r+r_size_seg, c:c+c_size_seg] = mask_cur_keep
                mask_full_list.append(mask_full_cur)
                mask_fseg_list.extend([(r, c)]*np.sum(tf_keep_mask))
                if info_mask_dict is not None:
                    mask_info_list.append(info_mask_dict[(r, c)][tf_keep_mask])
        mask_full = np.dstack(mask_full_list)
        mask_fseg = np.array(mask_fseg_list)
        mask_info = np.concatenate(mask_info_list)
        return mask_full, mask_fseg, mask_info
    else:
        warnings.warn('mode should be either "image" or "mask"')
        return None







