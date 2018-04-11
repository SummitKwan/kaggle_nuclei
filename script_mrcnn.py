""" script to try mask-rcnn """

import os
import sys
import random
import time
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import pickle
import copy
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2
import utils

# make sure the Mask_RCNN package exists under '..'
MRCNN_PKG_DIR = '..'
sys.path.append(MRCNN_PKG_DIR)
import Mask_RCNN as mrcnn



# os.environ['CUDA_VISIBLE_DEVICES']="1"

""" specify directories """

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn_logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

##
""" get data """

with open('./data/data_train.pickle', 'rb') as f:
    data_train = pickle.load(f)
with open('./data/data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)

def add_class_to_data_dict(data_dict):
    for key in data_dict:
        img = data_dict[key]['image']
        mask = data_dict[key]['mask']
        img_bw = np.mean(img, axis=2)
        lumi_fg = np.average(img_bw, weights=(mask>0))
        lumi_bg = np.average(img_bw, weights=(mask==0))
        img_class = 'dark' if lumi_fg > lumi_bg else 'bright'
        data_dict[key]['class'] = img_class
    return data_dict

data_train = add_class_to_data_dict(data_train)


""" prepare data for training """
yn_flip_dark_field = True
yn_augment = True
yn_split = True
yn_split_balance = True

def process_data(data_train, yn_flip_dark_field=True, yn_augment=True, yn_split=True, yn_split_balance=True, max_num_seg=12):
    data_train_after = copy.deepcopy(data_train)

    data_train_after = add_class_to_data_dict(data_train_after)

    # flip dark field
    if yn_flip_dark_field:
        for id_img in data_train_after:
            if data_train_after[id_img]['class'] == 'dark':
                data_train_after[id_img]['image'] = 255 - data_train_after[id_img]['image']

    # data augmentation by rotation
    if yn_augment:
        data_train_before, data_train_after = data_train_after, {}
        for id_img in data_train_before:
            image_cur = data_train_before[id_img]['image']
            mask_cur = data_train_before[id_img]['mask']
            if 'class' in data_train_before[id_img]:
                class_cur = data_train_before[id_img]['class']

            for i in range(4):
                id_img_new = id_img + '_rot{}'.format(i)
                data_train_after[id_img_new] = {}
                data_train_after[id_img_new]['image'] = np.rot90(image_cur, i)
                data_train_after[id_img_new]['mask'] = np.rot90(mask_cur, i)
                if 'class' in data_train_before[id_img]:
                    data_train_after[id_img_new]['class'] = class_cur

    # data split based on nuclei size
    temp = {}

    if yn_split:
        data_train_before, data_train_after = data_train_after, {}
        amplification_ideal = 12   # ideal size_patch/size_nuclei
        min_size_seg = 64
        for id_img in tqdm(data_train_before.keys()):
            img_cur  = data_train_before[id_img]['image']
            mask_cur = data_train_before[id_img]['mask']
            if 'class' in data_train_before[id_img]:
                class_cur = data_train_before[id_img]['class']

            size_nuclei = np.sqrt(np.sum(mask_cur > 0)*1.0 / (len((np.unique(mask_cur)))+1))
            size_nuclei = max(size_nuclei, 8)
            size_seg_opi = size_nuclei * amplification_ideal
            size_seg_0 = utils.floor_pow2(size_seg_opi)
            size_seg_0 = max(min_size_seg, size_seg_0)
            list_size_seg = [size_seg_0, size_seg_0*2]
            for size_seg in list_size_seg:
                img_cur_seg  = utils.img_split(img=img_cur,  size_seg=size_seg)
                mask_cur_seg = utils.img_split(img=mask_cur, size_seg=size_seg)
                for start_loc in img_cur_seg.keys():
                    data_train_after[(id_img, start_loc, size_seg)] = {}
                    data_train_after[(id_img, start_loc, size_seg)]['image'] = img_cur_seg[start_loc]
                    data_train_after[(id_img, start_loc, size_seg)]['mask'] = mask_cur_seg[start_loc]
                if 'class' in data_train_before[id_img]:
                    data_train_after[(id_img, start_loc, size_seg)]['class'] = class_cur

    # balance dataset
    if yn_split and yn_split_balance:
        data_train_before, data_train_after = data_train_after, {}
        data_train_after = {}
        seg_ids = list(data_train_before.keys())
        dict_img_seg = {}
        for seg_id in seg_ids:
            img_id = seg_id[0]
            if img_id not in dict_img_seg:
                dict_img_seg[img_id] = []
            else:
                dict_img_seg[img_id].append(seg_id)
        # plt.subplot(1,2,1)
        # plt.hist([len(value) for key, value in dict_img_seg.items()], 20)

        for seg_id in dict_img_seg:
            num_split = len(dict_img_seg[seg_id])
            if num_split > max_num_seg:   # if too many, select
                for id_new in random.choices(dict_img_seg[seg_id], k=max_num_seg):
                    data_train_after[id_new + (0, )] = data_train_before[id_new]
            else:
                for i in range(max_num_seg//num_split):
                    for id_new in dict_img_seg[seg_id]:
                        data_train_after[id_new + (i,)] = data_train_before[id_new]


    data_train_processed = data_train_after

    return data_train_processed

data_train_processed = process_data(data_train)


##
""" use external data """

if True:
    # with open('./data/data_train.pickle', 'rb') as f:
    #     data_train_official = pickle.load(f)
    with open('./data/data_train_fix.pickle', 'rb') as f:
        data_train_official = pickle.load(f)
    with open('./data/data_train_Amit.pickle', 'rb') as f:
        data_train_Amit = pickle.load(f)
    with open('./data/data_train_TNBC.pickle', 'rb') as f:
        data_train_TNBC = pickle.load(f)

    print(len(data_train_official), len(data_train_Amit), len(data_train_TNBC))

    data_train_official_processed = process_data(data_train_official, yn_flip_dark_field=True, yn_augment=True,
                                                 yn_split=True, yn_split_balance=True, max_num_seg=12)
    data_train_Amit_processed = process_data(data_train_Amit, yn_flip_dark_field=True, yn_augment=True,
                                                 yn_split=True, yn_split_balance=True, max_num_seg=24)  # 48
    data_train_TNBC_processed = process_data(data_train_TNBC, yn_flip_dark_field=True, yn_augment=True,
                                                 yn_split=True, yn_split_balance=True, max_num_seg=12)  # 24

    print(len(data_train_official_processed), len(data_train_Amit_processed), len(data_train_TNBC_processed))

    data_train_processed = {**data_train_official_processed, **data_train_Amit_processed, **data_train_TNBC_processed}

    print(len(data_train_processed))


##
""" create mrcnn class for using the model """

yn_mean_pixel_use_255 = False

class NucleiConfig(mrcnn.config.Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 2 foreground

    if yn_mean_pixel_use_255:
        # mean RGB value, for flipped images (bright field), use 255
        MEAN_PIXEL = np.array([255.0, 255.0, 255.0])

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 128

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = NucleiConfig()
config.display()


class_bright_dark = False

class NucleiDataset(mrcnn.utils.Dataset):

    def add_data_dict(self, data_dict):
        self.data_dict = data_dict
        if class_bright_dark:
            self.add_class("nuclei", 1, "dark")
            self.add_class("nuclei", 2, "bright")
        else:
            self.add_class("nuclei", 1, "all")
        for i, key in enumerate(data_dict):
            self.add_image('nuclei', i, key)

    def load_image(self, image_id):
        return self.data_dict[self.image_info[image_id]['path']]['image']

    def load_mask(self, image_id):
        mask_2D = self.data_dict[self.image_info[image_id]['path']]['mask']
        mask_3D = utils.mask_2Dto3D(mask_2D)

        class_ids = np.ones(mask_3D.shape[2], dtype=np.int32)
        if class_bright_dark:
            if self.data_dict[self.image_info[image_id]['path']]['class'] == 'dark':
                class_ids = np.ones(mask_3D.shape[2], dtype=np.int32)
            elif self.data_dict[self.image_info[image_id]['path']]['class'] == 'bright':
                class_ids = np.ones(mask_3D.shape[2], dtype=np.int32) * 2

        mask = mask_3D

        return mask, class_ids

##
""" prepare for model dataset """


proportion_val = 0.1
keys_data = list(data_train_processed.keys())
keys_data_shuffle =  random.sample(keys_data, k=len(keys_data))

num_train = int((1-proportion_val)*len(keys_data))
keys_train = keys_data_shuffle[:num_train]
keys_val   = keys_data_shuffle[num_train:]


dataset_train = NucleiDataset()
dataset_train.add_data_dict({key:data_train_processed[key] for key in keys_train})
dataset_train.prepare()

dataset_val = NucleiDataset()
dataset_val.add_data_dict({key:data_train_processed[key] for key in keys_val})
dataset_val.prepare()


# fig, ax = plt.subplots(1,2)
# i = 9
# plt.subplot(ax[0])
# plt.imshow(dataset_train.load_image(i))
# plt.subplot(ax[1])
# plt.imshow(dataset_train.load_mask(i)[0][:,:,0])


model = mrcnn.model.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
init_with = "specific"  # imagenet, coco, last, or specific
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
elif init_with == "specific":
    # use specific
    # model.load_weights(os.path.join(MODEL_DIR,
    #                                 'starting_point', '20180406T1458e0015.h5'), by_name=True)
    # model.load_weights(os.path.join(MODEL_DIR,
    #                                 'nuclei20180409T0200', 'mask_rcnn_nuclei_0015.h5'), by_name=True)
    # model.load_weights(os.path.join(MODEL_DIR,
    #                                 'nuclei20180410T2030', 'mask_rcnn_nuclei_0015.h5'), by_name=True)
    model.load_weights(os.path.join(MODEL_DIR,
                                    'nuclei20180410T1310', 'mask_rcnn_nuclei_0029.h5'), by_name=True)

##
"""  Training """

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/8,
            epochs=40,
            layers='4+')



# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=25,
#             layers="all")


## Test
""" model prediction """

image_per_GPU = 1

class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = image_per_GPU

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = mrcnn.model.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# specify model path
# model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
#                           'nuclei20180406T1458_bbox_bug_fix',
#                           "mask_rcnn_nuclei_0015.h5")
# model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
#                           'nuclei20180407T1423_train_with_external_data',
#                           "mask_rcnn_nuclei_0010.h5")
# model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
#                           'nuclei20180408T1807',
#                           "mask_rcnn_nuclei_0040.h5")
# model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
#                           'nuclei20180409T0200',
#                           "mask_rcnn_nuclei_0035.h5")
# model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
#                           'nuclei20180410T1310',
#                           "mask_rcnn_nuclei_0029.h5")
model_path = os.path.join(ROOT_DIR, 'mrcnn_logs',
                          'nuclei20180406T1458_bbox_bug_fix',
                          "mask_rcnn_nuclei_0015.h5")



# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


## plot a random detection result
""" plot a random detection result """

image_id = random.choice(list(data_test.keys()))

# image_id = '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466'  # purple
# image_id = '432f367a4c5b5674de2e2977744d10289a064e5704b21af6607b4975be47c580'  # many cells
# image_id = '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac'  # dim cells
# image_id = '295682d9eb5acb5c1976a460c085734bfaf38482b0a3f02591c2bfdcd4128549'
# image_id = '0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac'  # blood vessel
image = data_test[image_id]['image']

# image_id = random.choice(list(data_train.keys()))
# image = data_train[image_id]['image']


def image_dn(image=image):
    if np.mean(image) < 100:
        return 255-image
    else:
        return image


def mean_std_weighted(x, w=None):
    if w is None:
        w = np.ones(x.shape)
    x_mean = np.average(x, weights=w)
    x_std = np.sqrt(np.average((x-x_mean)**2, weights=w))
    return x_mean, x_std


def bound_modified_z(ys, threshold=3.5):

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    bound_size = threshold * median_absolute_deviation_y / 0.6745
    # modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
    #                      for y in ys]
    # np.where(np.abs(modified_z_scores) > threshold)
    return median_y-bound_size, median_y+bound_size

def bound_iqr(ys, reject_scale=1.8):
    quartile_1, quartile_3 = np.percentile(ys, (25, 75))
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * reject_scale)
    upper_bound = quartile_3 + (iqr * reject_scale)
    return lower_bound, upper_bound


def remove_outlier(image, mask3D, score=None):
    """ filter out outliers: score>0.8, size and luminance within 3 std """

    n = mask3D.shape[2]
    if score is None:
        score = np.ones(n)
    mask_size = np.sqrt(np.sum(mask3D, axis=(0, 1)))
    yn_keep = (mask_size > 3) & (score > 0.70)
    if n > 10:
        image_bw = np.mean(image, axis=2)
        mask_lumi = np.array([np.mean(np.average(image_bw, weights=mask3D[:, :, i])) for i in range(n)])
        back_lumi = np.mean(image_bw[np.sum(mask3D, axis=2) < 1])

        # size_mean, size_std = mean_std_weighted(mask_size[yn_keep], score[yn_keep])
        # lumi_mean, lumi_std = mean_std_weighted(mask_lumi[yn_keep], score[yn_keep])
        #
        # z_size = (mask_size - size_mean) / size_std

        # yn_keep = (score > 0.7) & (np.abs(z_size) < 3.5) & (mask_size > 4) \
        #           & (((lumi_mean-back_lumi)*z_lumi>0) | (np.abs(z_lumi) < 3)) & ((mask_lumi-back_lumi)*(lumi_mean-back_lumi)>0)

        # yn_keep = yn_keep & np.logical_not(np.abs(z_size) > 3.5) \
        #           & (np.abs(mask_lumi-back_lumi)*4 > np.abs(mask_lumi-lumi_mean))

        if n < 48:
            threshold_bound_modified_z = 8.0
        else:
            threshold_bound_modified_z = 6.0

        bound_size = np.exp(bound_modified_z(np.log(mask_size[yn_keep]), threshold_bound_modified_z))
        bound_lumi = bound_modified_z(mask_lumi[yn_keep], threshold_bound_modified_z)
        yn_keep = yn_keep & (mask_size >= bound_size[0]) & (mask_size <= bound_size[1]) \
                  & (mask_lumi >= bound_lumi[0]) & (mask_lumi <= bound_lumi[1]) \
                  & (np.abs(mask_lumi-back_lumi)*4 > np.abs(mask_lumi-np.mean(mask_lumi)))

    return yn_keep


def remove_overlap(mask3D, labels=None):
    """ in case of overlapping masks, the mask with small label index overwrites the one with larger label index """
    n, m, k = mask3D.shape
    if labels is None:
        labels = np.random.permutation(k)
    i_sort = np.argsort(np.argsort(labels))
    mask3D_res = np.zeros(shape=mask3D.shape, dtype=mask3D.dtype)
    for i in range(k):
        mask_cur = mask3D[:, :, i]
        mask_to_compare = mask3D[:, :, i_sort<i_sort[i]]
        pixel_overlap = mask_cur[:, :, None] * mask_to_compare
        yn_overlap = np.sum(pixel_overlap, axis=(0, 1)) > 0

        if np.any(yn_overlap):

            pixel_to_remove = (np.sum(pixel_overlap, axis=2) > 0)

            area_overlap = np.sum(pixel_overlap, axis=(0, 1))

            yn_discard_cur = np.any( (area_overlap[yn_overlap] > 0.5 * mask_cur.sum()) |
                                     (area_overlap[yn_overlap] > 0.5 * np.sum(mask_to_compare[:,:, yn_overlap], axis=(0,1))) )
            if yn_discard_cur:
                mask3D_res[:, :, i] = 0
            else:
                mask3D_res[:, :, i] = mask3D[:, :, i] - pixel_to_remove * mask3D[:, :, i]

        else:
            mask3D_res[:, :, i] = mask3D[:, :, i]

    return mask3D_res


def post_process(image, mask, score, flag_remove_outlier=True, flag_remove_overlap=True):

    yn_keep = np.zeros(mask.shape[2])

    # remove outlier:
    if flag_remove_outlier:
        yn_keep_from_outlier = remove_outlier(image, mask, score)
    else:
        yn_keep_from_outlier = np.ones(len(mask.shape[2]), dtype='bool')

    ID_yn_keep_from_outlier = np.where(yn_keep_from_outlier)[0]
    mask_cleaned = mask[:, :, yn_keep_from_outlier]
    score_cleaned = score[yn_keep_from_outlier]

    # remove overlap:
    if flag_remove_overlap:
        mask_cleaned = remove_overlap(mask_cleaned, -score_cleaned)
        # remove empty mask:
        yn_keep_from_overlap = np.sum(mask_cleaned, axis=(0, 1)) > 0
        mask_cleaned = mask_cleaned[:, :, yn_keep_from_overlap]
        yn_keep[ID_yn_keep_from_outlier[yn_keep_from_overlap]] = 1
        yn_keep = (yn_keep == 1)
    else:
        yn_keep = ID_yn_keep_from_outlier

    return yn_keep, mask_cleaned


def gen_mask_by_seg(image=image, size_seg=512, flag_use_dn=True):

    # flip sign of the dark field images
    if flag_use_dn:
        image = image_dn(image)

    image_segs = utils.img_split(image, size_seg=size_seg, overlap=0.4)

    # split, cmpute mask, stitch
    mask_segs = {}
    scores_segs = {}
    for loc_seg, image_seg in image_segs.items():
        image_detect_res = model.detect([image_seg], verbose=0)[0]
        masks_seg = image_detect_res['masks']
        scores_seg = image_detect_res['scores']
        if masks_seg.shape[:2] != image_seg.shape[:2]:   # special case if no mask
            masks_seg = np.zeros(shape=image_seg.shape[:2]+(0, ), dtype='int')
            scores_seg = np.zeros(shape=(0, ))
        mask_segs[loc_seg] = masks_seg
        scores_segs[loc_seg] = scores_seg
    mask_stitched, _, score_stiched = utils.img_stitch(mask_segs, mode='mask', info_mask_dict=scores_segs)

    if mask_stitched.shape[2] != 0:
        mask_size = np.average(np.sqrt(np.sum(mask_stitched, axis=(0, 1))), weights=score_stiched)
    else:
        mask_size = 16.0

    return mask_stitched, score_stiched, mask_size


def gen_mask_by_seg_iter(image=image, size_seg_ini=512, flag_plot=False, flag_use_dn=True):

    amplification_best = 12   # 12
    amplification_min = 10
    amplification_max = 14

    mask_stitched = None
    size_seg = size_seg_ini
    for i in range(5):
        mask_stitched, score_stitched, mask_size = gen_mask_by_seg(image, size_seg, flag_use_dn=flag_use_dn)
        yn_keep_mask, mask_post = post_process(image, mask_stitched, score_stitched,
                                               flag_remove_outlier=True, flag_remove_overlap=False)
        num_nuclei = mask_post.shape[2]
        size_nuclei = np.sqrt(np.sum(mask_post > 0) * 1.0 / num_nuclei)
        if flag_plot:
            plot_post_process(image=image, mask_org=mask_stitched, mask_post=mask_post, score=score_stitched,
                              yn_keep=yn_keep_mask, size_seg=size_seg, mask_size=mask_size)
        size_seg_new = utils.floor_pow2(size_nuclei * amplification_best)  # int(mask_size * amplification_best)
        if not(np.isfinite(size_nuclei)) or (size_seg == size_seg_new)\
                or (amplification_min < size_seg/size_nuclei < amplification_max):
            break
        else:
            size_seg = max(size_seg_new, 64)

    return mask_stitched, score_stitched, size_seg, mask_size


def plot_post_process(image, mask_org, mask_post, size_seg=0, mask_size=0, score=None, yn_keep=None):

    h_fig = plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('{}, size_seg={}'.format(image.shape[:2], size_seg))
    plt.subplot(2, 2, 2)
    utils.plot_mask2D(utils.mask_3Dto2D(mask_post))
    plt.axis('off')
    plt.title('mask_size={:.2f}'.format(mask_size))

    plt.subplot(2, 2, 3)
    mask_org_flat = np.any(mask_org, axis=2)
    mask_post_flat = np.any(mask_post, axis=2)
    plt.imshow(0 + mask_org_flat + (mask_org_flat != mask_post_flat), vmin=0, vmax=2)
    plt.axis('off')
    # plt.title('filtered out {}'.format(np.sum(yn_keep==False)))
    plt.title('n_orig={}, n_post={}'.format(mask_org.shape[2], mask_post.shape[2]))

    try:
        plt.subplot(6, 2, 8)
        plt.hist(score, bins=np.arange(0.5, 1.01, 0.02), alpha=0.5)
        plt.hist(score[yn_keep], bins=np.arange(0.5, 1.01, 0.02), alpha=0.5)
        plt.ylabel('score')

        plt.subplot(6, 2, 10)
        mask_size_org = np.sqrt(np.sum(mask_org, axis=(0, 1)))
        mask_size_post = np.sqrt(np.sum(mask_post, axis=(0, 1)))
        plt.hist(mask_size_org, bins=np.linspace(0, mask_size_org.max(), 20), alpha=0.5)
        plt.hist(mask_size_post, bins=np.linspace(0, mask_size_org.max(), 20), alpha=0.5)
        plt.ylabel('size')

        plt.subplot(6, 2, 12)
        image_bw = np.mean(image, axis=2)
        mask_lum_org = np.array(
            [np.mean(np.average(image_bw, weights=mask_org[:, :, i])) for i in range(mask_org.shape[2])])
        mask_lum_post = np.array(
            [np.mean(np.average(image_bw, weights=mask_post[:, :, i])) for i in range(mask_post.shape[2])])
        plt.hist(mask_lum_org, bins=np.arange(0, 256, 8), alpha=0.5)
        plt.hist(mask_lum_post, bins=np.arange(0, 256, 8), alpha=0.5)
        plt.ylabel('luminance')
    except:
        pass

    return h_fig


mask_stitched, score_stitched, size_seg, mask_size = gen_mask_by_seg_iter(image=image, size_seg_ini=512,
                                                                          flag_plot=True, flag_use_dn=True)
yn_keep_mask, mask_post = post_process(image, mask_stitched, score_stitched)
plot_post_process(image=image, mask_org=mask_stitched, mask_post=mask_post,
                  size_seg=size_seg, mask_size=mask_size, score=score_stitched, yn_keep=yn_keep_mask)


image_detection_in_pixel = np.concatenate((image, image), axis=1)
image_detection_in_pixel[:, image.shape[1]:, :] = utils.gen_mask_contour(mask_pred=mask_post, mask_true=None, image=image)
plt.figure()
plt.imshow(image_detection_in_pixel)
plt.show()

# mask_stitched, score_stitched, size_seg, mask_size = gen_mask_by_mask_number_iter(image=image, size_seg_ini=512, ideal_num_mask=20, flag_plot=True, flag_use_dn=True)
# plt.show()
if False:
    mask_pred = mask_stitched
    mask_true = utils.mask_2Dto3D(data_train[image_id]['mask'])
    plt.figure(figsize=(8, 8))
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(utils.gen_mask_contour(mask_pred, mask_true, image))
    plt.title('r: pred, g: groud truth')
    plt.axis('off')
    plt.subplot(2,2,3)
    utils.plot_mask2D(utils.mask_3Dto2D(mask_pred))
    plt.title('predict')
    plt.axis('off')
    plt.subplot(2,2,4)
    utils.plot_mask2D(utils.mask_3Dto2D(mask_true))
    plt.axis('off')
    plt.title('groud truth')


##
""" run throught all images """

data_detection = {}
time_str = time.strftime("%Y%m%d_%H%M%S")
DIR_DETECTION_RESULT = './detection_result'
if not(os.path.exists(DIR_DETECTION_RESULT)):
    os.mkdir(DIR_DETECTION_RESULT)
path_cur_detection_result = os.path.join(DIR_DETECTION_RESULT, time_str)
path_cur_detection_result_figs = os.path.join(path_cur_detection_result, 'figs')
os.mkdir(path_cur_detection_result)
os.mkdir(path_cur_detection_result_figs)

flag_isinteractive = plt.isinteractive()
plt.ioff()

data_to_use_type = 'test'  # or 'test'
if data_to_use_type == 'train':
    data_to_use = data_train
else:
    data_to_use = data_test


for i_image, image_id in enumerate(data_to_use):
    image = data_to_use[image_id]['image']

    print(i_image, image_id)

    tic_pred = time.time()
    # predict
    mask3D, score, size_seg, mask_size = gen_mask_by_seg_iter(image=image, size_seg_ini=512,
                                                                              flag_plot=False, flag_use_dn=True)
    tic_post = time.time()
    dur_pred = tic_post - tic_pred

    # post process
    yn_keep_mask, mask3D_pp = post_process(image, mask3D, score)

    tic_plot = time.time()
    dur_post = tic_plot - tic_post

    # plot
    plot_post_process(image=image, mask_org=mask3D, mask_post=mask3D_pp,
                      size_seg=size_seg, mask_size=mask_size, score=score, yn_keep=yn_keep_mask)
    plt.suptitle(np.mean(image))
    plt.savefig(os.path.join(path_cur_detection_result_figs, image_id))

    image_detection_in_pixel = np.concatenate((image, image), axis=1)

    # mask_true = utils.mask_2Dto3D(data_to_use[image_id]['mask'])
    image_detection_in_pixel[:, image.shape[1]:, :] = utils.gen_mask_contour(mask_pred=mask3D_pp, mask_true=None,
                                                                             image=image)
    sp.misc.imsave(os.path.join(path_cur_detection_result_figs, 'pixel_{}.png'.format(image_id)), image_detection_in_pixel)
    plt.close('all')

    tic_finish = time.time()
    dur_plot = tic_finish - tic_plot
    print('time: (predict, post_processing, plot) = {:.4f}, ={:.4f}, ={:.4f}'.format(dur_pred, dur_post, dur_plot))

    data_detection[image_id] = {}
    data_detection[image_id]['image'] = image
    data_detection[image_id]['mask3D'] = mask3D_pp
    data_detection[image_id]['size_seg_mask'] = (size_seg, mask_size)

with open(os.path.join(path_cur_detection_result, 'data_detection.pickle'), 'wb') as f:
    pickle.dump(data_detection, f)
if flag_isinteractive:
    plt.ion()
else:
    plt.ioff()



# store encoding result
mask_ImageId = []
mask_EncodedPixels = []
for image_id in data_detection:
    # mask_no_overlap = utils.mask_2Dto3D(utils.mask_3Dto2D(data_detection[image_id]['mask3D']))
    mask_no_overlap = data_detection[image_id]['mask3D']
    for i_mask in range(mask_no_overlap.shape[2]):
        mask_ImageId.append(image_id)
        mask_EncodedPixels.append(utils.rle_encoding(mask_no_overlap[:, :, i_mask]))

with open(os.path.join(path_cur_detection_result, 'test.csv'), 'w') as f:
    f.write('ImageId,EncodedPixels' + "\n")
    for image_id, rle in zip(mask_ImageId, mask_EncodedPixels):
        f.write(image_id + ',')
        f.write(" ".join([str(num) for num in rle]) + "\n")
    f.close()

# with open("test.csv") as f:
#     test = f.read()
#     for row in test:
#         print(row)
#     f.close()


##
""" performance evaluation """
path_cur_true_pred_compare = os.path.join(path_cur_detection_result, 'true_pred_fig')
os.mkdir(path_cur_true_pred_compare)
list_score = []

flag_isinteractive = plt.isinteractive()
plt.ioff()

for image_id in tqdm(data_to_use):
    image = data_to_use[image_id]['image']
    mask_true = data_to_use[image_id]['mask']
    mask_pred = utils.mask_3Dto2D(data_detection[image_id]['mask3D'])
    IOU_cur = utils.cal_prediction_IOU(mask_true, mask_pred)
    score_cur = utils.cal_score_from_IOU(IOU_cur)
    list_score.append(score_cur['all'])

    plt.figure(figsize=[8, 8])
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.fill_between(np.arange(0.5, 1.0, 0.05), score_cur['all'])
    plt.xlim([0.5, 1.0])
    plt.ylim([0, 1])
    plt.title('score={:.2f}'.format(score_cur['ave']))
    plt.subplot(2,2,3)
    utils.plot_mask2D(mask_true)
    plt.axis('off')
    plt.title('true')
    plt.subplot(2,2,4)
    utils.plot_mask2D(mask_pred)
    plt.axis('off')
    plt.title('prediction')

    plt.savefig(os.path.join(path_cur_true_pred_compare, image_id))
    plt.close('all')

if flag_isinteractive:
    plt.ion()
else:
    plt.ioff()


##
""" cluster image based on contrast """
temp = np.array([np.mean(image_dict['image'], axis=(0,1)) for image_id, image_dict in data_train.items()])
plt.hist(np.mean(temp, axis=1), bins=20)
plt.ion()
plt.show()
