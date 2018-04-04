""" script to try mask-rcnn """

import os
import sys
import random
import time
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
import utils

# make sure the Mask_RCNN package exists under '..'
MRCNN_PKG_DIR = '..'
sys.path.append(MRCNN_PKG_DIR)
import Mask_RCNN as mrcnn


import cv2


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


""" get data """

with open('./data/data_train.pickle', 'rb') as f:
    data_train = pickle.load(f)
with open('./data/data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)


""" prepare data for training """
yn_create_train_seg = False
yn_use_aug = True
yn_dark_nuclei = True

if yn_dark_nuclei:
    data_train_dark_nuclei = {}
    for id_img in data_train:
        data_train_dark_nuclei[id_img] = {}
        image_bw = np.mean(data_train[id_img]['image'], axis=2)
        if np.mean(image_bw[data_train[id_img]['mask']>0]) > np.mean(image_bw):
            data_train_dark_nuclei[id_img]['image'] = 255-data_train[id_img]['image']
        else:
            data_train_dark_nuclei[id_img]['image'] = data_train[id_img]['image']
        data_train_dark_nuclei[id_img]['mask'] = data_train[id_img]['mask']

if yn_create_train_seg:
    if yn_use_aug:
        data_train_aug = {}
        for id_img in data_train.keys():
            image_cur = data_train[id_img]['image']
            mask_cur = data_train[id_img]['mask']

            if yn_dark_nuclei:
                image_bw = np.mean(image_cur, axis=2)
                if np.mean(image_bw[mask_cur > 0]) > np.mean(image_bw):
                    image_cur = 255 - image_cur

            data_train_aug[id_img + '_o0'] = {}
            data_train_aug[id_img + '_o0']['image'] = image_cur
            data_train_aug[id_img + '_o0']['mask'] = mask_cur
            data_train_aug[id_img + '_o1'] = {}
            data_train_aug[id_img + '_o1']['image'] = np.rot90(image_cur, 1)
            data_train_aug[id_img + '_o1']['mask'] = np.rot90(mask_cur, 1)
            data_train_aug[id_img + '_o2'] = {}
            data_train_aug[id_img + '_o2']['image'] = np.rot90(image_cur, 2)
            data_train_aug[id_img + '_o2']['mask'] = np.rot90(mask_cur, 2)
            data_train_aug[id_img + '_o3'] = {}
            data_train_aug[id_img + '_o3']['image'] = np.rot90(image_cur, 3)
            data_train_aug[id_img + '_o3']['mask'] = np.rot90(mask_cur, 3)
            # data_train_aug[id_img + '_o4'] = {}
            # data_train_aug[id_img + '_o4']['image'] = np.fliplr(image_cur)
            # data_train_aug[id_img + '_o4']['mask'] = np.fliplr(mask_cur)
            # data_train_aug[id_img + '_o5'] = {}
            # data_train_aug[id_img + '_o5']['image'] = np.rot90(np.fliplr(image_cur), 1)
            # data_train_aug[id_img + '_o5']['mask'] = np.rot90(np.fliplr(mask_cur), 1)
            # data_train_aug[id_img + '_o6'] = {}
            # data_train_aug[id_img + '_o6']['image'] = np.rot90(np.fliplr(image_cur), 2)
            # data_train_aug[id_img + '_o6']['mask'] = np.rot90(np.fliplr(mask_cur), 2)
            # data_train_aug[id_img + '_o7'] = {}
            # data_train_aug[id_img + '_o7']['image'] = np.rot90(np.fliplr(image_cur), 3)
            # data_train_aug[id_img + '_o7']['mask'] = np.rot90(np.fliplr(mask_cur), 3)
        data_train_selection = data_train_aug
    else:
        data_train_selection = data_train

    data_train_seg = {}
    # split every image so that the diameter of every nuclei takes 1/16 ~ 1/8 of the image length
    list_amplification = [8, 16]
    for id_img in tqdm(data_train_selection.keys()):
        img_cur  = data_train_selection[id_img]['image']
        mask_cur = data_train_selection[id_img]['mask']
        size_nuclei = int(np.mean(np.sqrt(np.sum(utils.mask_2Dto3D(data_train_selection[id_img]['mask']), axis=(0, 1)))))
        for amplification in list_amplification:
            img_cur_seg  = utils.img_split(img=img_cur,  size_seg=size_nuclei * amplification)
            mask_cur_seg = utils.img_split(img=mask_cur, size_seg=size_nuclei * amplification)
            for start_loc in img_cur_seg.keys():
                data_train_seg[(id_img, start_loc, amplification)] = {}
                data_train_seg[(id_img, start_loc, amplification)]['image'] = img_cur_seg[start_loc]
                data_train_seg[(id_img, start_loc, amplification)]['mask'] = mask_cur_seg[start_loc]

    if yn_dark_nuclei:
        with open('./data/data_train_dn_seg.pickle', 'wb') as f:
            pickle.dump(data_train_seg, f)
    elif yn_use_aug:
        with open('./data/data_train_aug_seg.pickle', 'wb') as f:
            pickle.dump(data_train_seg, f)
    else:
        with open('./data/data_train_seg.pickle', 'wb') as f:
            pickle.dump(data_train_seg, f)

else:
    if yn_dark_nuclei:
        with open('./data/data_train_dn_seg.pickle', 'rb') as f:
            data_train_seg = pickle.load(f)
    elif yn_use_aug:
        with open('./data/data_train_aug_seg.pickle', 'rb') as f:
            data_train_seg = pickle.load(f)
    else:
        with open('./data/data_train_seg.pickle', 'rb') as f:
            data_train_seg = pickle.load(f)


""" create mrcnn class for using the model """

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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 foreground

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 128

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 5000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = NucleiConfig()
config.display()


class NucleiDataset(mrcnn.utils.Dataset):

    def add_data_dict(self, data_dict):
        self.data_dict = data_dict
        self.add_class("nuclei", 1, "regular")
        for i, key in enumerate(data_dict):
            self.add_image('nuclei', i, key)

    def load_image(self, image_id):
        return self.data_dict[self.image_info[image_id]['path']]['image']

    def load_mask(self, image_id):
        mask_2D = self.data_dict[self.image_info[image_id]['path']]['mask']
        mask_3D = utils.mask_2Dto3D(mask_2D)

        class_ids = np.ones(mask_3D.shape[2], dtype=np.int32)
        mask = mask_3D

        return mask, class_ids


""" prepare for full images """
if False:
    proportion_val = 0.2
    keys_data = list(data_train.keys())
    keys_data_shuffle =  random.sample(keys_data, k=len(keys_data))

    num_train = int((1-proportion_val)*len(keys_data))
    keys_train = keys_data_shuffle[:num_train]
    keys_val   = keys_data_shuffle[num_train:]


    dataset_train_all = NucleiDataset()
    dataset_train_all.add_data_dict(data_train)
    dataset_train_all.prepare()

    dataset_train = NucleiDataset()
    dataset_train.add_data_dict({key:data_train[key] for key in keys_train})
    dataset_train.prepare()

    dataset_val = NucleiDataset()
    dataset_val.add_data_dict({key:data_train[key] for key in keys_val})
    dataset_val.prepare()

    for id_img in data_test.keys():
        data_test[id_img]['mask'] = np.zeros(data_test[id_img]['image'].shape[:2] + (1,))
    dataset_test = NucleiDataset()
    dataset_test.add_data_dict(data_test)
    dataset_test.prepare()


""" prepare for splitted image """
proportion_val = 0.2
keys_data = list(data_train_seg.keys())
keys_data_shuffle =  random.sample(keys_data, k=len(keys_data))

num_train = int((1-proportion_val)*len(keys_data))
keys_train = keys_data_shuffle[:num_train]
keys_val   = keys_data_shuffle[num_train:]


dataset_train_seg_all = NucleiDataset()
dataset_train_seg_all.add_data_dict(data_train_seg)
dataset_train_seg_all.prepare()

dataset_train_seg = NucleiDataset()
dataset_train_seg.add_data_dict({key:data_train_seg[key] for key in keys_train})
dataset_train_seg.prepare()

dataset_val_seg = NucleiDataset()
dataset_val_seg.add_data_dict({key:data_train_seg[key] for key in keys_val})
dataset_val_seg.prepare()


# fig, ax = plt.subplots(1,2)
# i = 9
# plt.subplot(ax[0])
# plt.imshow(dataset_train.load_image(i))
# plt.subplot(ax[1])
# plt.imshow(dataset_train.load_mask(i)[0][:,:,0])


model = mrcnn.model.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
init_with = "coco"  # imagenet, coco, or last
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


# ## Training


model.train(dataset_train_seg, dataset_val_seg,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')

model.train(dataset_train_seg, dataset_val_seg,
            learning_rate=config.LEARNING_RATE,
            epochs=6,
            layers='4+')

model.train(dataset_train_seg, dataset_val_seg,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=6,
            layers="all")

## Validate
""" validate """

class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

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
#                           'nuclei20180401T2125_seg_ampli_12_24_48_50000_instance_per_epoch',
#                           "mask_rcnn_nuclei_0003.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


## plot a random detection result
""" plot a random detection result """

image_id = random.choice(list(data_test.keys()))

# image_id = '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466'
image = data_test[image_id]['image']


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


def post_process(image, mask3D, score=None):
    """ filter out outliers: score>0.8, size and luminance within 3 std """

    n = mask3D.shape[2]
    if n > 3:
        if score is None:
            score = np.ones(n)
        mask_size = np.sqrt(np.sum(mask3D, axis=(0, 1)))
        image_bw = np.mean(image, axis=2)
        mask_lumi = np.array([np.mean(np.average(image_bw, weights=mask3D[:, :, i])) for i in range(mask3D.shape[2])])

        size_mean, size_std = mean_std_weighted(mask_size, score**2)
        lumi_mean, lumi_std = mean_std_weighted(mask_lumi, score**2)

        z_size = (mask_size - size_mean) / size_std
        z_lumi = (mask_lumi - lumi_mean) / lumi_std

        yn_keep = (score > 0.8) & ((z_size**2 + z_lumi**2) < 3**2)
        return yn_keep
    else:
        return np.ones(n, dtype='bool')


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

# mask3D = np.zeros([60,60, 4], dtype='uint8')
# mask3D[0:20, 0:20, 0]=1
# mask3D[10:30, 10:30, 1]=1
# mask3D[5:40, 5:40, 2]=1
# mask3D[50:60, 50:60, 3]=1
# mask3D_no_overlap = remove_overlap(mask3D, [1,2,0,3])
# for i in range(4):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(mask3D[:,:,i])
#     plt.subplot(2, 4, i + 5)
#     plt.imshow(mask3D_no_overlap[:, :, i])


def gen_mask_by_seg(image=image, size_seg=256, flag_use_dn=False):

    # flip sign of the dark field images
    if flag_use_dn:
        image = image_dn(image)

    image_segs = utils.img_split(image, size_seg=size_seg, overlap=0.2)

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

    # post processing
    if False:  # filter out mask that is not darker than average
        pass

    mask_size = np.average(np.sqrt(np.sum(mask_stitched, axis=(0, 1))), weights=score_stiched)
    return mask_stitched, score_stiched, mask_size


def plot_detection_result(image, mask3D, size_seg=0, mask_size=0, score=None):

    num_row = 1 if score is None else 2
    fig_height = 4 if score is None else 8
    h_fig = plt.figure(figsize=(fig_height, 8))

    plt.subplot(num_row, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('{}, size_seg={}'.format(image.shape[:2], size_seg))
    plt.subplot(num_row, 2, 2)
    utils.plot_mask2D(utils.mask_3Dto2D(mask3D))
    plt.axis('off')
    plt.title('mask_size={:.2f}'.format(mask_size))
    if score is not None:
        plt.subplot(2, 2, 3)
        yn_keep = post_process(image, mask3D, score)
        plt.imshow(np.any(mask3D, axis=2).astype('int') + np.any(mask3D[:, :, yn_keep==False], axis=2), vmin=0, vmax=2)
        plt.axis('off')
        plt.title('filtered out {}'.format(np.sum(yn_keep==False)))

        try:
            plt.subplot(6, 2, 8)
            plt.hist(score, bins=np.arange(0.5, 1.01, 0.02))
            plt.ylabel('score')
            plt.subplot(6, 2, 10)
            mask_size = np.sqrt(np.sum(mask3D, axis=(0, 1)))
            plt.hist(mask_size, bins=np.linspace(0, mask_size.max(), 20))
            plt.ylabel('size')
            plt.subplot(6, 2, 12)
            image_bw = np.mean(image, axis=2)
            mask_lum = np.array([np.mean(np.average(image_bw, weights=mask3D[:, :, i])) for i in range(mask3D.shape[2])])
            plt.hist(mask_lum, bins=20)
            plt.ylabel('luminance')
        except:
            pass
    return h_fig


def gen_mask_by_seg_iter(image=image, size_seg_ini=512, flag_plot=False, flag_use_dn=False):

    amplification_min = 8    # 10
    amplification_max = 12    # 14
    amplification_best = 10   # 12

    mask_stitched = None
    mask_size = 50
    size_seg = size_seg_ini
    for i in range(5):
        mask_stitched, score_stitched, mask_size = gen_mask_by_seg(image, size_seg, flag_use_dn=flag_use_dn)
        if flag_plot:
            plot_detection_result(image, mask_stitched, size_seg, mask_size, score_stitched)
        if not(np.isfinite(mask_size)) or mask_size*amplification_min <= size_seg < mask_size*amplification_max:
            break
        else:
            size_seg = int(mask_size * amplification_best)

    # remove overlap:
    mask_stitched = remove_overlap(mask_stitched, -score_stitched)
    # remove empty mask:
    yn_keep = np.sum(mask_stitched, axis=(0, 1)) > 0
    mask_stitched = mask_stitched[:, :, yn_keep]
    score_stitched = score_stitched[yn_keep]

    if flag_plot:
        plot_detection_result(image, mask_stitched, size_seg, mask_size, score_stitched)

    return mask_stitched, score_stitched, size_seg, mask_size

mask_stitched, score_stitched, size_seg, mask_size = gen_mask_by_seg_iter(image=image, size_seg_ini=256, flag_plot=True, flag_use_dn=True)
plt.show()

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

data_to_use_type = 'train'  # or 'test'
if data_to_use_type == 'train':
    data_to_use = data_train
else:
    data_to_use = data_test


for image_id in tqdm(data_to_use):
    image = data_to_use[image_id]['image']
    mask3D, score, size_seg, mask_size = gen_mask_by_seg_iter(image=image, flag_plot=False, flag_use_dn=True)

    # plot
    h_fig = plot_detection_result(image, mask3D, size_seg, mask_size, score)
    plt.suptitle(np.mean(image))
    plt.savefig(os.path.join(path_cur_detection_result_figs, image_id))
    plt.close('all')

    # post process
    yn_keep_mask = np.ones(mask3D.shape[2], dtype='bool')
    if True:
        yn_keep_mask = post_process(image, mask3D)
    mask3D_pp = mask3D[:, :, yn_keep_mask]


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
    mask_no_overlap = utils.mask_2Dto3D(utils.mask_3Dto2D(data_detection[image_id]['mask3D']))
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
