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

if yn_create_train_seg:
    data_train_seg = {}
    # split every image so that the diameter of every nuclei takes 1/16 ~ 1/8 of the image length
    list_amplification = [8, 16]
    # id_img = np.random.choice(list(data_train.keys()))
    for id_img in data_train.keys():
        img_cur  = data_train[id_img]['image']
        mask_cur = data_train[id_img]['mask']
        size_nuclei = int(np.mean(np.sqrt(np.sum(utils.mask_2Dto3D(data_train[id_img]['mask']), axis=(0, 1)))))
        for amplification in list_amplification:
            img_cur_seg  = utils.img_split(img=img_cur,  size_seg=size_nuclei * amplification)
            mask_cur_seg = utils.img_split(img=mask_cur, size_seg=size_nuclei * amplification)
            for start_loc in img_cur_seg.keys():
                data_train_seg[(id_img, start_loc, amplification)] = {}
                data_train_seg[(id_img, start_loc, amplification)]['image'] = img_cur_seg[start_loc]
                data_train_seg[(id_img, start_loc, amplification)]['mask'] = mask_cur_seg[start_loc]

    # id_plot = random.choice(list(data_train_seg.keys()))
    # print(id_plot)
    # utils.plot_img_and_mask_from_dict(data_train_seg, id_plot)

    with open('./data/data_train_seg.pickle', 'wb') as f:
        pickle.dump(data_train_seg, f)
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
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

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


class NucleiDataset(mrcnn.utils.Dataset):

    def add_data_dict(self, data_dict):
        self.data_dict = data_dict
        self.add_class("nuclei", 1, "regular")
        for i, key in enumerate(data_dict):
            self.add_image('nuclei', i, key)

    def load_image(self, image_id):
        return self.data_dict[self.image_info[image_id]['path']]['image']
        # old code: resize
        # img = self.data_dict[self.image_info[image_id]['path']]['image']
        # old_size = img.shape[:2]  # old_size is in (height, width) format
        # ratio = float(config.IMAGE_MAX_DIM) / max(old_size)
        # new_size = tuple([int(x * ratio) for x in old_size])
        # im = cv2.resize(img, (new_size[1], new_size[0]))
        # delta_w = config.IMAGE_MAX_DIM - new_size[1]
        # delta_h = config.IMAGE_MAX_DIM - new_size[0]
        # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        # left, right = delta_w // 2, delta_w - (delta_w // 2)
        #
        # color = [0, 0, 0]
        # new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                             value=color)
        # return new_img

    def load_mask(self, image_id):
        mask_2D = self.data_dict[self.image_info[image_id]['path']]['mask']

        mask_3D = utils.mask_2Dto3D(mask_2D)
        # label_unique = np.unique(mask_2D)
        # label_unique = label_unique[label_unique > 0]
        # if len(label_unique) > 0:
        #     mask_3D = np.stack([mask_2D==label for label in label_unique], axis=2).astype('uint8')
        #
        # else:
        #     mask_3D = np.zeros(shape=mask_2D.shape[:2]+(0, ))

        class_ids = np.ones(mask_3D.shape[2], dtype=np.int32)
        mask = mask_3D

        # old code to resize mask
        # old_size = mask_3D.shape[:2]  # old_size is in (height, width) format
        # ratio = float(config.IMAGE_MAX_DIM) / max(old_size)
        # new_size = tuple([int(x * ratio) for x in old_size])
        # im = cv2.resize(mask_3D, (new_size[1], new_size[0]))
        # delta_w = config.IMAGE_MAX_DIM - new_size[1]
        # delta_h = config.IMAGE_MAX_DIM - new_size[0]
        # top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        # left, right = delta_w // 2, delta_w - (delta_w // 2)
        #
        # color = [0]
        # mask = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                             value=color)
        # mask = (mask>0.5).astype('uint8')
        # # mask = (np.sum(mask, axis=2) > 0.5).astype('uint8')
        # if len(mask.shape)==2:
        #     mask = mask[:,:,None]
        # # class_ids = np.array([class_ids[0]]).astype('int32')
        return mask, class_ids


""" prepare for full images """
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
init_with = "last"  # imagenet, coco, or last
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
            epochs=4,
            layers='heads')

model.train(dataset_train_seg, dataset_val_seg,
            learning_rate=config.LEARNING_RATE,
            epochs=12,
            layers='4+')

model.train(dataset_train_seg, dataset_val_seg,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=12,
            layers="all")

# ## Validate

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

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

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


## plot a random detection result
""" plot a random detection result """

image_id = np.random.randint(len(data_test))
image_name = dataset_test.source_image_link(image_id)

image = data_test[image_name]['image']


def gen_mask_by_seg(image=image, size_seg=256):

    image_segs = utils.img_split(image, size_seg=size_seg, overlap=0.2)

    mask_segs = {}
    for loc_seg, image_seg in image_segs.items():
        image_detect_res = model.detect([image_seg], verbose=0)[0]
        masks_seg = image_detect_res['masks']
        if masks_seg.shape[:2] != image_seg.shape[:2]:
            masks_seg = np.zeros(shape=image_seg.shape[:2]+(0, ), dtype='int')
        mask_segs[loc_seg] = masks_seg

    mask_stitched = utils.img_stitch(mask_segs, mode='mask')[0]

    mask_size = np.mean(np.sqrt(np.sum(mask_stitched, axis=(0, 1))))

    return mask_stitched, mask_size


def plot_detection_result(image, mask3D, size_seg=0, mask_size=0):
    h_fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(size_seg)
    plt.subplot(1, 2, 2)
    utils.plot_mask2D(utils.mask_3Dto2D(mask3D))
    plt.title(mask_size)
    return h_fig

def gen_mask_by_seg_iter(image=image, size_seg_ini=512, flag_plot=False):

    size_seg = size_seg_ini
    for i in range(5):
        mask_stitched, mask_size = gen_mask_by_seg(image, size_seg)

        if flag_plot:
            plot_detection_result(image, mask_stitched, size_seg, mask_size)
        if mask_size*10 <= size_seg < mask_size*14:
            break
        else:
            size_seg = int(mask_size * 12)
    return mask_stitched, size_seg, mask_size

gen_mask_by_seg_iter(image=image, flag_plot=True)

##
""" run throught all images """

data_detection = {}
time_str = time.strftime("%Y%m%d_%H%M%S")
DIR_DETECTION_RESULT = './detection_result'
if not(os.path.exists(DIR_DETECTION_RESULT)):
    os.mkdir(DIR_DETECTION_RESULT)
path_cur_detection_result = os.path.join(DIR_DETECTION_RESULT, time_str)
os.mkdir(path_cur_detection_result)

flag_isinteractive = plt.isinteractive()
plt.ioff()
for image_id in tqdm(data_test):
    image = data_test[image_id]['image']
    mask3D, size_seg, mask_size = gen_mask_by_seg_iter(image=image, flag_plot=False)
    data_detection[image_id] = {}
    data_detection[image_id]['image'] = image
    data_detection[image_id]['mask3D'] = mask3D
    data_detection[image_id]['size_seg_mask'] = (size_seg, mask_size)

    h_fig = plot_detection_result(image, mask3D, size_seg, mask_size)
    plt.savefig(os.path.join(path_cur_detection_result, image_id))
    plt.close('all')
with open(os.path.join(path_cur_detection_result, 'data_detection.pickle'), 'wb') as f:
    pickle.dump(data_detection, f)
if flag_isinteractive:
    plt.ion()
else:
    plt.ioff()


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

with open("test.csv") as f:
    test = f.read()
    for row in test:
        print(row)
    f.close()




""" below is legacy code """

##  Test on a random validation image
image_id = np.random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = mrcnn.model.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

mrcnn.model.log("original_image", original_image)
mrcnn.model.log("image_meta", image_meta)
mrcnn.model.log("gt_class_id", gt_class_id)
mrcnn.model.log("gt_bbox", gt_bbox)
mrcnn.model.log("gt_mask", gt_mask)

mrcnn.visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                  ['', ''], figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
mrcnn.visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            ['', ''], ax=get_ax())
##


##  Test on a random test image
image_id = np.random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = mrcnn.model.load_image_gt(dataset_test, inference_config,
                           image_id, use_mini_mask=False)



mrcnn.model.log("original_image", original_image)
mrcnn.model.log("image_meta", image_meta)
# mrcnn.model.log("gt_class_id", gt_class_id)
# mrcnn.model.log("gt_bbox", gt_bbox)
# mrcnn.model.log("gt_mask", gt_mask)

mrcnn.visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                  [''], figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
mrcnn.visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            ['', ''], ax=get_ax())
##



## Evaluation

# Compute VOC-Style mAP @ IoU=0.5
image_ids = np.random.choice(dataset_val.image_ids, 100)
APs = []
P_Kaggles = []
iou_thresholds = np.arange(0.5,0.95,0.05)
for i in iou_thresholds:
    APs_i = []
    P_Kaggle_i = []
    print('threshold: {}'.format(i))
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = mrcnn.model.load_image_gt(dataset_val, inference_config,
                                                                                  image_id, use_mini_mask=False)
        molded_images = np.expand_dims(mrcnn.model.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precision_kaggle, precisions, recalls, overlaps = mrcnn.utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                             iou_threshold = i)
        APs_i.append(AP)
        P_Kaggle_i.append(precision_kaggle)
    APs.append(APs_i)
    P_Kaggles.append(P_Kaggle_i)

print("mAP: ", np.mean(np.array(APs),axis=1))
print("Precision Kaggle: ", np.mean(np.array(P_Kaggles),axis=1))


##


test_img_keys = list(data_test.keys())
images = [data_test[key]['image'] for key in test_img_keys]
results = []
for image in images:
    results.append(model.detect([image], verbose=1))

mask_ImageId = []
mask_EncodedPixels = []
for i in range(len(results)):
    masks_i = results[i][0]['masks']
    for j in range(results[i][0]['masks'].shape[2]):
        sum_masks_i = np.sum(masks_i, axis=2)
        mask = masks_i[:,:,j]
        mask[sum_masks_i>1] = 0
        masks_i[:,:,j] = mask
        if np.sum(mask)>1:
            mask_ImageId.append(test_img_keys[i])
            recoded_mask = utils.rle_encoding(mask)
            mask_EncodedPixels.append(recoded_mask)
masks = {'ImageId':mask_ImageId,'EncodedPixels':mask_EncodedPixels}

with open("test.csv","w") as f:
    f.write(",".join(masks.keys()) + "\n")
    for i in range(len(masks['ImageId'])):
        f.write(masks['ImageId'][i] + ',')
        f.write(" ".join([str(n) for n in masks['EncodedPixels'][i]]) + "\n")
    f.close()

with open("test.csv") as f:
    test = f.read()
    for row in test:
        print(row)
    f.close()




