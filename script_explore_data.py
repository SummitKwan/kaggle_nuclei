""" script to understand data stucrure """

import os
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


""" understand the data folder structure """
# get image names
path_data = './data/stage1_train'
# path_data = './data/stage1_test'

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
                         for img_name in list_img_names if img_name[-3:] == 'png']
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



""" visualize example data """
plt.close()
id_to_show = np.random.choice(list(dict_ids.keys()))
img = ndimage.imread(dict_ids[id_to_show]['images'][0])
masks = [ndimage.imread(mask_file) for mask_file in dict_ids[id_to_show]['masks']]
h_fig, h_ax = plt.subplots(1, 2, figsize=(8, 4), sharex='all', sharey='all')
plt.axes(h_ax[0])
plt.imshow(img)
plt.axis('off')
plt.axes(h_ax[1])
plt.imshow(img)
list_mask_colors = np.random.rand(len(masks), 3)
list_mask_colors = np.append(list_mask_colors, [[0.5]]*len(masks), axis=1)
for i in range(len(masks)):
    mask = masks[i]
    mask_to_plot = (mask[:, :, None]*list_mask_colors[i][None, None, :]).astype('uint8')
    plt.imshow(mask_to_plot)
plt.axis('off')
plt.title(img.shape)




