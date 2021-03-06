import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

import matplotlib.pyplot as plt

def object_id_to_affordance_label(gt_label_addr):

    gt_label = np.array(Image.open(gt_label_addr))

    object_ids = np.unique(gt_label)[1:]
    # print("\tGT object_ids:", object_ids)

    gt_aff_mask = np.zeros((gt_label.shape[0], gt_label.shape[1]), dtype=np.uint8)
    gt_mask_one = np.ones((gt_label.shape[0], gt_label.shape[1]), dtype=np.uint8)

    for object_id in object_ids:
        aff_label = map_affordance_label(object_id)
        # print("\tGT aff_label:", aff_label)

        gt_mask = gt_mask_one * aff_label
        gt_aff_mask = np.where(gt_label == object_id, gt_mask, gt_aff_mask).astype(np.uint8)

    # plt.subplot(1, 2, 1)
    # plt.title("gt_label")
    # plt.imshow(gt_label)
    # print("\tgt_label:", np.unique(gt_label)[1:])
    # plt.subplot(1, 2, 2)
    # plt.title("gt_aff_mask")
    # plt.imshow(gt_aff_mask)
    # print("\tgt_aff_mask:", np.unique(gt_aff_mask)[1:])
    # plt.show()
    # plt.ioff()

    return gt_aff_mask

def map_affordance_label(current_id):

    # 1
    grasp = [
        1, # 'mallet_1_grasp'
        3, # 'spatula_1_grasp'
        5, # 'wooden_spoon_1_grasp'
        7, # 'screwdriver_1_grasp'
        9, # 'garden_shovel_1_grasp'
    ]

    screw = [
        8, # 'screwdriver_2_screw'
    ]

    scoop = [
        6, # 'wooden_spoon_3_scoop'
        10, # 'garden_shovel_3_scoop'
    ]

    pound = [
        2, # 'mallet_4_pound'
    ]

    support = [
        4, # 'spatula_2_support'
    ]

    if current_id in grasp:
        return 1
    elif current_id in screw:
        return 2
    elif current_id in scoop:
        return 3
    elif current_id in pound:
        return 4
    elif current_id in support:
        return 5
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        exit(1)

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/arl_dataset/syn/tools/'
new_data_path = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/syn/tools/'

use_random_idx = True
NUM_IMAGES = 21225/90000

objects = [
    '1_mallet/',
    '2_spatula/',
    '3_wooden_spoon/',
    '4_screwdriver/',
    '5_garden_shovel/',
]

# 2.
scenes = [
    '1_bench/',
    '2_work_bench/',
    '3_coffee_table/',
    '4_old_table/',
    '5_bedside_table/',
    '6_dr/',
]

# 3.
splits = ['']

train_test_split = 0.95

cameras = [
    'kinect/',
    'xtion/',
    'zed/'
]

# =================== images ext ========================
image_ext30 = '.cs.png'
image_ext20 = '.depth.mm.16.png'
image_ext10 = '.png'
image_exts = [
    image_ext10,
    image_ext20,
    image_ext30,
]

# =================== new directory ========================
offset_train, offset_test = 0, 0
train_files_len, test_files_len = 0, 0
for split in splits:
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts:
                    file_path = data_path + object + scene + split + camera + '??????' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("offset: ", offset_train, offset_test)
                    print("Loaded files: ", len(files))

                    ####################
                    # reduce num files
                    ####################
                    if use_random_idx:
                        files_idx = np.random.choice(np.arange(0, len(files), 1), size=int(NUM_IMAGES*len(files)), replace=False)
                        print("Chosen Files: ", len(files_idx))
                        files = files[files_idx]

                    ###############
                    # split files
                    ###############
                    np.random.seed(0)
                    total_idx = np.arange(0, len(files), 1)
                    train_idx = np.random.choice(total_idx, size=int(train_test_split * len(total_idx)), replace=False)
                    test_idx = np.delete(total_idx, train_idx)

                    train_files = files[train_idx]
                    test_files = files[test_idx]

                    print("Chosen Train Files {}/{}".format(len(train_files), len(files)))
                    print("Chosen Test Files {}/{}".format(len(test_files), len(files)))

                    if image_ext == '.png':
                        train_files_len = len(train_files)
                        test_files_len = len(test_files)

                    ###############
                    # train
                    ###############
                    split_folder = 'train/'

                    for idx, file in enumerate(train_files):
                        old_file_name = file
                        folder_to_move = new_data_path + split_folder

                        # image_num = offset + idx
                        count = 1000000 + offset_train + idx
                        image_num = str(count)[1:]
                        # print("image_num: ", image_num)

                        if image_ext == '.cs.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_aff_mask.png'
                            aff_mask = Image.fromarray(object_id_to_affordance_label(old_file_name))
                            aff_mask.save(move_file_name)

                        elif image_ext == '.depth.mm.16.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + 'depth/' + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
                            exit(1)

                    ###############
                    # test
                    ###############
                    split_folder = 'test/'

                    for idx, file in enumerate(test_files):
                        old_file_name = file
                        folder_to_move = new_data_path + split_folder

                        # image_num = offset + idx
                        count = 1000000 + offset_test + idx
                        image_num = str(count)[1:]

                        if image_ext == '.cs.png':
                            # print("BOOM LABEL")
                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_aff_mask.png'
                            aff_mask = Image.fromarray(object_id_to_affordance_label(old_file_name))
                            aff_mask.save(move_file_name)

                        elif image_ext == '.depth.mm.16.png':
                            # print("BOOM DEPTH")
                            move_file_name = folder_to_move + 'depth/' + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '.png':
                            # print("BOOM RGB")
                            move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
                            exit(1)

                ###############
                ###############
                offset_train += train_files_len
                offset_test += test_files_len

