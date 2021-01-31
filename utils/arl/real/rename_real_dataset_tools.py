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
data_path = '/data/Akeaveny/Datasets/arl_dataset/real/tools/'
new_data_path = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/real/tools/'

offset = 0

# =================== directories ========================
BASE = data_path
SUBFOLDER = '/images/'
images_path01 = BASE + 'arl_single_garden_shovel_1' + SUBFOLDER
images_path02 = BASE + 'arl_single_garden_shovel_2' + SUBFOLDER
images_path03 = BASE + 'arl_single_garden_shovel_3' + SUBFOLDER
images_path04 = BASE + 'arl_single_garden_shovel_4' + SUBFOLDER
images_path05 = BASE + 'arl_single_garden_shovel_5' + SUBFOLDER
images_path06 = BASE + 'arl_single_garden_shovel_6' + SUBFOLDER
images_path07 = BASE + 'arl_single_garden_shovel_7' + SUBFOLDER
images_path08 = BASE + 'arl_single_garden_shovel_8' + SUBFOLDER
#
images_path11 = BASE + 'arl_single_mallet_1' + SUBFOLDER
images_path12 = BASE + 'arl_single_mallet_2' + SUBFOLDER
images_path13 = BASE + 'arl_single_mallet_3' + SUBFOLDER
images_path14 = BASE + 'arl_single_mallet_4' + SUBFOLDER
images_path15 = BASE + 'arl_single_mallet_5' + SUBFOLDER
images_path16 = BASE + 'arl_single_mallet_6' + SUBFOLDER
images_path17 = BASE + 'arl_single_mallet_7' + SUBFOLDER
images_path18 = BASE + 'arl_single_mallet_8' + SUBFOLDER
#
images_path21 = BASE + 'arl_single_screwdriver_1' + SUBFOLDER
images_path22 = BASE + 'arl_single_screwdriver_2' + SUBFOLDER
images_path23 = BASE + 'arl_single_screwdriver_3' + SUBFOLDER
images_path24 = BASE + 'arl_single_screwdriver_4' + SUBFOLDER
images_path25 = BASE + 'arl_single_screwdriver_5' + SUBFOLDER
images_path26 = BASE + 'arl_single_screwdriver_6' + SUBFOLDER
images_path27 = BASE + 'arl_single_screwdriver_7' + SUBFOLDER
images_path28 = BASE + 'arl_single_screwdriver_8' + SUBFOLDER
#
images_path31 = BASE + 'arl_single_spatula_1' + SUBFOLDER
images_path32 = BASE + 'arl_single_spatula_2' + SUBFOLDER
images_path33 = BASE + 'arl_single_spatula_3' + SUBFOLDER
images_path34 = BASE + 'arl_single_spatula_4' + SUBFOLDER
images_path35 = BASE + 'arl_single_spatula_5' + SUBFOLDER
images_path36 = BASE + 'arl_single_spatula_6' + SUBFOLDER
images_path37 = BASE + 'arl_single_spatula_7' + SUBFOLDER
images_path38 = BASE + 'arl_single_spatula_8' + SUBFOLDER
#
images_path41 = BASE + 'arl_single_wooden_spoon_1' + SUBFOLDER
images_path42 = BASE + 'arl_single_wooden_spoon_2' + SUBFOLDER
images_path43 = BASE + 'arl_single_wooden_spoon_3' + SUBFOLDER
images_path44 = BASE + 'arl_single_wooden_spoon_4' + SUBFOLDER
images_path45 = BASE + 'arl_single_wooden_spoon_5' + SUBFOLDER
images_path46 = BASE + 'arl_single_wooden_spoon_6' + SUBFOLDER
images_path47 = BASE + 'arl_single_wooden_spoon_7' + SUBFOLDER
images_path48 = BASE + 'arl_single_wooden_spoon_8' + SUBFOLDER

objects = [images_path01, images_path02, images_path03, images_path04, images_path05, images_path06, images_path07, images_path08,
           images_path11, images_path12, images_path13, images_path14, images_path15, images_path16, images_path17, images_path18,
           images_path21, images_path22, images_path23, images_path24, images_path25, images_path26, images_path27, images_path28,
           images_path31, images_path32, images_path33, images_path34, images_path35, images_path36, images_path37, images_path38,
           images_path41, images_path42, images_path43, images_path44, images_path45, images_path46, images_path47, images_path48,
           ]

# 2.
scenes = ['']

# 3.
splits = ['']

train_test_split = 0.95

# 4.
cameras = ['']

# =================== images ext ========================
image_ext1 = '_rgb.png'
image_ext2 = '_depth.png'
image_ext3 = '_labels.png'

image_exts = [
            image_ext1,
            image_ext2,
            image_ext3
]

# =================== new directory ========================
offset_train, offset_test = 0, 0
train_files_len, test_files_len = 0, 0
for split in splits:
    offset = 0
    for object in objects:
        for scene in scenes:
            for camera in cameras:
                files_offset = 0
                for image_ext in image_exts:
                    file_path = object + scene + split + camera + '*' + image_ext
                    print("File path: ", file_path)
                    files = np.array(sorted(glob.glob(file_path)))
                    print("Loaded files: ", len(files))
                    print("offset: ", offset_train, offset_test)

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

                    if image_ext == '_rgb.png':
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

                        if image_ext == '_rgb.png':
                            move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            move_file_name = folder_to_move + 'depth/' + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_labels.png':
                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_aff_mask.png'
                            aff_mask = Image.fromarray(object_id_to_affordance_label(old_file_name))
                            aff_mask.save(move_file_name)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
                            exit(1)

                    ###############
                    # test
                    ###############
                    split_folder = 'test/'

                    for idx, file in enumerate(test_files):
                        old_file_name = file
                        folder_to_save = data_path + object + scene + split + camera
                        folder_to_move = new_data_path + split_folder + scene

                        # image_num = offset + idx
                        count = 1000000 + offset_test + idx
                        image_num = str(count)[1:]

                        if image_ext == '_rgb.png':
                            move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_depth.png':
                            move_file_name = folder_to_move + 'depth/' + np.str(image_num) + '_depth.png'
                            shutil.copyfile(old_file_name, move_file_name)

                        elif image_ext == '_labels.png':
                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
                            shutil.copyfile(old_file_name, move_file_name)

                            move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_aff_mask.png'
                            aff_mask = Image.fromarray(object_id_to_affordance_label(old_file_name))
                            aff_mask.save(move_file_name)

                        else:
                            print("*** IMAGE EXT DOESN'T EXIST ***")
                            exit(1)

                offset_train += train_files_len
                offset_test += test_files_len