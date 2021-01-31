from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)

#######################################
#######################################

'''
Model Selection:
'unet'
'deeplab'
'pretrained_deeplab'
'pretrained_deeplab_multi'
'pretrained_deeplab_multi_depth'        ### concat or add
'''

MULTI_PRED = True
USE_DEPTH_IMAGES = False
SEG_MODEL = 'pretrained_deeplab_multi'

BACKBONE = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'

SEG_SAVED_WEIGHTS = ''
DIS_SAVED_WEIGHTS = ''

PREDICT_SAVED_WEIGHTS = '/home/akeaveny/catkin_ws/src/ARLGAN/checkpoints/arl/segmentation/pretrained_deeplab_multi_depth__arl_real/Best_Seg_0.93435_Fwb_20_Epochs.pth'

#######################################
#######################################

EPOCHS = 20
BATCH_SIZE = 2
NUM_IMAGES_PER_EPOCH = 5000
ITERATIONS = int(EPOCHS * NUM_IMAGES_PER_EPOCH)
PREHEAT_STEPS = int(ITERATIONS/20)

LR = 0.0001

WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9

# IMG AUG
APPLY_IMAGE_AUG = True
TAKE_CENTER_CROP = True
CROP_H = 384
CROP_W = 384

# NUM IMGs
TRAIN_ON_SUBSET = True
NUM_TRAIN = int(NUM_IMAGES_PER_EPOCH * 0.8)
NUM_VAL   = int(NUM_IMAGES_PER_EPOCH * 0.2)
NUM_TEST  = 100

#######################################
#######################################

GAN = 'Vanilla'         ### Vanilla or LS


### ADPAT SEG NET
POWER = 0.9
LAMBDA_SEG = 0.1
LAMDA_ADV1 = 0.0002
LAMDA_ADV2 = 0.001

# CLAN
# LAMDA_WEIGHT = 0.01
# LAMDA_ADV = 0.001
# LAMDA_LOCAL = 40
# EPSILON = 0.4

#######################################
# PRELIM FOR SAVED WEIGHTS
#######################################

# prelim for naming
TRAINING_MODE = 'discriminator'  ### 'segmentation' or 'discriminator'
DIS_MODEL = 'AdaptSegNet'        ### AdaptSegNet or CLAN

DATASET_NAME = 'arl'             ### UMD or ARL
DATASET_TYPE = 'real'            ### REAL OR SYN

EXPERIMENT = SEG_MODEL + '_' + DIS_MODEL + '_' + DATASET_NAME + '_' + DATASET_TYPE
CHECKPOINT_DIR_PATH = str(ROOT_DIR_PATH) + '/checkpoints/' + DATASET_NAME + '/' + TRAINING_MODE + '/' + EXPERIMENT + '/'

MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH)
BEST_MODEL_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_SEG_MODEL.pth'
BEST_DIS1_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_DIS1_MODEL.pth'
BEST_DIS2_SAVE_PATH = str(CHECKPOINT_DIR_PATH) + 'BEST_DIS2_MODEL.pth'

#######################################
# ARL DATASET PRELIM
#######################################

NUM_CHANNELS = 3     # rgb images
NUM_CLASSES = 1 + 5 # background + objects

AFFORDANCE_START = 0
AFFORDANCE_END = 5

#######################################
# ARL DATASET FOR SEGMENTATION
#######################################

### test
TEST_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/test/tools/'
RGB_IMG_EXT = '.png'
DEPTH_SUFFIX = '_depth'
DEPTH_IMG_EXT = '_depth.png'
GT_MASK_EXT = '_aff_mask.png'
GT_MASK_SUFFIX = '_aff_mask'

### real
# DATASET_DIR_PATH = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/real/tools/'
# ### TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
# RGB_IMG_EXT = '.png'
# DEPTH_SUFFIX = '_depth'
# DEPTH_IMG_EXT = '_depth.png'
# GT_MASK_EXT = '_aff_mask.png'
# GT_MASK_SUFFIX = '_aff_mask'

### syn
DATASET_DIR_PATH = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/syn/tools/'
### TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
RGB_IMG_EXT = '.png'
DEPTH_SUFFIX = '_depth'
DEPTH_IMG_EXT = '_depth.png'
GT_MASK_EXT = '_aff_mask.png'
GT_MASK_SUFFIX = '_aff_mask'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

RGB_DIR_PATH = DATASET_DIR_PATH + 'train/rgb/'
DEPTH_DIR_PATH = DATASET_DIR_PATH + 'train/depth/'
MASKS_DIR_PATH = DATASET_DIR_PATH + 'train/masks/'
VAL_PRED_DIR_PATH = DATASET_DIR_PATH + 'train/pred/'

TEST_RGB_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/rgb/'
TEST_DEPTH_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/depth/'
TEST_MASKS_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/masks/'
TEST_PRED_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/pred/'

#######################################
# ARL DATASET FOR DISCRIMINATOR
#######################################

### source
SOURCE_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/syn/tools/'
SOURCE_RGB_IMG_EXT = '.png'
SOURCE_DEPTH_SUFFIX = '_depth'
SOURCE_DEPTH_IMG_EXT = '_depth.png'
SOURCE_GT_MASK_EXT = '_aff_mask.png'
SOURCE_GT_MASK_SUFFIX = '_aff_mask'

SOURCE_RGB_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/rgb/'
SOURCE_DEPTH_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/depth/'
SOURCE_MASKS_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/masks/'

### real
TARGET_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/arl_dataset/ARLGAN/real/tools/'
TARGET_RGB_IMG_EXT = '.png'
TARGET_DEPTH_SUFFIX = '_depth'
TARGET_DEPTH_IMG_EXT = '_depth.png'
TARGET_GT_MASK_EXT = '_aff_mask.png'
TARGET_GT_MASK_SUFFIX = '_aff_mask'

GT_SAVE_MASK_EXT = '_gt.png'
PRED_MASK_EXT = '_pred.png'

TARGET_RGB_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/rgb/'
TARGET_DEPTH_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/depth/'
TARGET_MASKS_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/masks/'

TARGET_PRED_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/pred/'

# #######################################
# UMD DATASET PRELIM
# #######################################
#
# NUM_CHANNELS = 3    # rgb images
# NUM_CLASSES = 1 + 7 # background + objects
#
# AFFORDANCE_START = 0
# AFFORDANCE_END = 7
#
# #######################################
# UMD DATASET FOR SEGMENTATION
# #######################################
#
# ## syn
# DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
# TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
# RGB_IMG_EXT = '.png'
# DEPTH_SUFFIX = '_depth'
# DEPTH_IMG_EXT = '_depth.png'
# GT_MASK_EXT = '_gt_affordance.png'
# GT_MASK_SUFFIX = '_gt_affordance'
#
# ### real
# # DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
# # TEST_DATASET_DIR_PATH = DATASET_DIR_PATH
# # RGB_IMG_EXT = '.jpg'
# # DEPTH_SUFFIX = '_depth'
# # DEPTH_IMG_EXT = '_depth.png'
# # GT_MASK_EXT = '_label.png'
# # GT_MASK_SUFFIX = '_label'
#
# GT_SAVE_MASK_EXT = '_gt.png'
# PRED_MASK_EXT = '_pred.png'
#
# RGB_DIR_PATH = DATASET_DIR_PATH + 'train/rgb/'
# DEPTH_DIR_PATH = DATASET_DIR_PATH + 'train/depth/'
# MASKS_DIR_PATH = DATASET_DIR_PATH + 'train/masks/'
# VAL_PRED_DIR_PATH = DATASET_DIR_PATH + 'train/pred/'
#
# TEST_RGB_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/rgb/'
# TEST_DEPTH_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/depth/'
# TEST_MASKS_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/masks/'
# TEST_PRED_DIR_PATH = TEST_DATASET_DIR_PATH + 'test/pred/'
#
# #######################################
# UMD DATASET FOR DISCRIMINATOR
# #######################################
#
# ### source
# SOURCE_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/syn/objects/hammer/'
# SOURCE_RGB_IMG_EXT = '.png'
# SOURCE_DEPTH_SUFFIX = '_depth'
# SOURCE_DEPTH_IMG_EXT = '_depth.png'
# SOURCE_GT_MASK_EXT = '_gt_affordance.png'
# SOURCE_GT_MASK_SUFFIX = '_gt_affordance'
#
# SOURCE_RGB_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/rgb/'
# SOURCE_DEPTH_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/depth/'
# SOURCE_MASKS_DIR_PATH = SOURCE_DATASET_DIR_PATH + 'train/masks/'
#
# ### real
# TARGET_DATASET_DIR_PATH = '/data/Akeaveny/Datasets/part-affordance_combined/UNET/umd/real/objects/hammer/'
# TARGET_RGB_IMG_EXT = '.jpg'
# TARGET_DEPTH_SUFFIX = '_depth'
# TARGET_DEPTH_IMG_EXT = '_depth.png'
# TARGET_GT_MASK_EXT = '_label.png'
# TARGET_GT_MASK_SUFFIX = '_label'
#
# GT_SAVE_MASK_EXT = '_gt.png'
# PRED_MASK_EXT = '_pred.png'
#
# TARGET_RGB_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/rgb/'
# TARGET_DEPTH_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/depth/'
# TARGET_MASKS_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/masks/'
#
# TARGET_PRED_DIR_PATH = TARGET_DATASET_DIR_PATH + 'train/pred/'