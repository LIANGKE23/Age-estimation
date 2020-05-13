from yacs.config import CfgNode as CN

_C = CN()

#################################################### Ke LIANG ##########################################################
# Model
_C.MODEL = CN()
##### uncomment the model you want to use
#_C.MODEL.ARCH = "se_resnet50"
#_C.MODEL.ARCH = "Simple_CNN_Model"
#_C.MODEL.ARCH = "General_Residual_Model"
_C.MODEL.ARCH = "Modified_Residual_Model"
_C.MODEL.IMG_SIZE = 224

#Transfer Learning
##### only when you choose General Residual Model and Modified Residual Model in the MODEL.ARCH above, you can modify this
_C.Transfer = True  #True means "Combining the model with Transfer learning" False means opposite

#Dataset
#### apparent represents estimation on apparent age; real represents estimation on real age
_C.dataset_type = "apparent" #apparent or real
########################################################################################################################

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.EPOCHS = 80
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 64
