# Age Estimation
PyTorch-based CNN implementation for estimating age from face images.

#### Explanation for each files
config.py is for the parameters used in the training process and testing process.
dataset.py is for generating the data we need.
demo.py is for test specific cases.
model.py is for the models I built.
test.py is for the testing.
train.py is for the training.
checkpoint folder is to save the models, you can go into it. There is an readme.txt for that folder.
data folder is to save data, you can go into it. There is an readme.txt for that folder.
Visualization_features_for_layers.py is for feature map visualization. (directly run this file, since it is a seperate function file I wrote)
visual_layers folder is to save the outputs of the Visualization_features_for_layers.py.



Here is the general explanation for how to run the code.

## Requirements

```bash
pip install -r requirements.txt
```

## Train

#### Download Dataset

I already upload my dataset on the google drive, you can download them in the following link for different split methods.
https://drive.google.com/drive/folders/1JioCcPiDriYt-hsWVW32HIkpmsaPUU2l?usp=sharing
> The APPA-REAL database contains 7,613 images with associated real and apparent age labels. The total number of apparent votes is around 250,000. On average we have around 38 votes per each image and this makes the average apparent age very stable (0.3 standard error of the mean).

There are 3 zip files in the link representing 3 split methods for the data.

#### Train Model
Train a model using the APPA-REAL dataset.
See `python train.py -h` for detailed options.

```bash
python train.py --data_dir [PATH/TO/appa-real-number] --tensorboard tf_log
```

Check training progress:

```bash
tensorboard --logdir=tf_log
```

<img src="misc/tfboard.png" width="400px">

#### Training Options
You can change training parameters including model architecture using additional arguments like this:

for example:

```bash
python train.py --data_dir [PATH/TO/appa-real-number] MODEL.ARCH Modified_Residual_Model Transfer True dataset_type real
```

MODEL.ARCH is for the different models
Transfer is for whether you use transfer learning or not
dataset_type is for different data set you want to estimate on 

All the parameters defined in [config.py] can be changed using this style, and you can check and modify them on the config.py
After you modify them, you only need to execute the following command:
```bash
python train.py --data_dir [PATH/TO/appa-real-number]
```

#### Exited model
The models I trained and tested were uploaded in the following link. You can download them direccted test them or run 
the demo.py
https://drive.google.com/drive/folders/1tqGB4QSEVjAuZO4afSG0GsOPGlBjpTqY?usp=sharing

#### Test Trained Model
Evaluate the trained model using the APPA-REAL test dataset.

```bash
python test.py --data_dir [PATH/TO/appa-real-number] --resume [PATH/TO/BEST_MODEL.pth]
```
########################################################################################################################
This work is based on the https://github.com/yu4u/age-estimation-pytorch, while I modified a lot on the original code.

I rewrite the whole model.py code for all of my 5 models, and I rewrite the train function inside of the train.py and 
the config.py, instead of the default.py, and I write the code for visualization for the feature map for each layers, 
named Visualization_features_for_layers. For the other codes, I modified some parts inside of them.

All the places I modified or rewrote, I marked it as shown below.
#####################################  KE LIANG###########################################################################
### code I wrote
########################################################################################################################


## Demo
This part is to test on specific cases. 

Webcam is required.
See `python demo.py -h` for detailed options.

```bash
python demo.py
```

Using `--img_dir` argument, images in that directory will be used as input:

```bash
python demo.py --img_dir [PATH/TO/IMAGE_DIRECTORY]
```

Further using `--output_dir` argument,
resulting images will be saved in that directory (no resulting image window is displayed in this case):

```bash
python demo.py --img_dir [PATH/TO/IMAGE_DIRECTORY] --output_dir [PATH/TO/OUTPUT_DIRECTORY]
```

