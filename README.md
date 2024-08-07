# landslide-detection

The occurence of landslide events is expected to increase as a result of climate change. Our ability to predict areas prone to future landslide events relies on us having a dataset detailing the locations of past occurences. To address this, machine learning models can be trained to segment satellite imagery into landslide and non-landslide categories.

## data.py
All data is provided as part of the LandSlide4Sense 2022 competition [1]. The data class creates 6-channel input images and mask images from the .h5 files provided by the competition organisers. 
- Channels 1 to 3 comprise an RGB image acquired from Sentinel-2 satellite data (bands 2 to 4).
- Channel 4 is an NDVI image also produced using Sentinel-2 satellite data (bands 4 and 8).
- Channels 5 and 6 consist of a digital elevation model and slope data acquired from ALOS PALSAR.
  
![data](https://github.com/user-attachments/assets/09c5daba-cd62-4dbf-af1f-895445c60e9d)

Each channel is min-max normalised, then the data is split into training and validation datasets and a PyTorch DataLoader for each is provided as attributes of the class.

## unet.py
A U-Net model is used with a similar implementation to that of the original paper [2], with the exception that here the 3x3 convolutions are padded. This adjustment helps to control the size reduction of the feature maps in the contracting path of the U-Net, which is necessary considering the small size of the input images (128x128) compared to those of the original U-Net implementation (572x572).

## trainer.py
The model is trained using binary cross entropy as the loss function and Adam as the optimisation algorithm. The training and validation losses are computed each epoch to be printed and plotted for monitoring of the training progress.

## evaluator.py
The precision, recall, F1 score and intersection over union metrics computed pixel-wise over the validation dataset can then be printed for a specified model and confidence threshold. Results shown below correspond to a model trained for 25 epochs with a confidence threshold of 0.5.

<img src="https://github.com/user-attachments/assets/4702af68-558f-4d72-97e9-023cbbca793e" width="400"/>

## References

[1] O. Ghorbanzadeh et al., "The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide Detection From Multisource Satellite Imagery," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 15, pp. 9927-9942, Nov 2022.

[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention*, vol 9351, Springer, 2015, pp. 234-241.
