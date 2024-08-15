# Landslide Segmentation

The number of landslide events is expected to increase as a result of climate change. Successfully predicting which areas are prone to landsliding requires having a dataset that details the locations of past landslide occurrences. To address this, machine learning models can be trained to detect landslide locations from satellite imagery.

## Data
All data is provided as part of the LandSlide4Sense 2022 competition [1]. 

6-channel input images are used as the features:
- Channels 1 to 3 comprise an RGB image acquired from Sentinel-2 satellite data (bands 2 to 4).
- Channel 4 is an NDVI image also produced using Sentinel-2 satellite data (bands 4 and 8).
- Channels 5 and 6 consist of a digital elevation model and slope data acquired from ALOS PALSAR.
  
A binary mask image that indicates which pixels correspond to landslide locations is used as the label. Each input channel is min-max normalised, then the data is divided into training (832 image), validation (64 image), and test (64 image) splits and a PyTorch DataLoader for each split is created. 

An example of the input channels and corresponding label is shown below.
  
![data](https://github.com/user-attachments/assets/09c5daba-cd62-4dbf-af1f-895445c60e9d)

## Approaches

### A) Baseline Approach
A U-Net model is used with a similar implementation to that of the original paper [2], with the exception that here the 3x3 convolutions are padded. This adjustment reduces the size reduction of the feature maps in the contracting path of the U-Net, which is necessary considering the small size of the input images (128x128) compared to those of the original U-Net implementation (572x572). The model is trained using binary cross entropy as the loss function and Adam as the optimisation algorithm. 

### B) Baseline Approach with Data Augmentation
Coming soon.

## Results

The confidence threshold that maximises the pixel-wise IoU on the validation dataset is found and used to calculate the pixel-wise IoU and F1 score on the validation and test datasets.

<table>
  <tr>
    <th></th>
    <th colspan="2">Validation</th>
    <th colspan="2">Test</th>
  </tr>
  <tr>
    <th></th>
    <th>IoU</th>
    <th>F1</th>
    <th>IoU</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>A</td>
    <td>0.54</td>
    <td>0.70</td>
    <td>0.46</td>
    <td>0.63</td>
  </tr>
</table>


## References

[1] O. Ghorbanzadeh et al., "The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide Detection From Multisource Satellite Imagery," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 15, pp. 9927-9942, Nov 2022.

[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention*, vol 9351, Springer, 2015, pp. 234-241.
