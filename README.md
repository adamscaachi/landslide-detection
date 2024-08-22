# Landslide Segmentation

The number of landslide events is expected to increase as a result of climate change. Successfully predicting which areas are prone to landsliding requires having a dataset that details the locations of past landslide occurrences. To address this, machine learning models can be trained to detect landslide locations from satellite imagery.

## Data
All data is provided as part of the LandSlide4Sense 2022 competition. 

For each image patch there are 14 provided imagery bands:
- Multispectral data from Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.
- Slope data from ALOS PALSAR: B13.
- Digital elevation model (DEM) from ALOS PALSAR: B14.

Additional bands can also be made through combinations of the provided bands, these include:
- Normalised difference vegetation index (NDVI), given by $`\frac{\text{B8}-\text{B4}}{\text{B8}+\text{B4}}`$.
- Normalised difference water index (NDWI), given by $`\frac{\text{B3}-\text{B8}}{\text{B3}+\text{B8}}`$.
  
A binary mask image that indicates which pixels correspond to landslide locations is also provided. The choice of which bands are fed to the model can be passed as an argument to the data class. Each band is min-max normalised, and the data is divided into training (832 image), validation (64 image), and test (64 image) splits.

## Model

A U-Net model is used with a similar implementation to that of the original paper, with the exception that here the 3x3 convolutions are padded. This adjustment reduces the size reduction of the feature maps in the contracting path of the U-Net, which is necessary considering the small size of the input images (128x128) compared to those of the original U-Net implementation (572x572). 

A data augmentation strategy is implemented whereby each image is augmentated as it is sent to the network, thereby allowing the model to be trained on different variations of the training data each epoch. Augmentations are selected and applied randomly from the following list: flip, rotation, contrast adjustment, brightness adjustment, and dropout. 

The model is trained using binary cross entropy as the loss function and the B2, B3, B4, B13, B14, and NDVI imagery bands as features. The confidence threshold that maximises the pixel-wise IoU on the validation dataset is found and used to calculate the pixel-wise IoU and F1 score on the validation and test datasets.

<div align="center">
<table>
  <tr>
    <th colspan="2">Validation</th>
    <th colspan="2">Test</th>
  </tr>
  <tr>
    <th>IoU</th>
    <th>F1</th>
    <th>IoU</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>0.56</td>
    <td>0.72</td>
    <td>0.52</td>
    <td>0.68</td>
  </tr>
</table>
</div>
