# Landslide Segmentation

The number of landslide events is expected to increase as a result of climate change. Successfully predicting which areas are prone to landsliding requires having a dataset that details the locations of past landslide occurrences. To address this, machine learning models can be trained to detect landslide locations from satellite imagery.

## Data
All data is provided as part of the LandSlide4Sense 2022 competition [1]. 

For each image patch there are 14 provided imagery bands:
- Multispectral data from Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.
- Slope data from ALOS PALSAR: B13.
- Digital elevation model (DEM) from ALOS PALSAR: B14.

Additional bands can also be made through combinations of the provided bands, these include:
- Normalised difference vegetation index (NDVI) which is given by $`\frac{\text{B8}-\text{B4}}{\text{B8}+\text{B4}}`$.
- Normalised difference water index (NDWI) which is given by $`\frac{\text{B3}-\text{B8}}{\text{B3}+\text{B8}}`$.
  
A binary mask image that indicates which pixels correspond to landslide locations is also provided. The choice of which bands are fed to the model can be passed as an argument to the data class. Each band is min-max normalised, and the data is divided into training (832 image), validation (64 image), and test (64 image) splits.

## Approaches

### A) Baseline Approach (IoU = 0.49)
A U-Net model is used with a similar implementation to that of the original paper [2], with the exception that here the 3x3 convolutions are padded. This adjustment reduces the size reduction of the feature maps in the contracting path of the U-Net, which is necessary considering the small size of the input images (128x128) compared to those of the original U-Net implementation (572x572). The model is trained using binary cross entropy as the loss function and the B4, B3, B2, NDVI, B13 and B14 imagery bands as features.

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
    <td>0.55</td>
    <td>0.71</td>
    <td>0.49</td>
    <td>0.65</td>
  </tr>
</table>


## References

[1] O. Ghorbanzadeh et al., "The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide Detection From Multisource Satellite Imagery," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 15, pp. 9927-9942, Nov 2022.

[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention*, vol 9351, Springer, 2015, pp. 234-241.
