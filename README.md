# Remote Sensing Hyperspectral Image validation
Validation module for various Hyperspectral Remote Sensing Image quality aspects: radiometric sensivities, radiometric offset, optical modulation transfer function, signal to noise ratio, ground sampling distance, quantitative dynamic range

Validations were done using images of desired location on earth for the corresponding aspect:

* Radiometric offsets: image of Atlantic or Pacific ocean at night
* Radiometric sensivity (or radiometric gain): image of homogeneous areas such as Libya desert, Algeria desert, taken at descending phase of the orbit
* Optical Modulation Transfer Function (MTF): image of checkerboard target such as Salon de Provence, France
* Signal to Noise Ratio: image of homogeneous areas
* Geometric resolution (Ground Sampling Distance): image of any scene on earth with at least two control points whose coordinate (latitude and longitude) is known.
* Dynamic Range: image of areas on earth with highest/lowest relative reflectance or absolute radiance for saturation assessment

### 1. Radiometric offsets:
Radiometric offsets is actually the dark signal/dark noise provided by the imaging sensor. Validation of Radiometric offsets is done using image taken at Atlantic or Pacific ocean at night

To use the validation tools as Webservices with Swagger UI:
```
cd DS_Validation/
python START_DS_VALIDATION.py
```

To compute dark signal of image

```
python ds_validation.py folder_path
```

`folder_path` is the absolute path of the folder contains images of ocean scenes taken at night

To compare two dark signal files

```
python compare_ds.py ds_path1 ds_path2
```

To perform Image correction with computed dark signal

```
python image_ds_correction.py img_path ds_path channel_name
```

There are 3 parameters:
* `img_path`: path to the image data file, which is a HDF5 array
* `ds_path`: path to the computed dark signal file
* `channel_name`: name of the spectral band

### 2. Radiometric sensivities:
Radiometric sensivities is actually the gain of each pixel in the imaging sensor. Gain coeffient is the multiply of absolute gain and relative gains. While absolute gains are remain fix for the entire life of the RS satelite, the relative gains can be validate and corrected. Validation of Radiometric sensivity is done using image taken at homogeneous areas such as the desert

To use the validation tools as Webservices with Swagger UI:
```
cd PRNU_Validation/
python START_PRNU_VALIDATION.py
```

To compute sensivities of image

```
python prnu_validation.py folder_path ds_path cpf_path
```

* `folder_path` is the absolute path of the folder contains image of the desert. 
* `ds_path` is the absolute path to the calculated dark signal file from section 1
* `cpf_path` is the path to the calibration file of the imaging instrument

To compare two sensivity files

```
python compare_prnu.py prnu_path1 prnu_path2
```

To perform Image correction with computed sensivities

```
python image_prnmu_correction.py img_path ds_path prnu_path channel_name
```

There are 4 parameters:
* `img_path`: path to the image data file, which is a HDF5 array
* `ds_path`: path to the computed dark signal file
* `prnu_path`: path to the computed sensivity file
* `channel_name`: name of the spectral band

### 3. Optical Modulation Transfer Function
Optical Modulation Transfer Function show how sharp an image. The validation of the MTF require image taken at specific location on-earth with chessboard target. One of the popular calibration site is the Salon de Provence RS calibration site in France

To use the validation tools as Webservices with Swagger UI:
```
cd MTF_Validation/
python START_MTF_VALIDATION.py
```

To compute the along-track MTF
```
python mtf_validation_async_normalized.py img_path True
```

To compute the across-track MTF
```
python mtf_validation_async_normalized.py img_path False
```

`img_path` is the absolute path to the 10bits TIFF image after Level 1 processing. The Validation of MTF require selection of a sub-area in the image where modulation (brighness) transition occur in either vertical (North-South) or horizontal (West-East) axis.

### 4. Signal to Noise Ratio
SNR show how "clean" an image. Since there is theoretically impossible for acquiring pure "signal" image with no noise in orbit, the SNR is calculated specific to an image in a specific condition. The signal power is calculated as the mean of digital number value over a homogeneous region, with the noise power is calculated as the standard deviation of the noise within the same region.

To use the validation tools as Webservices with Swagger UI:
```
cd SNR_Validation/
python START_SNR_VALIDATION.py
```

To compute the SNR:
```
python snr_validation_async.py img_path rotate_angle
```

* `img_path` is the absolute path to the 10bits TIFF image after Level 1 processing. 
* `rotate_angle` is the angle to rotate the image counter clockwise. This is to increase the "selectable" area of the chessboard target, by align the edge of the chessboard to the North-South and East-West axis

### 5. Ground Sampling Distance
Ground Sampling distance is actually the resolution of the Image. Resolution are validated along track and across track, using at least two control points on Earth with known latitude and longitude.

To use the validation tools as Webservices with Swagger UI:
```
cd GSD_Validation/
python START_GSD_VALIDATION.py
```

To compute the GSD:
```
python gsd_validation_async.py img_path lat1 long1 lat2 long2
```

* `img_path` is the absolute path to the 10bits TIFF image after Level 1 processing. 
* `lat1` : latitude of control point 1
* `long1` : longitude of control point 1
* `lat2` : latitude of control point 2
* `long2` : longitude of control point 2

