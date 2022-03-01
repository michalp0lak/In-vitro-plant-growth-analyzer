# In-vitro-plant-growth-analyzer

## Description
The software is designed to evaluate batch of rgb images generated with standardized protocol of Arabidopsis in-vitro growing experiment.
Software is coded in Python3 and mainly uses OpenCV, scipy, scikit-image libraries.

Single image contains well-plate of given well-grid (growing areas) structure. This is region of interest (ROI) for further analysis.

The goal is to evaluate spatial and color pattern of plant in each well (growing area).

Software suppose various image formats jpg, png, bmp, tiff, tif as an input data.

## Processing pipeline
![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/raw.png?raw=true)

Process of single image analysis follows these steps:

1. ROI is cropped from raw image

![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/roi.png?raw=true)

2. Reading barcode data

![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/barcode.png?raw=true)

3. Well plate is separated into growing areas according to suitable well plate mask

![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/mask.png?raw=true)

4. Plant segmentation in growing area

![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/grow_area.png?raw=true)
![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/plant_mask.png?raw=true)
![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/segmented_plant.png?raw=true)

5. All plant and growing areas contours are painted in raw image. For each well, plant area and color indexes are evaluated.
 
![alt text](https://github.com/PolakMichalMLT/In-vitro-plant-growth-analyzer/blob/master/readme_images/final.png?raw=true)

## Output of analysis

The output of analysis is in **results** folder containing raw images with painted contours, text file **failures** with name of images whose analysis failed, **analysis_config.json**  file with **well_grid_analyzer.py** configuration and **xlsx** file with structured analysis output described below.
  - **filename** - name of raw image file
  - **date** - measurement timestamp
  - **location** - camera position
  - **x_coordinate** - camera position x-coordinate
  - **y_coordinate** - camera position y-coordinate
  - **barcode_data** - decoded barcode information
  - **well_row** - well grid row index
  - **well_column** - well grid column index
  - **pixel_num** - segmented pixel-area of plant in well
  - **r_mean** - mean of red-channel of plant area
  - **g_mean** - mean of green-channel of plant area
  - **b_mean** - mean of blue-channel of plant area
  
  ## User Guide
  
  1. From **master** branch download **well_experiments** repository with  green download button in top-right corner of git page layout. Download repository as a zip file.
  2. Unzip repository folder to arbitrary path in your computer.
  3. At your local computer in unziped repository you can find **global_variables.py** file, inthis file modify:
   - path = 'path to folder with data'
   - masks_path = 'this folder is in unziped repository folder (folder name is masks), you have to modify it according to path on your local machine'
   - col_num, row_num - With nuber of rows and columns you specify, which type of well grid is used for experiment. Grid size has to be standardized. It can't be choosed arbitrary.
  4. Start Anaconda prompt terminal
  5. Navigate python interpreter into repository folder = In Anaconda prompt terminal type **cd path/of/unziped/repository/**
  6. Install software dependencies = In Anaconda prompt terminal type **pip install -r requirements.txt**
  7. If you exported data from PSI storage system you need form images into proper format for analysis = In Anaconda prompt terminal type **python create_batch.py**. After **create_batch.py** execution you should find new folder **batch** in your data folder. Here you can find batch of images, which is used for analysis.
  8. Analyse your data = In Anaconda prompt terminal type **python exec.py** and wait for your results.
