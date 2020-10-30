import json
import pandas as pd
import numpy as np
import segmentation as sg
import cv2
from skimage import measure
import analysis_tools as als
import global_variables
import impreps as imps

def well_processor(well, roi, mask):
    
    assert (type(roi)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(roi.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(roi) >= 0 & np.amax(roi) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask) >= 0) & (np.amax(mask) <= 1), 'mask_well has to be binary image'
    
    h,w = roi.shape[0:2]

    #Crop well from tray with label bounding box
    left_column = round(max(0,well.bbox[1]))
    right_column = min(w, well.bbox[3])
    up_row = round(max(0, well.bbox[0]))
    down_row = min(h,well.bbox[2])
 
    rgb_well = roi[up_row:down_row,left_column:right_column,:]
    mask_well = mask[up_row:down_row,left_column:right_column]

    # evaluation of well with different algorithm
    shadearea_mask = sg.well_shade_search(mask_well, rgb_well)
    bclosearea_mask = sg.well_bclosearea_search(mask_well, rgb_well)


    #comparison of algorithms
    final_mask = sg.alghoritm_comparison(rgb_well, shadearea_mask, bclosearea_mask)
    
    plant_frame = rgb_well[np.where(final_mask>0)]

    if(plant_frame.any()):

        bgr_means = np.mean(plant_frame ,axis=tuple(range(plant_frame.ndim-1)))

    else:

        bgr_means = np.array([None,None,None])

    return final_mask, bgr_means


def image_processor(image, mask, file, col_num, row_num):
    
    assert (type(image)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(image.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask) >= 0) & (np.amax(mask) <= 1), 'mask_well has to be binary image'
    
    assert type(file) == str, 'filename hast be string'
            
    metadata = imps.image_metadata_handler(file)
    
    #Crop tray
    tray_roi = imps.roi_cropper(image)
    h,w = tray_roi.shape[0:2]
    
    #Decode barcode dat
    barcode_data = imps.barcode_reader(tray_roi)
    
    #resize mask to roi shape
    mask = cv2.resize(mask,(w, h),0,0,cv2.INTER_NEAREST)
    
    tray_plant_mask = np.zeros(mask.shape)

    #find wells in image
    #https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
    mask_labels = measure.label(mask*255)
    #compute propeties of wells
    mask_properties = measure.regionprops(mask_labels)

    wells = als.well_former(mask_properties, col_num, row_num)
                                     
    data = []

    for well in wells:

        segmented_plant_mask, means = well_processor(well, tray_roi, mask)

        pixels = segmented_plant_mask.sum()
        plant_id = str(int(barcode_data[-3:])) + '_' + str(well.row) + '_' + str(well.column)

        data.append(dict(zip(('filename', 'date', 'time', 'location', 'x_coordinate', 'y_coordinate', 'plant_id', 'barcode_data','well_row','well_column','pixel_num','r_mean', 'g_mean', 'b_mean'),
                             (file, metadata.date, metadata.time, metadata.location, metadata.x_coordinate, metadata.y_coordinate, plant_id, barcode_data, well.row, well.column, pixels, 
                                means[2], means[1], means[0]))))
        
        
        well_coordinates = np.where(segmented_plant_mask>0)
        cols = np.round(well_coordinates[0]+well.bbox[0])
        rows = np.round(well_coordinates[1]+well.bbox[1])

        tray_coordinates = (cols,rows)
        tray_plant_mask[tray_coordinates] = 1
        

    plants_perim = als.bwperim(tray_plant_mask*255, n=4) 
    wells_perim = als.bwperim(mask*255, n=4)
   
    return data, wells_perim, plants_perim, tray_roi


def process_images(imageLoad):
    
    print("[INFO] Starting process {}".format(imageLoad["id"]))
    print("[INFO] For process {}, there is {} images in processing queue".format(imageLoad["id"], len(imageLoad["files_names"])))
    
    ##Tray mask loading and formatting
    well_num = global_variables.row_num*global_variables.col_num
    #Load mask of wells from file
    mask = cv2.imread(global_variables.masks_path + str(well_num) +'.png')
    #Convert image to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #Transform to binary image for better wells localization
    mask = (mask > 0).astype('uint8')
    
    final_data = []

    f = open(imageLoad["temp_path"] + "failures_{}.txt".format(imageLoad["id"]),"w+")
    
    for imageName in imageLoad["files_names"]:
        
        try:
        
            image = cv2.imread(imageLoad["input_path"] + imageName)

            image_data, well_contours, plant_contours, roi = image_processor(image, mask, imageName, global_variables.col_num, global_variables.row_num)

            contours_wells, _ = cv2.findContours(well_contours.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours_plants, _ = cv2.findContours(plant_contours.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            contoured_roi_ = cv2.drawContours(roi, contours_wells, -1, (255, 0, 0), 1)
            contoured_roi = cv2.drawContours(contoured_roi_, contours_plants, -1, (0, 0, 255), 1)

            final_data = final_data + image_data

            cv2.imwrite(imageLoad["output_path"] + imageName, contoured_roi)
            
        except Exception as e:

            print(e)
            f.write(imageName + '\n')
    
    df = pd.DataFrame(final_data)
    df = df[['filename', 'date', 'time', 'location', 'x_coordinate', 'y_coordinate', 'plant_id', 'barcode_data','well_row','well_column','pixel_num','r_mean', 'g_mean', 'b_mean']]
    df.sort_values(by=['location', 'date', 'well_row','well_column'])
    df.to_excel(imageLoad["temp_path"] + '/batch_result_{}.xlsx'.format(imageLoad["id"]))