import datetime
import re
import cv2
import pyzbar.pyzbar as pz
import itertools
import os
import numpy as np

from skimage.color import rgb2hsv
from skimage.color import rgb2lab
from scipy.ndimage.filters import median_filter
import scipy
from scipy import ndimage
from skimage.morphology import binary_opening, binary_closing, label
from scipy.ndimage.morphology import distance_transform_edt

import scipy.io
from skimage import measure
import pandas as pd

import global_variables


def image_metadata_handler(filename):
    
    #split filename
    metadata = filename.split('_')
    #check if filename is in 6-component format
    assert len(metadata) == 6, 'Format of filename is wrong'
    
    #define  output object
    class IMAGE_METADATA:
        
            def __init__(self, date, x_coordinate, y_coordinate, location):

                self.date = date
                self.x_coordinate = x_coordinate
                self.y_coordinate = y_coordinate
                self.location = location 
                
    #date metadata
    date_metadata = metadata[2].split('-')
    #date formatting
    date = datetime.datetime(int(date_metadata[1]), int(date_metadata[2]), int(date_metadata[3]), int(date_metadata[4]), int(date_metadata[5]), int(date_metadata[6]),)

    #tray metadata
    tray = metadata[4]
    
    #tray coordinates
    coord = tray.split('-')[1].split('x')
    x_coordinate = int(coord[0])
    y_coordinate = int(coord[1])
    
    #tray location
    location = tray.split('-')[1]
    #assign metadata to result object
    image_metadata = IMAGE_METADATA(date, x_coordinate, y_coordinate, location) 
    
    return image_metadata


def roi_cropper(image):
    
    assert (type(image)  == np.ndarray), 'Input data has to be RGB image' 
    assert (len(image.shape) == 3),'Input data has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    #Cropping cental area of image in x-coordinate 
    cropped = image[:,500:2150,:]  

    #Multiplication of Greeen and Red channel (Tomas trick)
    product = np.double(cropped[:,:,1]) * np.double(cropped[:,:,2])

    #Standardization of grayscale image
    product = product - np.amin(product)
    product = product / np.amax(product)

    #Identification of blue lines (Tomas method) -> grayscale image
    blue_lines = (product < 0.15) & (cropped[:,:,0] > 100)

    #Sum of columns in grayscale image
    col_sum = np.sum(blue_lines,axis = 0)
    #half of the size of image in x-coordinate
    half1 = len(col_sum)//2

    #left side of image
    left = col_sum[:half1]
    #differences of neighbouring columns
    peak_left = np.diff(left)
    #find left boundary as minimum of differences
    left_boundary = np.where(peak_left == np.amin(peak_left))
    #shift left boundary
    left_boundary = left_boundary[0] + 30
    
    #right side of image
    right = col_sum[half1+1:]
    #differences of neighbouring columns
    peak_right = np.diff(right)
    #find right boundary as maximum of differences
    right_boundary = np.where(peak_right == np.amax(peak_right))
    #shift right boundary
    right_boundary = half1 + right_boundary[0] - 30
    
    #Sum of rows in grayscale image
    row_sum = np.sum(blue_lines,axis = 1) 
    #half of the size of image in y-coordinate
    half2 = len(row_sum)//2

    #up side of image
    up = row_sum[:half2]
    #differences of neighbouring rows
    peak_up = np.diff(up)
    #find up boundary as minimum of differences
    up_boundary = np.where(peak_up == np.amin(peak_up))
    #shift up boundary
    up_boundary = up_boundary[0] + 30

    #down side of image
    down = row_sum[half2+1:]
    #differences of neighbouring rows
    peak_down = np.diff(down)
    #find down boundary as maximum of differences
    down_boundary = np.where(peak_down == np.amax(peak_down))
    #shift down boundary
    down_boundary = half2 + down_boundary[0] 

    #cropp roi
    roi = cropped[up_boundary[0]:down_boundary[0],left_boundary[0]:right_boundary[0],:]
    #roi = cropped[0:down_boundary[0],left_boundary[0]:right_boundary[0],:]

    return roi

def barcode_reader(image):
    
    assert (type(image)  == np.ndarray), 'Input data has to be RGB image' 
    assert (len(image.shape) == 3),'Input data has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    #roi shape
    h,w = image.shape[0:2]

    #crop barcode area
    code_area = image[h-90:h-10,200:900,:]
    
    barcode_reader_output = pz.decode(code_area)
    
    if(not barcode_reader_output):
        
        #barcode area shap
        h_bar, w_bar = code_area.shape[0:2]

        #resizing barcode area. to 3-times bigger image
        code_area_big = cv2.resize(code_area,(3*w_bar, 3*h_bar))

        #convert barcode to grayscale image
        code_area_gray = cv2.cvtColor(code_area_big,cv2.COLOR_BGR2GRAY)

        #sharppening of grayscale barcode
        code_area_sharp = sharp_image(code_area_gray, 10, 0.4, 'laplace')
        code_area_sharp = code_area_sharp.astype('uint8')
        
        # Otsu's thresholding
        thresh,code_area_thresholded = cv2.threshold(code_area_sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        barcode_reader_output = pz.decode(code_area_thresholded)
        
        if(not barcode_reader_output):
            
            #Peter thresholding
            barcode_sharp = (code_area_sharp > thresh-20).astype('uint8')*255
            
            #Make a border
            barcode_sharp = cv2.copyMakeBorder(barcode_sharp,top=100,bottom=100,left=100,right=100,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])
    
            barcode_reader_output = pz.decode(barcode_sharp)
        
            if(not barcode_reader_output):
    
                #histogram equalization of  sharpened grayscale barcode
                #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                code_area_equalized = clahe.apply(code_area_sharp)

                #Peter thresholding
                barcode_equalized = (code_area_equalized > 140).astype('uint8')*255

                #Make a border
                barcode_equalized = cv2.copyMakeBorder(barcode_equalized,top=100,bottom=100,left=100,right=100,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

                barcode_reader_output = pz.decode(barcode_sharp)
                
    barcode_data = None
    
    if(barcode_reader_output):
        
        barcode_data = barcode_reader_output[0][0].decode("UTF-8")     
    
    return barcode_data

def well_former(mask_properties, col_num, row_num):

    class well:

        def __init__(self, row, column, bbox, centroid):

            self.row = row
            self.column = column
            self.bbox = bbox
            self.centroid = centroid    

    wells = []

    row_ids = range(1, row_num+1)
    row_ids = list(itertools.chain.from_iterable(itertools.repeat(x, col_num) for x in row_ids))
    col_ids = range(1, col_num+1)
    col_ids = list(itertools.chain.from_iterable(itertools.repeat(x, row_num) for x in col_ids))


    row_sort = sorted(mask_properties, key=lambda area: area.centroid[0]);
    col_sort = sorted(mask_properties, key=lambda area: area.centroid[1]);

    row_order = []
    col_order = []

    for i, row in enumerate(row_sort):

        row_order.append((row_ids[i], row.label))

    for i, col in enumerate(col_sort):

        col_order.append((col_ids[i], col.label))    


    for i, prop in enumerate(mask_properties):

        row = [item[0] for item in row_order if item[1] == prop.label]
        col = [item[0] for item in col_order if item[1] == prop.label]

        well_iter = well(row[0], col[0], prop.bbox, prop.centroid)
        wells.append(well_iter)
        
    return wells


def well_shade_search(mask_well, rgb_well):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'


    #image conversion to hsv color space (equation of conversion: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
    hsv_well = rgb2hsv(rgb_well)

    #get image channels
    h_well, s_well, v_well = cv2.split(hsv_well)

    #segmentation of seed based on hue
    h_plant_well = ((h_well < 0.5) & (h_well > 0.17)).astype('uint8')

    #noise filtering from segmented image
    green_plant = median_filter(h_plant_well, 5)

    plant_mask = np.zeros(green_plant.shape) 

    #whitening of area around well area
    v_shade_well = v_well.copy()
    v_shade_well[~mask_well.astype('bool')] = 1

    #sharpening of v_shade_well
    blurred_f = ndimage.gaussian_filter(v_shade_well, 1)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 25
    v_sharped_well = blurred_f + alpha * (blurred_f - filter_blurred_f)

    ######Looking for thresholds
    #Thresholds are computed as mean value of plant area and noisy shade areas
    #Edge threshold is combination of these two thresholds and identify edges

    #whitening of seed area -> this gives noisy background of pixels which were segmented, but doesn't belong to seed area
    noise_background = v_shade_well.copy()
    #Removing green plant pixels from active pixels
    noise_background[green_plant>0] = 1

    #Computing of background threshold from background noisy pixels
    #Background threshold is mean value of all active pixels in v-channel except plant green pixels
    background_thresh = np.mean(noise_background[noise_background != 1])

    #Foreground threshold is mean value of v-channel plant green pixels
    plant_thresh = np.mean(v_shade_well[green_plant>0])

    #Thresholds difference
    thresh_diff = background_thresh - plant_thresh

    #threshold for edge detection
    edge_thresh =  plant_thresh + (thresh_diff/4)

    #Thresholding of sharped well with edge_threshold
    v_thresholded_well = v_sharped_well < edge_thresh

    #Binary opening of thresholded well
    v_opened_well = binary_opening(v_thresholded_well).astype('uint8')*255

    #filter noisy pixels
    v_filtered_well = median_filter(v_opened_well, 6)

    #Find objects/labels in filtered well
    v_well_labels = measure.label(v_filtered_well)
    #Compute object properties
    v_well_properties = measure.regionprops(v_well_labels)
    #number of objects
    num_labels = len(v_well_properties)

    green_proportion = np.zeros((1,num_labels))

    #Computation of green area in detected objects/labels
    for i, prop in enumerate(v_well_properties):

        object_coordinates = np.where(v_well_labels == prop.label)

        if(len(object_coordinates[0]) > 30):

            green_sum = sum(green_plant[object_coordinates]==1)
            green_proportion[0,i] = (green_sum/len(object_coordinates[0]))*100

    #If there is any founded object in well       
    if(num_labels != 0):

        #find value of the most green object in well
        max_percentage = np.max(green_proportion)

        #compute 80 percent value of maximum
        percentage_80 = 8*max_percentage/10

        #If the value of the most green object is more than 5 percent
        if(max_percentage>5):

            #for each object
            for i, prop in enumerate(v_well_properties):

                #if green value of object is at least 80 of most green object
                if(green_proportion[0,i] > percentage_80):

                    #find object positions in well
                    object_coordinates = np.where(v_well_labels==prop.label)

                    #Criterium defined by size of identified object
                    crit = len(object_coordinates[0]) * len(object_coordinates[1])

                    #if object is not too big
                    if(crit < 17000**2):

                        #Than pixels of this object is considered to be plant/seed in resulting mask
                        plant_mask[object_coordinates] = 1
                
        #Corection of empty areas in plant/seed area

        #fill empty areas
        filledmask = imfill(plant_mask)
        #find holes
        hole = filledmask-plant_mask
        #find holes as a morphological objects
        hole_labels = measure.label(hole,4)
        #Compute hole morphological propoerties
        hole_properties = measure.regionprops(hole_labels)

        #for each founded hole
        for i,prop in enumerate(hole_properties):

            #take hole's coordinate
            coordinates = (prop.coords[:,0],prop.coords[:,1])
            #compute median of area defined with hole coordinates in v-chanell of sharped well image
            hole_median = np.median(v_sharped_well[coordinates])

            #if area median is higher than 2/3 of threshold
            if(hole_median > (edge_thresh/1.5)):
                #and area is not too big
                if(plant_mask.shape[0]*plant_mask.shape[1]*0.2 > coordinates[0].shape[0]):
                    #hole area is considered to be plant
                    plant_mask[coordinates] = 1
                    
    return plant_mask



def well_bclosearea_search(mask_well, rgb_well):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'

    #image conversion to hsv color space (equation of conversion: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
    hsv_well = rgb2hsv(rgb_well)

    #image conversion to lab color space
    lab_well = cv2.cvtColor(rgb_well, cv2.COLOR_BGR2LAB)

    #get hsv image channels
    h_well, s_well, v_well = cv2.split(hsv_well)

    #get hsv image channels
    l_well, a_well, b_well = cv2.split(lab_well)

    #segmentation of seed based on hue
    h_plant_well = ((h_well < 0.5) & (h_well > 0.17)).astype('uint8')

    #noise filtering from segmented image
    green_plant = median_filter(h_plant_well, 5)

    #array for output plant mask
    plant_mask = np.zeros(green_plant.shape) 

    #whitening of area around well area
    v_shade_well = v_well.copy()
    v_shade_well[~mask_well.astype('bool')] = 1

    #sharpening of v_shade_well
    blurred_f = ndimage.gaussian_filter(v_shade_well, 1)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 25
    v_sharped_well = blurred_f + alpha * (blurred_f - filter_blurred_f)


    #####Looking for thresholds
    #Thresholds are computed as mean value of plant area and noisy shade areas
    #Edge threshold is combination of these two thresholds and identify edges

    #whitening of seed area -> this gives noisy background of pixels which were segmented, but doesn't belong to seed area
    noise_background = v_shade_well.copy()
    #Removing green plant pixels from active pixels
    noise_background[green_plant>0] = 1

    #Computing of background threshold from background noisy pixels
    #Background threshold is mean value of all active pixels in v-channel except plant green pixels
    background_thresh = np.mean(noise_background[noise_background != 1])

    #Foreground threshold is mean value of v-channel plant green pixels
    plant_thresh = np.mean(v_shade_well[green_plant>0])

    #Thresholds difference
    thresh_diff = background_thresh - plant_thresh

    #threshold for edge detection
    edge_thresh =  plant_thresh + (thresh_diff/4)

    #Thresholding of sharped well with edge_threshold
    v_thresholded_well = v_sharped_well < edge_thresh

    #Binary opening of thresholded well
    v_opened_well = binary_opening(v_thresholded_well).astype('uint8')*255

    #filter noisy pixels
    v_filtered_well = median_filter(v_opened_well, 6)

    #Find objects/labels in filtered well
    v_well_labels = measure.label(v_filtered_well)
    #Compute object properties
    v_well_properties = measure.regionprops(v_well_labels)
    #number of objects
    num_labels = len(v_well_properties)

    if(v_well_properties):
        
        #At this point method starts to be different of well_shade search
        #Method is based on Lab color space. It uses b-channel of Lab color space, at this channel is even really small green object visible.
        #This method helps to include small green objects within segmentation.

        #array of green proportion value of identified ojects
        green_proportion = np.zeros((1,num_labels))
        #array of b-channel medfian value of identified ojects
        b_channel_medians = np.zeros((1,num_labels))

        #for each identified object in well
        for i, prop in enumerate(v_well_properties):
            #if object is big enough
            if(prop.area > 50):

                #find object coordinates
                coordinates = (prop.coords[:,0],prop.coords[:,1])
                #Compute b-channel median of identified object
                b_channel_medians[0,prop.label-1] = np.median(b_well[coordinates])


        #Maximal median value
        max_med = np.max(b_channel_medians)
        #Index of object with maximal median value
        max_med_ind = np.argmax(b_channel_medians)


        #If there is some green objetc
        if(max_med > 0):

            #Compute green proportion of each identified object in identified green plant    
            for i, prop in enumerate(v_well_properties):

                #find object coordinates
                object_coordinates = np.where(v_well_labels == prop.label)

                #compute green proportion of object in plant area
                green_sum = sum(green_plant[object_coordinates]==1)
                green_proportion[0,prop.label-1] = (green_sum/len(object_coordinates[0]))*100

            #Define plant mask based on hsv color space v-channel
            plant_mask[v_well_labels==max_med_ind+1] = 1

            #Argument, which stops while cycle
            adding = True



            #While cycle tries to add some other objects to plant mask
            #Idea is to take shade noisy pixels surrounding identified plant and try to identify them as plant based on comparison of
            #b-channel median of tested object and green plant median (which is maximal)
            added_labels = []
            
            while adding:

                #Distance of non-active pixels (small shadows) from, active pixels (big plant area)
                distance = distance_transform_edt(1-plant_mask) 
                #define neighborhood, which is close enough
                neighborhood = distance < 10
                #and remove  green plant area pixels
                neighborhood[plant_mask.astype('bool')] = False

                # array of potentional objects, which can belong to plant area
                potential_objects = np.zeros((1,num_labels)).astype('bool')

                #for each identified object
                for i, prop in enumerate(v_well_properties):

                    #object coordinates
                    coordinates = (prop.coords[:,0],prop.coords[:,1])

                    #if object is in neignborhood
                    if(neighborhood[coordinates].any()):
                        #it is considered as a potentional plant
                        potential_objects[0,i] = True    
                
                #if there is not any potentional plants, stop while cycle with adding argument
                if(~potential_objects.any()):
                    adding = False

                #If there are potentional plants
                else:

                    #Define mask for new objects
                    new_objects = np.zeros(plant_mask.shape)

                    #get indexes of potentional objects, which will be tested
                    tested_objects = np.where(potential_objects == True)
                    
                    #for each tested object
                    for i in range(0,tested_objects[0].shape[0]):

                        #index of tested object
                        idx = tested_objects[1][i]
                       
                        #If object b-channel median is higher than 2/3 of plant b-channel median
                        if((b_channel_medians[0,idx] > (2/3)*max_med) & (idx not in added_labels)):
                            
                            #define mask of new object, which will bi included in plant mask
                            new_objects[v_well_labels==idx] = 1
                            added_labels.append(idx)
                    #If there are pixels, which are considered as a plant
                    if(new_objects.any()):
                        
                        #refine plant mask
                        plant_mask = cv2.bitwise_or(plant_mask, new_objects)
                        
                    else:

                        adding = False   


        #Corection of empty areas in plant/seed area

        #fill empty areas
        filledmask = imfill(plant_mask)
        #find holes
        hole = filledmask-plant_mask
        #find holes as a morphological objects
        hole_labels = measure.label(hole,4)
        #Compute hole morphological properties
        hole_properties = measure.regionprops(hole_labels)

        #for each founded hole
        for i,prop in enumerate(hole_properties):

            #take hole's coordinate
            coordinates = (prop.coords[:,0],prop.coords[:,1])
            #compute median of area defined with hole coordinates in b-chanell well 
            hole_median = np.median(b_well[coordinates])

            #if area median is higher than 2/3 of threshold
            if(hole_median > (max_med/3)):
                #and area is not too big
                if(plant_mask.shape[0]*plant_mask.shape[1]*0.2 > coordinates[0].shape[0]):

                    #hole area is considered to be plant
                    plant_mask[coordinates] = 1  

    return plant_mask


def well_lab_search(mask_well, rgb_well):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'

    lab_well = cv2.cvtColor(rgb_well, cv2.COLOR_BGR2LAB)

    #get image channels
    l_well, a_well, b_well = cv2.split(lab_well)
    b_median = np.median(b_well)
    
    #print(np.median(b_well))
    l_well_segm = ((l_well > 110) & (l_well < 210 - (230-np.median(l_well)))).astype('uint8')
    
    #noise filtering from segmented image
    green_plant = median_filter(l_well_segm, 3)
    green_plant[~mask_well.astype('bool')] = 0
    #green_plant = l_well_segm
    
    #Find objects/labels in filtered well
    well_labels = measure.label(green_plant)
    #Compute object properties
    well_objects = measure.regionprops(well_labels)

    plant_mask = np.zeros(rgb_well.shape[0:2]) 
    
    #Check for 200 pixels minimum of object size
    for i, well_object in enumerate(well_objects):
        
        object_coordinates = np.where(well_labels == well_object.label)
        #print(len(object_coordinates[0]))
        
        #Check for 200 pixels minimum of object
        if(len(object_coordinates[0]) > 200 and len(object_coordinates[0]) < 5000):

            
            #print(np.median(b_well[object_coordinates]))
            
            #compute median of b-channel of given labeled object
            b_value = np.median(b_well[object_coordinates])

            if(abs(b_median-b_value) > 4):
 
                plant_mask[object_coordinates] = 1 
    
    return plant_mask


def alghoritm_comparison(rgb_well, shadearea_mask, bclosearea_mask):
    
    #Check of input arguments correctness
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(shadearea_mask)  == np.ndarray), 'Mask has to be binary image' 
    assert (len(shadearea_mask.shape) == 2), 'Mask has to be binary image'
    assert (np.amin(shadearea_mask) >= 0) & (np.amax(shadearea_mask) <= 1), 'Mask has to be binary image'
    
    assert (type(bclosearea_mask)  == np.ndarray), 'Mask has to be binary image' 
    assert (len(bclosearea_mask.shape) == 2), 'Mask has to be binary image'
    assert (np.amin(bclosearea_mask) >= 0) & (np.amax(bclosearea_mask) <= 1), 'Mask has to be binary image'
    
    from skimage.color import rgb2hsv

    h,w = rgb_well.shape[0:2]

    if(int(np.sum(shadearea_mask)) == h*w and int(np.sum(bclosearea_mask)) == h*w):

        choosen_mask = np.zeros(rgb_well.shape[0:2])

    elif(int(np.sum(shadearea_mask)) == h*w):

        choosen_mask = bclosearea_mask

    elif(int(np.sum(bclosearea_mask)) == h*w):

        choosen_mask = shadearea_mask

    #Find best result
    
    else:
        
        shadearea_coords = np.where(shadearea_mask == 1)
        bclosearea_coords = np.where(bclosearea_mask == 1)
        
        if(shadearea_coords[0].shape[0] > 0 and bclosearea_coords[0].shape[0] > 0):
            
            #find bigger 
            bigger_area_size = max(shadearea_coords[0].shape[0], bclosearea_coords[0].shape[0])
            #areas size differernce
            size_diff = abs(shadearea_coords[0].shape[0]-bclosearea_coords[0].shape[0])

            #Compute means and deviations in h-channel of hsv color space

            #image conversion to hsv color space (equation of conversion: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
            hsv_well = rgb2hsv(rgb_well)

            #get hsv image channels
            h_well, s_well, v_well = cv2.split(hsv_well)

            #stats of first method
            shadearea_mean = np.mean(h_well[shadearea_coords])
            shadearea_deviation = np.std(h_well[shadearea_coords])

            #stats of second method
            bclosearea_mean = np.mean(h_well[bclosearea_coords])
            bclosearea_deviation = np.std(h_well[bclosearea_coords])


            # vem vetší pokud jeji odchylka neni upln? mimo - co je mimo? - v
            # testovanem pripad? je odchylka vetsi oblasti dokonce mensí - je
            # to tak vždy? (pokud tam neni bordel?)

            #If masked object of bclose method is bigger than masked object of shade method
            if(bclosearea_coords[0].shape[0] > shadearea_coords[0].shape[0]):
                #If deviation smaller object is bigger
                if(bclosearea_deviation < shadearea_deviation):

                    #bclosearea method mask is considered to be better result - masked object is more compact
                    choosen_mask = bclosearea_mask

                #if bclose method deviation is bigger, some noisy objects were probaly added. So shade method mask is considered to be better.
                else:

                    choosen_mask = shadearea_mask

            #If shade method masked object is bigger, this method reult is taken       
            else:

                choosen_mask = shadearea_mask

        elif(shadearea_coords[0].shape[0] == 0 and bclosearea_coords[0].shape[0] > 0):
        
            choosen_mask = bclosearea_mask
            
        elif(shadearea_coords[0].shape[0] > 0 and bclosearea_coords[0].shape[0] == 0):
        
            choosen_mask = shadearea_mask 
            
        else:    

            choosen_mask = shadearea_mask 
            
            
        #Potential improvement with using contour/area ratio (This need testing)

        #perimeter_shade = bwperim(shadearea_mask, n=4)
        #perimeter_shade_size = np.where(perimeter_shade > 0)[0].shape[0]
        #shade_area_size = np.where(shadearea_mask > 0)[0].shape[0]
        #shade_ratio = perimeter_shade_size/shade_area_size

        #perimeter_bclose = bwperim(bclosearea_mask, n=4)
        #perimeter_bclose_size = np.where(perimeter_bclose > 0)[0].shape[0]
        #bclose_area_size = np.where(bclosearea_mask > 0)[0].shape[0]
        #bclose_ratio = perimeter_bclose_size/bclose_area_size

    return choosen_mask



##############Processors

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
    #shadearea_mask = well_shade_search(mask_well, rgb_well)
    #bclosearea_mask = well_bclosearea_search(mask_well, rgb_well)
    lab_mask = well_lab_search(mask_well, rgb_well)

    #comparison of algorithms
    #final_mask = alghoritm_comparison(rgb_well, shadearea_mask, bclosearea_mask)
    final_mask = lab_mask 
    
    plant_frame = rgb_well[np.where(final_mask>0)]

    bgr_means = np.mean(plant_frame ,axis=tuple(range(plant_frame.ndim-1)))

    return final_mask, bgr_means


def image_processor(image, mask, file, col_num, row_num):
    
    assert (type(image)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(image.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask) >= 0) & (np.amax(mask) <= 1), 'mask_well has to be binary image'
    
    assert type(file) == str, 'filename hast be string'
    
    #class well_out:

        #def __init__(self, row, column, bbox, value, barcode_data, plant_mask):

            #self.row = row
            #self.column = column
            #self.bbox = bbox
            #self.value = value 
            #self.barcode_data = barcode_data
            #self.plant_mask = plant_mask
            
    metadata = image_metadata_handler(file)
    
    #Crop tray
    tray_roi = roi_cropper(image)
    h,w = tray_roi.shape[0:2]
    
    #Decode barcode dat
    barcode_data = barcode_reader(tray_roi)
    
    #resize mask to roi shape
    mask = cv2.resize(mask,(w, h),0,0,cv2.INTER_NEAREST)
    
    tray_plant_mask = np.zeros(mask.shape)

    #find wells in image
    #https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
    mask_labels = measure.label(mask*255)
    #compute propeties of wells
    mask_properties = measure.regionprops(mask_labels)

    wells = well_former(mask_properties, col_num, row_num)
                                     
    data = []

    for well in wells:

        segmented_plant_mask, means = well_processor(well, tray_roi, mask)

        pixels = segmented_plant_mask.sum()

        data.append(dict(zip(('filename', 'date', 'location', 'x_coordinate', 'y_coordinate', 'barcode_data','well_row','well_column','pixel_num','r_mean', 'g_mean', 'b_mean'),
                             (file, metadata.date, metadata.location, metadata.x_coordinate, metadata.y_coordinate, barcode_data,well.row, well.column, pixels, 
                                means[2], means[1], means[0]))))
        
        
        well_coordinates = np.where(segmented_plant_mask>0)
        cols = np.round(well_coordinates[0]+well.bbox[0])
        rows = np.round(well_coordinates[1]+well.bbox[1])

        tray_coordinates = (cols,rows)
        tray_plant_mask[tray_coordinates] = 1
        

    plants_perim = bwperim(tray_plant_mask*255) 
    wells_perim = bwperim(mask*255)
   
    return data, wells_perim, plants_perim, tray_roi



##############HELPERS
def sharp_image(image, sigma, power, method):
    
    import cv2
    from scipy.ndimage.filters import median_filter
    
    #https://www.idtools.com.au/unsharp-masking-python-opencv/
    
    assert (type(method) == str) & ((method == 'kernel') | (method == 'laplace')), 'Method of image sharpening has to be krenle or laplace'
    assert (type(image)  == np.ndarray) & (len(image.shape) == 2) & (np.amin(image) >= 0) & (np.amax(image) <= 255), 'Input data has to be grayscale image' 
    assert type(sigma) == int, 'size gives the shape that is taken from the input array, at every element position, to define the input to the filter function'
    assert ((power > 0) & (power < 1)), 'Value of power parameter has to be in (0,1) interval'

    
    if(method == 'laplace'):
    
        # Median filtering
        image_mf = median_filter(image, sigma)

        # Calculate the Laplacian
        lap = cv2.Laplacian(image_mf,cv2.CV_64F)

        # Calculate the sharpened image
        sharp = image-power*lap
   
    elif(method == 'kernel'):
        
        kernel= np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-2,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

        # applying the sharpening kernel to the input image & displaying it.
        sharp = cv2.filter2D(image, -1, kernel)
        
       
    return sharp


def imfill(image):
    
    #Check of input arguments correctness
    assert (type(image)  == np.ndarray), 'Input has to be binary image' 
    assert (len(image.shape) == 2), 'Input has to be binary image'
    assert (np.amin(image) >= 0) & (np.amax(image) <= 1), 'Input has to be binary image'

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    im_th = im_th.astype('uint8')

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    im_floodfill = cv2.floodFill(im_th, mask, (0,0), 255)
    im_floodfill = im_floodfill[1]
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out



def bwperim(bw, n=4):
            
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    
    if n not in (4,8):
        
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
        
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
        
    return ~idx * bw







def rgb2LAB(image):
        
    assert (type(image)  == np.ndarray), 'Input data has to be RGB image' 
    assert (len(image.shape) == 3),'Input data has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'



    def func(mat):
        
        mat_1 = (mat >= 0.008856).astype('uint8')
        mat_2 = (mat < 0.008856).astype('uint8')
            
        res_mat =np.multiply(mat_1,np.power(mat, 1/3)) + np.multiply(mat_2,((841/108) * mat + (4/29)))
        
        return res_mat
    
    def invgammacorrection(matrix):
    
        indicator_mat_1 = (matrix <= 0.0404482362771076).astype('uint8')
        indicator_mat_2 = (matrix > 0.0404482362771076).astype('uint8')

        res_mat = np.multiply(indicator_mat_1, matrix/12.92) + np.multiply(np.power(((matrix+0.055)/1.055),2.4),indicator_mat_2)
        return res_mat

    #RGB values lie between 0 to 1.0
    b,g,r = cv2.split(image)
    
    R = invgammacorrection(r/255)
    G = invgammacorrection(g/255)
    B = invgammacorrection(b/255)
    
    
    
    
    #Conversion Matrix from RGB to XYZ
    matrix = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.057]])
    matrix = np.linalg.inv(matrix)
    
    #RGB2XYZ
    X = matrix[0,0]*R + matrix[1,0]*G + matrix[2,0]*B 
    Y = matrix[0,1]*R + matrix[1,1]*G + matrix[2,1]*B 
    Z = matrix[0,2]*R + matrix[1,2]*G + matrix[2,2]*B 
    
    #Normalize for D65 white point
    X = X / 0.950456
    Z = Z / 1.088754
    ################
  
    
    fx = func(X)
    fy = func(Y)
    fz = func(Z)
    
    # Calculate the L
    L = 116 * fy - 16.0

    # Calculate the a 
    a = 500*(fx - fy)

    # Calculate the b
    b = 200*(fy - fz)
   
    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    Lab = cv2.merge((L,a,b))
    
    return Lab




if __name__ == '__main__':

    assert (type(global_variables.path)==str) & os.path.exists(global_variables.path), 'Path to folder with batch of images does not exist'
    assert (type(global_variables.row_num)==int), 'Number of rows has to be integer'
    assert (type(global_variables.col_num)==int), 'Number of columns has to be integer'

    well_num = global_variables.row_num*global_variables.col_num

    assert os.path.exists(global_variables.masks_path + str(well_num) +'.png'), 'Mask does not exist'
    
    batch_path = global_variables.path + '/batch/'
    output_path = batch_path + 'results/'
    
    if(not os.path.exists(output_path)):
        
        os.makedirs(output_path)
    
    formats = ('.JPG','.jpg','.PNG','.png','.bmp','.BMP','.TIFF','.tiff','.TIF','.tif')

        
    files = [file for file in os.listdir(global_variables.batch_path) if file.endswith(formats)]
    
    ##Tray mask loading and formatting
    #Load mask of wells from file
    mask = cv2.imread(global_variables.masks_path + str(well_num) +'.png')
    #Convert image to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #Transform to binary image for better wells localization
    mask = (mask > 0).astype('uint8')
    
    final_data = []
    
    f = open(output_path + "failures.txt","w+")

    for file in files:

        try:
        
            image = cv2.imread(global_variables.batch_path + file)

            image_data, well_contours, plant_contours, roi = image_processor(image, mask, file, global_variables.col_num, global_variables.row_num)

            _, contours_wells, _ = cv2.findContours(well_contours.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            _, contours_plants, _ = cv2.findContours(plant_contours.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
           
            contoured_roi_ = cv2.drawContours(roi, contours_wells, -1, (255, 0, 0), 1)
            contoured_roi = cv2.drawContours(contoured_roi_, contours_plants, -1, (0, 0, 255), 1)
            
            final_data = final_data + image_data
            
            cv2.imwrite(output_path + file, contoured_roi)

        except Exception as e:

            print(e)
            f.write(file + '\n')
        
        
    df = pd.DataFrame(final_data)
    df = df[['filename', 'date', 'location', 'x_coordinate', 'y_coordinate', 'barcode_data','well_row','well_column','pixel_num','r_mean', 'g_mean', 'b_mean']]
    df.to_excel(output_path + '/batch_result.xlsx')