import cv2
import numpy as np

from skimage.color import rgb2hsv
from skimage.color import rgb2lab
from skimage.morphology import binary_opening, binary_closing, label
from skimage import measure
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import median_filter
import scipy


from operator import attrgetter

import analysis_tools as als


def well_shade_search(mask_well, rgb_well, h_max=0.5, h_min=0.17, alpha = 25, plant_min_size=100, plant_max_size=80000, green_index_minimum=0.05, max_percentage_level=0.8, hole_size_crit=0.001):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'

    assert (h_max>h_min and h_max<1), 'Maximal value of h channel has to be lower than 1 and higher than h_min argument'
    assert (h_min>0 and h_min<h_max), 'Minimal value of h channel has to be lower than h_max argument and higher than 0'
    assert (type(alpha)==int), 'Alpha argument is integer'
    assert (type(plant_min_size)==int), 'Plant_min_size argument is integer'
    assert (green_index_minimum>0 and green_index_minimum<1), 'green_index_minimum argument value has to be lower than 1 argument and higher than 0'
    assert (max_percentage_level>0 and max_percentage_level<1), 'max_percentage_level argument value has to be lower than 1 argument and higher than 0'    
    assert (hole_size_crit>0 and hole_size_crit<1), 'hole_size_crit argument value has to be lower than 1 argument and higher than 0'



    #image conversion to hsv color space (equation of conversion: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
    hsv_well = rgb2hsv(rgb_well)

    #get image channels
    h_well, s_well, v_well = cv2.split(hsv_well)

    #segmentation of seed based on hue
    h_plant_well = ((h_well < h_max) & (h_well > h_min)).astype('uint8')

    #noise filtering from segmented image
    green_plant = median_filter(h_plant_well, 5)

    plant_mask = np.zeros(green_plant.shape) 

    #whitening of area around well area
    v_shade_well = v_well.copy()
    v_shade_well[~mask_well.astype('bool')] = 1

    #sharpening of v_shade_well
    blurred_f = ndimage.gaussian_filter(v_shade_well, 1)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
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
    well_labels = measure.label(v_filtered_well)
    #Compute object properties
    well_properties = measure.regionprops(well_labels)
     
    #If there is any founded object in well 
    if(well_properties):

        #Computation of green area in detected objects/labels
        for i, prop in enumerate(well_properties):

            coordinates = np.where(well_labels == prop.label)

            prop.green_index = (sum(green_plant[coordinates]==1)/len(coordinates[0]))

        #find value of the most green object in well    
        green_index_max = max(well_properties, key=attrgetter('green_index')).green_index
      
        #compute given percentage of maximal green value 
        green_index_prop = max_percentage_level*green_index_max
        

        #If the value of the most green object is more than 5 percent
        if(green_index_max > green_index_minimum):

            #for each object
            for i, prop in enumerate(well_properties):
                
                #if object is not too small or big
                if(prop.area>plant_min_size and prop.area<plant_max_size):
                
                    #if green value of object is at least given percentage of most green object
                    if(prop.green_index > green_index_prop):

                        #Than pixels of this object is considered to be plant/seed in resulting mask
                        object_coordinates = np.where(well_labels == prop.label)
                        plant_mask[object_coordinates] = 1

                        
        #Corection of empty areas in plant/seed area

        #fill empty areas
        filledmask = als.imfill(plant_mask)
        #find holes
        holes = filledmask-plant_mask
        #find holes as a morphological objects
        holes_labels = measure.label(holes,connectivity=1)
        #Compute hole morphological propoerties
        holes_properties = measure.regionprops(holes_labels)

        #for each founded hole
        for i,prop in enumerate(holes_properties):

            #take hole's coordinates
            coordinates = (prop.coords[:,0],prop.coords[:,1])
            #compute median of area defined with hole coordinates in v-chanell of sharped well image
            hole_median = np.median(v_sharped_well[coordinates])

            #if area median is higher than 2/3 of threshold
            if(hole_median > (edge_thresh/1.5)):
                
                #and area is not too big
                if(plant_mask.shape[0]*plant_mask.shape[1]*hole_size_crit > prop.area):
                    
                    #hole area is considered to be plant
                    plant_mask[coordinates] = 1
                    
    return plant_mask



def well_bclosearea_search(mask_well, rgb_well, h_max=0.5, h_min=0.17, alpha = 25, plant_min_size=100, plant_max_size=80000, neighbor_distance=10, max_percentage_level=2/3, hole_size_crit=0.001):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'

    assert (h_max>h_min and h_max<1), 'Maximal value of h channel has to be lower than 1 and higher than h_min argument'
    assert (h_min>0 and h_min<h_max), 'Minimal value of h channel has to be lower than h_max argument and higher than 0'
    assert (type(alpha)==int), 'Alpha argument is integer'
    assert (type(plant_min_size)==int), 'Plant_min_size argument is integer'
    assert (type(neighbor_distance)==int), 'neighbor_distance argument value has to be integer'
    assert (max_percentage_level>0 and max_percentage_level<1), 'max_percentage_level argument value has to be lower than 1 argument and higher than 0'  
    assert (hole_size_crit>0 and hole_size_crit<1), 'Hole_size_crit argument value has to be lower than 1 argument and higher than 0'

    #image conversion to hsv color space (equation of conversion: https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
    hsv_well = rgb2hsv(rgb_well)

    #image conversion to lab color space
    lab_well = cv2.cvtColor(rgb_well, cv2.COLOR_BGR2LAB)

    #get hsv image channels
    h_well, s_well, v_well = cv2.split(hsv_well)

    #get hsv image channels
    l_well, a_well, b_well = cv2.split(lab_well)

    #segmentation of seed based on hue
    h_plant_well = ((h_well < h_max) & (h_well > h_min)).astype('uint8')

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
    well_labels = measure.label(v_filtered_well)
    #Compute object properties
    well_properties = measure.regionprops(well_labels)

    if(well_properties):
        
        #At this point method starts to be different of well_shade search
        #Method is based on Lab color space. It uses b-channel of Lab color space, at this channel is even really small green object visible.
        #This method helps to include small green objects within segmentation.

        
        #Computation of green area in detected objects/labels
        for i, prop in enumerate(well_properties):
            
            #find object coordinates
            coordinates = np.where(well_labels == prop.label)

            #Compute green proportion of each identified object in identified green plant  
            prop.green_index = (sum(green_plant[coordinates]==1)/len(coordinates[0]))
            
            #Compute b-channel median of identified object
            prop.b_med = np.median(b_well[coordinates])
            
            prop.neighbor = False
        
        #find b-channel median value of the most green object in well    
        b_med_max = max(well_properties, key=attrgetter('b_med')).b_med
        
        #find area size value of the most green object in well 
        area_size = max(well_properties, key=attrgetter('b_med')).area
        
        #If there is some green objetc
        if(b_med_max > 0 and area_size > plant_min_size and area_size < plant_max_size):

            #Define plant mask based on hsv color space v-channel
            plant_mask[well_labels==max(well_properties, key=attrgetter('b_med')).label] = 1

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
                neighborhood = distance < neighbor_distance
                #and remove  green plant area pixels
                neighborhood[plant_mask.astype('bool')] = False

                #for each identified object
                for i, prop in enumerate(well_properties):

                    #object coordinates
                    coordinates = (prop.coords[:,0],prop.coords[:,1])

                    #if object is in neignborhood
                    if(neighborhood[coordinates].any()):
                        #it is considered as a potentional plant
                        prop.neighbor = True    
                               
                neighbor_index = [prop.neighbor for prop in well_properties]
                
                #if there is not any potentional plants, stop while cycle with adding argument
                if(~any(neighbor_index)):

                    adding = False

                #If there are potentional plants
                else:

                    #Define mask for new objects
                    new_objects = np.zeros(plant_mask.shape)

                    #get potentional objects, which will be tested
                    tested_objects = [prop for prop in well_properties if prop.neighbor == True]
                    
                    #for each tested object
                    for i, prop in enumerate(tested_objects):
                       
                        #If object b-channel median is higher than given percentage of plant b-channel median
                        if((prop.b_med > max_percentage_level*b_med_max) & (prop.label not in added_labels)):
                            
                            #define mask of new object, which will be included in plant mask
                            new_objects[well_labels==prop.label] = 1
                            added_labels.append(prop.label)
                            
                    #If there are pixels, which are considered as a plant
                    if(new_objects.any()):
                        
                        #refine plant mask
                        plant_mask = cv2.bitwise_or(plant_mask, new_objects)
                        
                    else:

                        adding = False   


        #Corection of empty areas in plant/seed area

        #fill empty areas
        filledmask = als.imfill(plant_mask)
        #find holes
        holes = filledmask-plant_mask
        #find holes as a morphological objects
        holes_labels = measure.label(holes, connectivity=1)
        #Compute hole morphological properties
        holes_properties = measure.regionprops(holes_labels)

        #for each founded hole
        for i,prop in enumerate(holes_properties):

            #take hole's coordinate
            coordinates = (prop.coords[:,0],prop.coords[:,1])
            #compute median of area defined with hole coordinates in b-chanell well 
            hole_median = np.median(b_well[coordinates])

            #if area median is higher than 2/3 of threshold
            if(hole_median > (edge_thresh/1.5)):
                
                #and area is not too big
                if(plant_mask.shape[0]*plant_mask.shape[1]*hole_size_crit > prop.area):

                    #hole area is considered to be plant
                    plant_mask[coordinates] = 1  

    return plant_mask


def well_lab_search(mask_well, rgb_well, l_max=210, l_min=110, plant_min_size=100, plant_max_size=5000, plant_noise_diff=4):
    
    assert (type(rgb_well)  == np.ndarray), 'rgb_well has to be RGB image' 
    assert (len(rgb_well.shape) == 3),'rgb_well has to be RGB image'
    assert np.amin(rgb_well) >= 0 & np.amax(rgb_well) <= 255, 'rgb_well has to be RGB image'
    
    assert (type(mask_well)  == np.ndarray), 'mask_well has to be binary image' 
    assert (len(mask_well.shape) == 2), 'mask_well has to be binary image'
    assert (np.amin(mask_well) >= 0) & (np.amax(mask_well) <= 1), 'mask_well has to be binary image'

    assert (l_max>l_min and l_max<255), 'Maximal value of h channel has to be lower than 1 and higher than h_min argument'
    assert (l_min>0 and l_min<l_max), 'Minimal value of h channel has to be lower than h_max argument and higher than 0'
    assert (type(plant_noise_diff)==int), 'plant_noise_diff argument is integer'
    assert (type(plant_min_size)==int), 'Plant_min_size argument is integer'
    assert (type(plant_max_size)==int), 'Plant_max_size argument is integer'

    lab_well = cv2.cvtColor(rgb_well, cv2.COLOR_BGR2LAB)

    #get image channels
    l_well, a_well, b_well = cv2.split(lab_well)
    b_median = np.median(b_well)
    
    
    l_well_segm = ((l_well > l_min) & (l_well < l_max - (230-np.median(l_well)))).astype('uint8')
    
    #noise filtering from segmented image
    green_plant = median_filter(l_well_segm, 3)
    green_plant[~mask_well.astype('bool')] = 0
    
    
    #Find objects/labels in filtered well
    well_labels = measure.label(green_plant)
    #Compute object properties
    well_objects = measure.regionprops(well_labels)

    plant_mask = np.zeros(rgb_well.shape[0:2]) 
    
    #Check for 200 pixels minimum of object size
    for i, well_object in enumerate(well_objects):
        
        object_coordinates = np.where(well_labels == well_object.label)
        
        
        #Check for 200 pixels minimum of object
        if(len(object_coordinates[0]) > plant_min_size and len(object_coordinates[0]) < plant_max_size):
            
            #compute median of b-channel of given labeled object
            b_value = np.median(b_well[object_coordinates])

            if(abs(b_median-b_value) > plant_noise_diff):
 
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