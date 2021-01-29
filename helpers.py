import cv2
import numpy as np
import inspect
import pandas as pd

centroids = [(135,145), (135,380), (135,615), (135,850), (135,1085), (135,1330),
             (370,145), (370,380), (370,615), (370,850), (370,1085), (370,1330),
             (605,145), (605,380), (605,615), (605,850), (605,1085), (605,1330),
             (840,145), (840,380), (840,615), (840,850), (840,1085), (840,1330)]


def show_wells(roi, centroids):

    # Center coordinates 
    center_coordinates = (135, 145) 
    
    # Radius of circle 
    radius = 105  
    
    # Blue color in BGR 
    color = (255, 0, 0)   
    
    # Line thickness of 2 px 
    thickness = 2

    for center in centroids:
        
        # Using cv2.circle() method 
        # Draw a circle with blue line borders of thickness of 2 px 
        roi = cv2.circle(roi, center, radius, color, thickness)

   

    cv2.imshow('roi',roi)
    #cv2.imshow('roi',cv2.resize(roi,(roi.shape[1]//2, roi.shape[0]//2)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_well_mask(roi, centroids):

    mask = np.zeros(roi.shape)

    for center in centroids:

        cv2.circle(mask, center, 105, (255,255,255), -1)

    cv2.imwrite('C:/Users/polami05/Coding/Repositories/well_experiments/' + str(len(centroids)) + '.png',mask)


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def barcode_corrector(data):
    
    count_data = data.groupby(['location','barcode_data']).size().reset_index().rename(columns={0:'count'})
    
    for loc in count_data['location'].unique():
    
        location_data = count_data[count_data['location'] == loc]
        most_probable_barcode = location_data.loc[location_data ['count'].idxmax()]['barcode_data']
        
        if most_probable_barcode:

            indexes = data[data['location'] == loc].index
            data.loc[indexes, 'barcode_data'] = most_probable_barcode
            
    return data