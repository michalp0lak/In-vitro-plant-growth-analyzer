import cv2
import numpy as np
import pyzbar.pyzbar as pz
import datetime
import analysis_tools as als


def image_metadata_handler(filename):

    #define  output object
    class IMAGE_METADATA:
        
            def __init__(self, date, x_coordinate, y_coordinate, location):

                self.date = date
                self.x_coordinate = x_coordinate
                self.y_coordinate = y_coordinate
                self.location = location

    #split filename
    metadata = filename.split('_')
    #check if filename is in 6-component format
    assert len(metadata) == 6, 'Format of filename is wrong'
    
    if metadata[2] == 'date-0000-00-00-00-00-00': date = None

    else:

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


def roi_cropper(image, rg_product_thresh=0.15, b_thresh=100):
    
    assert (type(image)  == np.ndarray), 'Input data has to be RGB image' 
    assert (len(image.shape) == 3),'Input data has to be RGB image'
    assert np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    h,w = image.shape[0:2]

    #Cropping cental area of image in x-coordinate 
    cropped = image[:,500:2150,:]  

    #Multiplication of Green and Red channel (Tomas trick)
    product = np.double(cropped[:,:,1]) * np.double(cropped[:,:,2])

    #Standardization of grayscale image
    product = product - np.amin(product)
    product = product / np.amax(product)

    #Identification of blue lines (Tomas method) -> grayscale image
    blue_lines = (product < rg_product_thresh) & (cropped[:,:,0] > b_thresh)

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


    if np.absolute(np.min(peak_up)) > 100:
    
        #find up boundary as minimum of differences
        up_boundary = np.where(peak_up == np.amin(peak_up))

        #shift down boundary by 30 pixels
        up_boundary = up_boundary[0] + 30
    
    else:
    
        up_boundary = np.array([0]) + 30


    #down side of image
    down = row_sum[half2+1:]
    #differences of neighbouring rows
    peak_down = np.diff(down)

    if np.absolute(np.max(peak_down)) > 100:
    
        #find down boundary as maximum of differences
        down_boundary = np.where(peak_down == np.amax(peak_down))

        #shift down boundary
        down_boundary = half2 + down_boundary[0]
    
    else:
    
        down_boundary = h

    #cropp roi
    roi = cropped[up_boundary[0]:down_boundary[0],left_boundary[0]:right_boundary[0],:]

    return roi

def barcode_reader(image, equalization_constant=140, border_size=100, thresh_shift=20):
    
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
        code_area_sharp = als.sharp_image(code_area_gray)
        code_area_sharp = code_area_sharp.astype('uint8')
        
        # Otsu's thresholding
        thresh,code_area_thresholded = cv2.threshold(code_area_sharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        barcode_reader_output = pz.decode(code_area_thresholded)
        
        if(not barcode_reader_output):
            
            #Peter thresholding
            barcode_sharp = (code_area_sharp > thresh-thresh_shift).astype('uint8')*255
            
            #Make a border
            barcode_sharp = cv2.copyMakeBorder(barcode_sharp,top=border_size,bottom=border_size,left=border_size,right=border_size,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])
    
            barcode_reader_output = pz.decode(barcode_sharp)
        
            if(not barcode_reader_output):
    
                #histogram equalization of  sharpened grayscale barcode
                #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                code_area_equalized = clahe.apply(code_area_sharp)

                #Peter thresholding
                barcode_equalized = (code_area_equalized > equalization_constant).astype('uint8')*255

                #Make a border
                barcode_equalized = cv2.copyMakeBorder(barcode_equalized,top=border_size,bottom=border_size,left=border_size,right=border_size,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

                barcode_reader_output = pz.decode(barcode_sharp)
                
    barcode_data = None
    
    if(barcode_reader_output):
        
        barcode_data = barcode_reader_output[0][0].decode("UTF-8")     
    
    return barcode_data