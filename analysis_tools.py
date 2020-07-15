import cv2
from scipy.ndimage.filters import median_filter
import numpy as np
import itertools


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


    row_sort = sorted(mask_properties, key=lambda area: area.centroid[0])
    col_sort = sorted(mask_properties, key=lambda area: area.centroid[1])

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

def sharp_image(image, sigma=10, power=0.4, method='laplace'):

    
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


def imfill(image, low_thresh=220, up_thresh=255):
    
    #Check of input arguments correctness
    assert (type(image)  == np.ndarray), 'Input has to be binary image' 
    assert (len(image.shape) == 2), 'Input has to be binary image'
    assert (np.amin(image) >= 0) & (np.amax(image) <= 1), 'Input has to be binary image'

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(image, low_thresh, up_thresh, cv2.THRESH_BINARY_INV)
    im_th = im_th.astype('uint8')

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels bigger than the image.
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