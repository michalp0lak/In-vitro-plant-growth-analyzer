import time
import os, sys
import warnings
from multiprocessing import cpu_count
from multiprocessing import Pool
import pandas as pd
import numpy as np
import json
import global_variables
import helpers as hs
import processors as ps
import impreps as imps 
import segmentation as sg
import analysis_tools as at

warnings.filterwarnings("ignore")



if __name__ == '__main__':
    
    start_time = time.time()
    
    assert (type(global_variables.path)==str) & os.path.exists(global_variables.path), 'Path to folder with batch of images does not exist'
    assert (type(global_variables.row_num)==int), 'Number of rows has to be integer'
    assert (type(global_variables.col_num)==int), 'Number of columns has to be integer'

    well_num = global_variables.row_num*global_variables.col_num

    assert os.path.exists(global_variables.masks_path + str(well_num) +'.png'), 'Mask does not exist'
    
    input_path = global_variables.path + 'batch/'
    output_path = global_variables.path + 'results/'
    temp_path = output_path + 'temp/'
    
    if(not os.path.exists(output_path)):

        os.makedirs(output_path)

    if(not os.path.exists(temp_path)):

        os.makedirs(temp_path)
    
    formats = ('.JPG','.jpg','.PNG','.png','.bmp','.BMP','.TIFF','.tiff','.TIF','.tif')
    

    try:

        files = [file for file in os.listdir(input_path) if file.endswith(formats)]

        procs = cpu_count()
        procIDs = list(range(0, procs))

        numImagesPerProc = len(files) / float(procs)
        numImagesPerProc = int(np.ceil(numImagesPerProc))
            
        chunkedPaths = list(hs.chunk(files, numImagesPerProc))
        
        # initialize the list of payloads
        imageLoads = []

        # loop over the set chunked image paths
        for (i, fileNames) in enumerate(chunkedPaths):

            # construct a dictionary of data for the payload, then add it
            # to the payloads list
            data = {
                "id": i,
                "files_names": fileNames,
                "input_path": input_path,
                "output_path": output_path,
                "temp_path": temp_path
            }
            imageLoads.append(data)
            
        structured_data = []
        
        # construct and launch the processing pool
        print("[INFO] launching pool using {} processes.".format(procs))
        print("[INFO] All CPU capacity is used for data analysis. You won't be able to use your computer for any other actions.")

        pool = Pool(processes=procs)
        pool.map(ps.process_images, imageLoads)
        pool.close()
        pool.join()

        print("[INFO] Pool of processes was closed")
        print("[INFO] Aggregating partial results into structured data set.")

        xlsx_files = [file for file in os.listdir(temp_path) if file.endswith('xlsx')]
        txt_files = [file for file in os.listdir(temp_path) if file.endswith('txt')]

        frames = []

        for xlsx in xlsx_files:
            
            frames.append(pd.read_excel(temp_path + xlsx, engine='openpyxl'))
            
        structured_result = pd.concat(frames, ignore_index=True)
        structured_result = structured_result[['filename', 'date', 'time', 'location', 'x_coordinate', 'y_coordinate', 'plant_id', 'barcode_data','well_row','well_column','pixel_num','r_mean', 'g_mean', 'b_mean']]
        structured_result = hs.barcode_corrector(structured_result)

        structured_result.sort_values(by=['location', 'date', 'well_row','well_column'])
        structured_result.to_excel(output_path + 'exp_result.xlsx')


        with open(output_path + 'failures.txt', 'w') as outfile:
            for fname in txt_files:
                with open(temp_path+fname) as infile:
                    for line in infile:
                        outfile.write(line)
                        
        files = [file for file in os.listdir(temp_path)]
                                             
        for f in files:
            os.remove(temp_path+f)
            
        os.rmdir(temp_path)


        analysis_configuration = {'version': global_variables.version, 'plate': well_num,'image_metadata_handler':hs.get_default_args(imps.image_metadata_handler), 'roi_cropper':hs.get_default_args(imps.roi_cropper),
                                  'barcode_reader':hs.get_default_args(imps.barcode_reader), 'well_former':hs.get_default_args(at.well_former), 
                                  'well_shade_search':hs.get_default_args(sg.well_shade_search), 'well_bclosearea_search':hs.get_default_args(sg.well_bclosearea_search),
                                  'alghoritm_comparison':hs.get_default_args(sg.alghoritm_comparison), 'well_processor':hs.get_default_args(ps.well_processor),
                                  'image_processore':hs.get_default_args(ps.image_processor),'sharp_image':hs.get_default_args(at.sharp_image), 'imfill':hs.get_default_args(at.imfill),
                                  'bwperim':hs.get_default_args(at.bwperim)}

        with open(output_path + "analysis_config.json", "w") as config_file:
            json.dump(analysis_configuration, config_file)

        print("[INFO] ANALYSIS WAS FINISHED")

    except Exception as e:
        
        exception_type, exception_object, exception_traceback = sys.exc_info()

        filename = exception_traceback.tb_frame.f_code.co_filename

        line_number = exception_traceback.tb_lineno
        
        print('{} - {}: {}'.format(filename, line_number, exception_traceback))
        
    print("--- %s seconds ---" % (time.time() - start_time))