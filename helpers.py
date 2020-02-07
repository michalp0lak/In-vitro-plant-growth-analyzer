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



path = 'D:/Data_projects/2020-01-18--12-55-15/'

def form_images(path)

    batch_path = path +'batch/'
    rounds = [file for file in os.listdir(path + 'rounds/')]

    time_stream = open(path + "start-date.txt", "r")
    time = time_stream.readline()
    spl = time.split(' ')

    date = spl[0].split('/')
    Time = spl[1].split(':')
    exp_date = date[2] + '-' + date[0] + '-' + date[1] + '-' + Time[0] + '-' + Time[1] + '-' + Time[2]


    for Round in rounds:

        round_path = path + 'rounds/' + Round +  '/'
        round_num = int(Round)

        tray_id = list()

        with open(round_path + "id.txt", "r") as f:

            for line in f:
                sp = line.split('\t')
                tray_id.append((int(sp[0]),sp[1].split('\n')[0]))

        trays_path = round_path + 'trays/'
        trays = [file for file in os.listdir(trays_path)]

        for tray in trays:

            image_path = trays_path + tray + '/' + 'rgbs/' + '01/'

            time_stream = open(image_path + "start-date.txt", "r")
            time = time_stream.readline()
            spl = time.split(' ')

            date = spl[0].split('/')
            Time = spl[1].split(':')
            real_date = date[2] + '-' + date[0] + '-' + date[1] + '-' + Time[0] + '-' + Time[1] + '-' + Time[2]

            ID = [item[1] for item in tray_id if item[0] == int(tray)][0]

            os.rename(image_path + 'image.png', batch_path + 'rgb_exp-' + exp_date + '_date-' + real_date + '_round-' + 
                      str(round_num) + '_tray-' + ID + '_cam-1.png')