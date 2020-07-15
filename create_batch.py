import global_variables
import os
from shutil import copyfile

def form_images(path):

    batch_path = path +'batch/'

    if(not os.path.exists(batch_path)):
        
        os.makedirs(batch_path)

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


            copyfile(image_path + 'image.png', batch_path + 'rgb_exp-' + exp_date + '_date-' + real_date + '_round-' + 
                      str(round_num) + '_tray-' + ID + '_cam-1.png')


if __name__ == '__main__':

    form_images(global_variables.path)