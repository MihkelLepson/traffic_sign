from os import listdir
import cv2
import sys
import pandas as pd
import os
import numpy as np
import csv

#This scripts extracts traffic signs and their labels from
#https://btsd.ethz.ch/shareddata/
#The file to download is BelgiumTSC_Training (171.3MBytes)

#input_path should lead to folder, where are subfolders by image classes: 00000, 00001, 00002, ...
input_path0 = sys.argv[0]
output_path = sys.argv[1]
img_size = int(sys.argv[2])

#We want to merge different datasets so we have to manually label them to match previous
def get_labels(i):
    if i == 0:
        return "WARNING", "UNEVEN_SURFACE"
    elif i == 1:
        return "WARNING", "BUMP"
    elif i == 1:
        return "WARNING", "SLIPPERY_SURFACE"
    elif i == 3:
        return "WARNING", "CURVE_LEFT"
    elif i == 4:
        return "WARNING", "CURVE_RIGHT"
    elif i == 5:
        return "WARNING", "CURVES_FIRST_LEFT"
    elif i == 6:
        return "WARNING", "CURVES_FIRST_RIGHT"
    elif i == 7:
        return "WARNING", "CHILDREN"
    elif i == 8:
        return "WARNING", "CYCLISTS"
    elif i == 9:
        return "WARNING", "DOMESTIC_ANIMALS"
    elif i == 10:
        return "WARNING", "ROADWORKS"
    elif i == 11:
        return "WARNING", "TRAFFIC_SIGNALS"
    elif i == 12:
        return "WARNING", "LEVEL_CROSSING_WITH_BARRIERS_AHEAD"
    elif i == 13:
        return "WARNING", "DANGER"
    elif i == 14:
        return "WARNING", "ROAD_NARROWS"
    elif i == 15:
        return "WARNING", "ROAD_NARROWS_LEFT"
    elif i == 16:
        return "WARNING", "ROAD_NARROWS_RIGHT"
    elif i == 17:
        return "WARNING", "CROSSROADS_WITH_MINOR"
    elif i == 18:
        return "WARNING", "CROSSROADS_PRIORITY_RIGHT"
    elif i == 19:
        return "WARNING", "GIVE_WAY"
    elif i == 20:
        return "PRIORITY", "GIVE_WAY_TO_ONCOMING"
    elif i == 21:
        return "PRIORITY", "STOP"
    elif i == 22:
        return "PROHIBITORY", "NO_ENTRY"
    elif i == 23:
        return "PROHIBITORY", "NO_PEDAL_CYCLES"
    elif i == 24:
        return "PROHIBITORY", "WEIGHT_LIMIT"
    elif i == 25:
        return "PROHIBITORY", "NO_VECHILES_HEAVY"
    elif i == 26:
        return "PROHIBITORY", "WIDTH_LIMIT"
    elif i == 27:
        return "PROHIBITORY", "HEIGHT_LIMIT"
    elif i == 28:
        return "PROHIBITORY", "NO_VECHILES"
    elif i == 29:
        return "PROHIBITORY", "NO_LEFT_TURN"
    elif i == 30:
        return "PROHIBITORY", "NO_RIGHT_TURN"
    elif i == 31:
        return "PROHIBITORY", "NO_OVERTAKING"
    elif i == 32:
        return "PROHIBITORY", "?_SIGN"
    elif i == 33:
        return "MANDATORY", "SHARED_PEDESTRIAN_AND_CYCLE_PATH"
    elif i == 34:
        return "MANDATORY", "GO_STRAIGHT"
    elif i == 35:
        return "MANDATORY", "GO_LEFT"
    elif i == 36:
        return "MANDATORY", "GO_STRAIGHT_OR_TURN_RIGHT"
    elif i == 37:
        return "MANDATORY", "ROUNDABOUT"
    elif i == 38:
        return "MANDATORY", "CYCLE_PATH"
    elif i == 39:
        return "MANDATORY", "SEGREGATED_PEDESTRIAN_AND_CYCLE_PATH"
    elif i == 40:
        return "PROHIBITORY", "NO_PARKING"
    elif i == 41:
        return "MANDATORY", "NO_STOPPING_NO_STANDING"
    elif i == 42:
        return "MANDATORY", "NO_PARKING_FROM_1ST_TO_15TH"
    elif i == 43:
        return "MANDATORY", "NO_PARKING_FROM_16TH_TO_31ST"
    elif i == 44:
        return "PRIORITY", "PRIORITY_OVER_ONCOMING"
    elif i == 45:
        return "INDICATION", "PARKING"
    elif i == 46:
        return "INDICATION", "PARKING_RESERVED_FOR_DISABLED"
    elif i == 47:
        return "INDICATION", "PARKING_RESERVED_CARS"
    elif i == 48:
        return "INDICATION", "PARKING_RESERVED_TRUCKS"
    elif i == 49:
        return "INDICATION", "PARKING_RESERVED_COACHES"
    elif i == 50:
        return "INDICATION", "PARKING_ON_VERGE_OR_SIDEWALK"
    elif i == 51:
        return "SPECIAL_REGULATIONS", "LIVING_STREET"
    elif i == 52:
        return "SPECIAL_REGULATIONS", "LIVING_STREET_END"
    elif i == 53:
        return "SPECIAL_REGULATIONS", "ONE_WAY_STREET"
    elif i == 54:
        return "INDICATION", "NO_THROUGH_ROAD"
    elif i == 55:
        return "INDICATION", "END_OF_CONSTRUCTION"
    elif i == 56:
        return "SPECIAL_REGULATIONS", "PEDESTRIAN_CROSSING"
    elif i == 57:
        return "SPECIAL_REGULATIONS", "CYCLIST_CROSSING"
    elif i == 58:
        return "INDICATION", "PARKING_LOT"
    elif i == 59:
        return "SPECIAL_REGULATIONS", "BUMP"
    elif i == 60:
        return "PRIORITY", "PRIORITY_ROAD_END"
    elif i == 61:
        return "PRIORITY", "PRIORITY_ROAD"
    elif i == 62:
        return "PRIORITY", "PRIORITY_ROAD"
    else:
        return "MISC", "MISC"
		
		
imgs = []
label_type = []
label_sign = []

for i in range(43):
    images = {}
    if "train" == "train": # xd
        if i >= 10:
            folder = '000' + str(i) + '/'
        else:
            folder = '0000' + str(i) + '/'
        input_path = input_path0 + folder
    else:
        input_path = input_path0
    images_files = [f for f in listdir(input_path0 + folder)]
    for image_file in images_files:
        if image_file[-3:] != 'csv':
            images[image_file] = (cv2.imread(input_path0 + folder + image_file))

    with open(input_path0 + folder + 'GT-' + folder.replace('/','') + '.csv') as f:
        f.readline()  #First line is column names
        line = f.readline()
        while line != '':
            x = line.split(";")

            img = images[x[0]][int(x[4]):int(x[6]),int(x[3]):int(x[5]),:]
            lbl1, lbl2 = get_labels(int(x[7]))
            imgs.append(cv2.resize(img, [img_size,img_size]))
            label_type.append(lbl1)
            label_sign.append(lbl2)
            line = f.readline()
        f.close()
				
data = np.array([label_type,label_sign])
data = np.transpose(data)
df = pd.DataFrame(data=data, columns=["type", "sign"])

label_path = []

if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(len(imgs)):
    if not os.path.exists(output_path):
        os.makedirs(output_path + 'dataset/')
    cv2.imwrite(output_path + 'dataset/' + 'img' + str(i) + '.jpg' ,imgs[i])
    label_path.append(output_path + 'dataset/' + 'img' + str(i) + '.jpg')
    
df['img_path'] = label_path
df.to_pickle(output_path + 'df.pkl')
