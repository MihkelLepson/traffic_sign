from os import listdir
import cv2
import sys
import pandas as pd
import os
import numpy as np
import csv

#This scripts extracts traffic signs and their labels from
#https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
#The file to download is GTSRB_Final_Training_Images.zip or GTSRB_Final_Test_Images.zip
#To use on training, the fourth argument when launching scripts must be "train".

#input_path should lead to folder, where are subfolders by image classes: 00000, 00001, 00002, ...
input_path0 = sys.argv[0]
output_path = sys.argv[1]
img_size = int(sys.argv[2])

#The annotation file contains only class_id. There's no file for class_id, meaning if we want to merge different
#datasets, we have to manually label them
def get_labels(i):
    if i == 0:
        return "PROHIBITORY", "20_SIGN"
    elif i == 1:
        return "PROHIBITORY", "30_SIGN"
    elif i == 2:
        return "PROHIBITORY", "50_SIGN"
    elif i == 3:
        return "PROHIBITORY", "60_SIGN"
    elif i == 4:
        return "PROHIBITORY", "70_SIGN"
    elif i == 5:
        return "PROHIBITORY", "80_SIGN"
    elif i == 6:
        return "PROHIBITORY", "80_SIGN_END"
    elif i == 7:
        return "PROHIBITORY", "100_SIGN"
    elif i == 8:
        return "PROHIBITORY", "120_SIGN"
    elif i == 9:
        return "PROHIBITORY", "NO_OVERTAKING"
    elif i == 10:
        return "PROHIBITORY", "NO_OVERTAKING_HEAVY"
    elif i == 11:
        return "WARNING", "CROSSROADS_WITH_MINOR"
    elif i == 12:
        return "PRIORITY", "PRIORITY_ROAD"
    elif i == 13:
        return "PRIORITY", "GIVE_WAY"
    elif i == 14:
        return "PRIORITY", "STOP"
    elif i == 15:
        return "PROHIBITORY", "NO_VECHILES"
    elif i == 16:
        return "PROHIBITORY", "NO_VECHILES_HEAVY"
    elif i == 17:
        return "PROHIBITORY", "NO_ENTRY"
    elif i == 18:
        return "WARNING", "DANGER"
    elif i == 19:
        return "WARNING", "CURVE_LEFT"
    elif i == 20:
        return "WARNING", "CURVE_RIGHT"
    elif i == 21:
        return "WARNING", "CURVES_FIRST_LEFT"
    elif i == 22:
        return "WARNING", "UNEVEN_SURFACE"
    elif i == 23:
        return "WARNING", "SLIPPERY_SURFACE"
    elif i == 24:
        return "WARNING", "ROAD_NARROWS_RIGHT"
    elif i == 25:
        return "WARNING", "ROADWORKS"
    elif i == 26:
        return "WARNING", "TRAFFIC_SIGNALS"
    elif i == 27:
        return "WARNING", "PEDESTRIANS"
    elif i == 28:
        return "WARNING", "CHILDREN"
    elif i == 29:
        return "WARNING", "CYCLISTS"
    elif i == 30:
        return "WARNING", "ICE_OR_SNOW"
    elif i == 31:
        return "WARNING", "WILD_ANIMALS"
    elif i == 32:
        return "PROHIBITORY", "ALL_END"
    elif i == 33:
        return "MANDATORY", "TURN_RIGHT"
    elif i == 34:
        return "MANDATORY", "TURN_LEFT"
    elif i == 35:
        return "MANDATORY", "GO_STRAIGHT"
    elif i == 36:
        return "MANDATORY", "GO_STRAIGHT_OR_TURN_RIGHT"
    elif i == 37:
        return "MANDATORY", "GO_STRAIGHT_OR_TURN_LEFT"
    elif i == 38:
        return "MANDATORY", "PASS_RIGHT_SIDE"
    elif i == 39:
        return "MANDATORY", "PASS_LEFT_SIDE"
    elif i == 40:
        return "MANDATORY", "ROUNDABOUT"
    elif i == 41:
        return "MANDATORY", "NO_OVERTAKING_END"
    elif i == 42:
        return "MANDATORY", "NO_OVERTAKING_HEAVY_END"
    elif i == 43:
        return "MANDATORY", "NO_OVERTAKING_HEAVY_END"
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
