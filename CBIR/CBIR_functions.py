# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:25:49 2024

@author: Louise
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.color as skc
from skimage import io
import cv2
import scipy.spatial.distance as dist
import math

import os
import glob
import csv

from PIL import Image

import shutil

def open_image(path):
    """
    path : path to the image
    Returns the image
    """
    
    Image = skio.imread(path)
    return Image

def print_image(Image):
    """
    path : path to the image
    Print the image
    """
    
    plt.figure(figsize=(7,6))
    plt.imshow(Image)
    plt.title('Original Image')

def mean_HSV(path_directory):
    """
    Compute the mean of each channel of HSV
    Return the 3 means corresponding to the 3 channels
    """
    #Search the files with a png extension
    data_path_img = os.path.join(path_directory,'*png')
    img_files = glob.glob(data_path_img)
    
    mean_H_list, mean_S_list, mean_V_list = [], [], []
    
    for i in range(len(img_files)):
        
        # img[i] = open_image(img_files[i])
        
        HSV_version = skc.rgb2hsv(open_image(img_files[i]))
        
        mean_H = np.mean(HSV_version[:,:,0])
        mean_S = np.mean(HSV_version[:,:,1])
        mean_V = np.mean(HSV_version[:,:,2])
        
        mean_H_list.append(mean_H)
        mean_S_list.append(mean_S)
        mean_V_list.append(mean_V)

    return mean_H_list, mean_S_list, mean_V_list 

def gabor_features(img_dir, file_directory):
    """
    img_dir : enter the path to the directory where all 
    the images are stored
    
    file_directory : path where you want to store the an csv file containing 
    the results

    Create a csv file with 50 Gabor features for each image
    
    #Features extracted from this are "Local Energy" and "Mean Amplitude" 
    at different angles and wavelengths (frequencies)
    Number of features extracted = number of angles chosen * number of wavelengths chosen
    
    https://github.com/Rohit-Kundu/Traditional-Feature-Extraction/blob/main/Gabor.py
    
    Returns the name of the file with the data of the Gabor features
    """
    #Importing the images
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)

    eo=len(files)

    img = []
    image_name = []
    for f1 in files:
        data = cv2.imread(f1)
        img.append(data)
        image_name.append(os.path.basename(f1))

    gamma=0.5
    sigma=0.56
    theta_list=[0, np.pi, np.pi/2, np.pi/4, 3*np.pi/4] #Angles
    phi=0
    lamda_list=[2*np.pi/1, 2*np.pi/2, 2*np.pi/3, 2*np.pi/4, 2*np.pi/5] #wavelengths
    num=1

    # Creating headings for the csv file
    gabor_label=['Image Name']
    for i in range(50):
        gabor_label.append('Gabor'+str(i+1))

    # checking if the directory where you want to store Gabor.csv exist or not. 
    if not os.path.exists(file_directory): 
       #if the directory is not present then create it. 
       os.makedirs(file_directory) 
    
    output_csv_path = os.path.join(file_directory, 'dataset.csv')
    with open(output_csv_path,'w+',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(gabor_label)
        
        for i in range(eo):
            img[i] = cv2.cvtColor(img[i] , cv2.COLOR_BGR2GRAY)
            local_energy_list=[]
            mean_ampl_list=[]
            
            for theta in theta_list:
                for lamda in lamda_list:
                    kernel=cv2.getGaborKernel((3,3),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)
                    fimage = cv2.filter2D(img[i], cv2.CV_8UC3, kernel)
                    
                    mean_ampl=np.sum(abs(fimage))
                    mean_ampl_list.append(mean_ampl)
                    
                    local_energy=np.sum(fimage**2)
                    local_energy_list.append(local_energy)
                    
                    num+=1
            writer.writerow([image_name[i]]+local_energy_list+mean_ampl_list)
            
    return output_csv_path

def add_data_csv(file_path, new_data):
    """
    file_path : path towards the existing CSV file
    new_data : Dictionnary containing the new data to add where the keys
    are the names of the columns and the values are the lists of data
    
    Add new datas at the end of the existing columns in the csv file
    """
    #Read the existing data
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
   
    headers = rows[0]
   
    num_rows = len(rows) - 1  #We don't consider the first row (name of image)
    for header, data in new_data.items():
        if len(data) < num_rows:
            #Complete the missing data with None
            new_data[header].extend([None] * (num_rows - len(data)))
        elif len(data) > num_rows:
            raise ValueError(f"La longueur des nouvelles données pour la colonne '{header}' ({len(data)}) dépasse le nombre de lignes dans le fichier CSV ({num_rows}).")

    #Add the new data to the corresponding columns
    for header in headers:
        if header in new_data:
            for i, row in enumerate(rows[1:], start=0):
                row.append(new_data[header][i])
   
    # Add columns for the new data 
    for header, data in new_data.items():
        if header not in headers:
            headers.append(header)
            for i, row in enumerate(rows[1:], start=0):
                row.append(data[i])
   
    #Update the csv file with the new data
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def read_data(txtFile):
    """
    txtFile : file containing info on the bounding box
    
    Return a list the differents informations that
    was in the txt file
    
    In order of apperance :
    [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, 
     subject, class number]
    """
    with open(txtFile, "r") as file :
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def extract_data(lines, delimiter = ' '):
    """
    lines : lines from which you want to extract the data
    delimiter : The default is ' '.

    Return data that is a list
    """
    data = []
    for line in lines :
        values = line.split(delimiter)
        data.append(values)
    return data

def save_image(image, directory, base_name,counter):
    """
    image : image that you want to save
    directory : directory where you want to save
    base_name : under which name you want to save the image
    counter : under which numero you want to save the image
    
    Saves the image according to the different parameters entered
    """
    
    filename = f"{base_name}_{counter}.png"
    complete_path = os.path.join(directory, filename)
    io.imsave(complete_path, image)
    
def zone_of_interest(path_directory):
    """
    path_directory : path where the image and txt folder are
    
    Crop the initial image, the cropped image contains only Nephrops
    or NephropsSystem
    These image are saved in a "Crop" folder
    """
   
    img_dir = os.path.join(path_directory,'images')
    label_dir = os.path.join(path_directory,'labelTxt' )
    
    saved_img = os.path.join(path_directory,'dataset/cropElements')
    
    #Search the files with a png extension
    data_path_img = os.path.join(img_dir,'*png')
    
    #Search the files with a txt extension
    data_path_label = os.path.join(label_dir,'*txt')
    
    img_files = glob.glob(data_path_img)
    txt_files = glob.glob(data_path_label)
    
    c1 = 0
    c2 = 0
    
    # Normalize image
    for i in range(len(img_files)):
        #We suppose that each image has a corresponding txt file
        content = read_data(txt_files[i])
        data = extract_data(content)
        for k in range(len(data)):
            xmin = round(float(data[k][0]))
            ymin = round(float(data[k][1]))
            xmax = round(float(data[k][2]))
            ymax = round(float(data[k][5]))
            
            # temp = open_image(img_files[i])
            # crop = temp[ymin:ymax,xmin:xmax,:]
            
            temp = open_image(img_files[i])
            normalized_image = cv2.normalize(temp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            crop = normalized_image[ymin:ymax,xmin:xmax,:]
            
            if(data[k][8] == 'Nephrops'):
                save_image(crop, saved_img, 'Nephrops', c1)
                c1 += 1
                
            #If we want also to work on NephropsSystem   
            # elif(data[k][8] == 'NephropsSystem'): 
            #     save_image(crop, saved_img, 'NephropsSystem', c2)
            #     c2 += 1

def cut_image(image, saved_directory):
    """
    image : image that you want to cut into rectangles

    Returns xmin_list, xmax_list, ymin_list, ymax_list that are
    the list of coordinates of the differents rectangles that compose 
    the image. Here, we are only focusing on the bottom half of the image
    """
    height, width, channels = image.shape
    #image_new = image[int(height/2):,int(width/15):int((14*width)/15),:]
    image_new = image
    height, width, channels = image_new.shape
    width_divider = 16 #To change the size of the rectangles
    height_divider = 12 #To change the size of the rectangles

    crop_width = math.floor(width / width_divider)
    crop_height = math.floor(height / height_divider)

    c1 = 0 #If you want to save the different rectangles 
    
    xmin_list, xmax_list, ymin_list, ymax_list = [],[],[],[]

    step = 0.25
    
    for i in np.arange(0,width_divider,step):
        for j in np.arange(0,height_divider,step):
            xmin = math.floor(i * crop_width)
            xmax = math.floor((i + 1) * crop_width)
            ymin = math.floor(j * crop_height)
            ymax = math.floor((j + 1) * crop_height)
            
            xmin_list.append(xmin)
            xmax_list.append(xmax)
            ymin_list.append(ymin)
            ymax_list.append(ymax)
            
            #If you want to save the different rectangles 
            # Crop the image
            crop = image_new[ymin:ymax, xmin:xmax]

            # Save the cropped image
            save_image(crop, saved_directory, 'part', c1)
            c1 += 1
    
    return xmin_list, xmax_list, ymin_list, ymax_list

def resize_dataset(path_dataset):
    """
    path_dataset : path towards the folder where we want to resize the images

    Returns saved_directory : path where the resized image were saved
    """
    
    data_path_img = os.path.join(path_dataset,'*png')
    
    img_files = glob.glob(data_path_img)
    
    saved_directory = os.path.join(os.path.dirname(path_dataset),'resizedDataset')
    
    c1 = 0
    
    for i in range(len(img_files)):

        # Load the image
        image = cv2.imread(img_files[i])
        
        # Define the desired dimensions for the resized image
        width = 64
        height = 72
        
        # Resize the image
        resized_image = cv2.resize(image, (width, height))
    
        if not os.path.exists(saved_directory): 
            #if the dataset directory is not present then create it. 
            os.makedirs(saved_directory)
        
        filename = 'resizedImage'  + str(c1) + '.png'
        # Save the resized image
        path = os.path.join(saved_directory , filename)
        cv2.imwrite(path, resized_image)
        c1 += 1
    
    return saved_directory

def create_dataset(path_directory):
    """
    path_directory : path with the image of the database
    
    Create a dataset store in a csv file

    Returns file_path : path toward the csv file
    """
    dataset_path = os.path.join(path_directory,'dataset')
    print(dataset_path)
    
    # checking if the directory dataset exist or not. 
    if not os.path.exists(dataset_path): 
       #if the dataset directory is not present then create it. 
       os.makedirs(dataset_path) 

    cropElements_path = os.path.join(dataset_path,'cropElements')
    print(cropElements_path)
    # checking if the directory cropElements exist or not. 
    if not os.path.exists(cropElements_path): 
       #if the cropElements directory is not present then create it. 
       os.makedirs(cropElements_path) 

    zone_of_interest(path_directory)
    
    # resize_dataset(cropElements_path)
    
    # resized_path = os.path.join(dataset_path,'resizedDataset')
    # if not os.path.exists(resized_path): 
    #    #if the resized_path directory is not present then create it. 
    #    os.makedirs(resized_path)
    # print(resized_path)
    
    onlySelection_path = os.path.join(dataset_path,'onlySelection')
    
    
    file_path = gabor_features(onlySelection_path, dataset_path)
    print(file_path)
    
    H,S,V = mean_HSV(onlySelection_path)
    
    new_data = {
        # 'H mean': H,
        'S mean': S,
        'V mean': V
    }

    add_data_csv(file_path, new_data)
    
    return file_path

def analyze_image(image_path):
    """
    image_path : path towards the image that we want to analyze

    Returns file_path : csv file with the different features (Gabor + HSV)
    """
    
    image1 = open_image(image_path)
    
    image = cv2.normalize(image1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    saved_directory = os.path.join(os.getcwd(),'DividedImages')
    if not os.path.exists(saved_directory): 
        #if the dataset directory is not present then create it. 
        os.makedirs(saved_directory)
    
    xmin, xmax, ymin, ymax = cut_image(image, saved_directory)
    
    file_path = gabor_features(saved_directory, saved_directory)
    
    H,S,V = mean_HSV(saved_directory)
    
    new_data = {
        # 'H mean': H,
        'S mean': S,
        'V mean': V,
        'xmin' : xmin,
        'xmax' : xmax,
        'ymin' : ymin,
        'ymax' : ymax
    }

    add_data_csv(file_path, new_data)
    
    return file_path

def find_similar(dataset_file, image_path, threshold, weight_factor=10):
    """
    dataset_file : path towards the csv containing the features of the reference photos
    image_path : path towards the csv containing the features of the image that we want to analyze
    threshold : threshold for the minimum distance
    weight_factor : factor to weight the last two columns more

    Returns below_threshold_indices : index of the image that are below the threshold
    """
    with open(dataset_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_dataset = list(reader)
       
    with open(image_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_image = list(reader)
   
    below_threshold_indices = []
   
    for ImgNbr in range(1, len(rows_image)):
        for i in range(1, len(rows_dataset)):
            total_distance = 0.0
            weight_sum = 0.0
           
            for j in range(1, len(rows_dataset[i])):
                dataset_features = float(rows_dataset[i][j])
                image_features = float(rows_image[ImgNbr][j])
                features_distance = dist.euclidean([image_features], [dataset_features])
               
                # Apply weight factor to the last two columns
                if j >= len(rows_dataset[i]) - 2:
                    weight = weight_factor
                else:
                    weight = 1.0
                
                weight = 1.0
               
                total_distance += weight * features_distance
                weight_sum += weight
           
            mean_distance = total_distance / weight_sum

            # Check if mean distance is below the threshold
            if mean_distance < threshold:
                below_threshold_indices.append(ImgNbr)
                break  # No need to check further if one dataset image is already below threshold
   
    return below_threshold_indices
 
def draw_bb(image_path, coordinates_list, output_path):
    """
    image_path : path towards the query image
    coordinates_list : a list of a list containing coordinates of the 
    rectangles as [xmin, xmax, ymin, ymax]
    output_path : path where you want to save the modified image 
    
    Draw one or multiple rectangle on an image
    Save the modified image
    """
    # Lire l'image en mode couleur
    image = cv2.imread(image_path)
   
    # Vérifier si l'image a été correctement lue
    if image is None:
        print(f"Erreur : Impossible de lire l'image depuis le chemin : {image_path}")
        return
   
    # Définir la couleur et l'épaisseur du rectangle
    color = (0, 0, 255)  # Rouge en BGR
    thickness = 2  # Épaisseur de la ligne du rectangle
   
    # Dessiner tous les rectangles sur l'image
    for coords in coordinates_list:
        if len(coords) == 4:
            xmin, xmax, ymin, ymax = coords
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        else:
            print(f"Coordonnées incorrectes : {coords}")
    # Enregistrer l'image modifiée
    cv2.imwrite(output_path, image)
    print(f"Image enregistrée avec succès à : {output_path}")
    
    return output_path
   
    # # Print the rectangles
    # cv2.imshow('Image avec rectangles', image)
    # cv2.waitKey(0)  # Attendre une pression de touche pour fermer la fenêtre
    # cv2.destroyAllWindows()  # Fermer toutes les fenêtres OpenCV

def get_coordinates(index, image_path):
    """
    index : index of the images that we want to get the path
    image_path : path towards the csv file containing the data
    
    Returns coordinates_int : a list containing the coordinates
    """
    with open(image_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_image = list(reader)
        
    tot_coord = []
    
    for i in index:
        image_features = (rows_image[i][:])
        coordinates = image_features[-4:]
        tot_coord.append(coordinates)

    coordinates_int = [[int(value) for value in sublist] for sublist in tot_coord]

    return coordinates_int

def filter_images(input_folder, output_folder, min_width, max_width, min_height, max_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
   
    #Go through all the files in the input folder
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
       
        #Check if the file is an image
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                # Check if the image is between the given dimension
                if min_width <= width <= max_width and min_height <= height <= max_height:
                    # Copy the image in the output folder
                    shutil.copy(filepath, os.path.join(output_folder, filename))
        except IOError:
            # If the file isn't an image, continue
            continue
