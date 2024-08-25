# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:18:44 2024

@author: Louise
"""
#Example of the initial coordinates obtained with VoTT

# # 182.0 174.0 228.0 174.0 228.0 228.0 182.0 228.0 Nephrops 0
# # 117.0 194.0 150.0 194.0 150.0 235.0 117.0 235.0 Nephrops 0
# # 96.0 165.0 237.0 165.0 237.0 252.0 96.0 252.0 NephropsSystem 0

        
import os
import glob

# Change path
txtFile = "data/labels/train/St05T00.92.txt"
txtPath = "data/labels/val"

def read_data(txtFile):
    """
    Read data from the txt file
    """
    with open(txtFile, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def extract_data(lines, delimiter=' '):
    """
    Extract data from each lines
    """
    data = []
    for line in lines:
        values = line.split(delimiter)
        data.append(values)
    return data

def arrange_data(data, txtFile, img_width, img_height):
    """
    Write the data according to the YOLO format 
    """
    print(f"Processing file: {os.path.basename(txtFile)}")
   
    with open(txtFile, "w") as f:
        for j in range(len(data)):
            xmin = round(float(data[j][0]))
            ymin = round(float(data[j][1]))
            xmax = round(float(data[j][2]))
            ymax = round(float(data[j][5]))
           
            xcenter = xmin + (xmax - xmin) / 2
            width = (xmax - xmin)
           
            ycenter = ymin + (ymax - ymin) / 2
            height = (ymax - ymin)
           
            # Normaliser les coordonnées
            xcenter /= img_width
            width /= img_width
            ycenter /= img_height
            height /= img_height

            if data[j][8] == 'Nephrops':
                label = 0
                
            if data[j][8] == 'NephropsSystem':
                label = 1
           
            text = f"{label} {xcenter} {ycenter} {width} {height}"
           
            if j < len(data) - 1:
                text += "\n"
           
            f.write(text)

def save_data(txtPath, img_width, img_height):
    """
    Read the txt file from the folder and write them all 
    according to the YOLO format  
    """
    txt_files = glob.glob(os.path.join(txtPath, '*.txt'))
    print(f"Found files: {txt_files}")
   
    for txt_file in txt_files:
        content = read_data(txt_file)
        data = extract_data(content)
        arrange_data(data, txt_file, img_width, img_height)

def remove_lines_starting_with_one(txtPath):
    """
    Remove lines that correspond to the Nephrops Systems
    """
    # Récupère tous les fichiers .txt dans le dossier spécifié
    txt_files = glob.glob(os.path.join(txtPath, '*.txt'))
    print(f"Found files: {txt_files}")
   
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
       
        # Filtrer les lignes qui ne commencent pas par '1'
        filtered_lines = [line for line in lines if not line.strip().startswith('1')]
       
        # Écrire les lignes filtrées dans le fichier
        with open(txt_file, 'w') as file:
            file.writelines(filtered_lines)
        print(f"Processed file: {txt_file}")


#Define the size of the image (modify as needed) (useful for the normalization)
image_width = 768
image_height = 432

#Format the data
save_data(txtPath, image_width, image_height)

# Remove the lines that consider the Nephrops burrows
remove_lines_starting_with_one(txtPath)