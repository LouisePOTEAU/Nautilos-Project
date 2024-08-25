# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:07:51 2024

@author: Louise
"""

from CBIR_functions import *
import time

def main():
    plt.close('all')
    
    #####
    #MODIFY PATH & QUERY IMAGE 
    
    path_directory = 'DatabaseTot'
    dataset_file = 'DatabaseTot/dataset/dataset.csv'
    file_image_path = 'DividedImages/dataset.csv'

    # path_directory = 'GT-IRBIM-v2-full_LESS'
    # dataset_file = 'GT-IRBIM-v2-full_LESS/dataset/dataset.csv'
    # file_image_path = 'DividedImages/dataset.csv'

    threshold = 3000 # Distance threshold
    
    # image_path = 'DatabaseTot/images/St05T01.92.png'
    
    NumImg = '2N_3'
    image_path = 'Tests/' + NumImg + '.png'
    
    #####
    
    # For a first time use you have to create the dataset
    # For the other times, comment the line below
    create_dataset(path_directory)
    
    
    analyze_image(image_path)

    index = find_similar(dataset_file, file_image_path, threshold)
    
    coord = get_coordinates(index, file_image_path)
    
    output_path = 'St05T00.5_with_rectangles.png'
    draw_bb(image_path, coord, output_path)
    
    image = open_image(output_path)
    height, width, channels = image.shape
    image_new = image[int(height/3):,:,:]
    print_image(image_new)
    
    #If you want to save the image
    # filename = f"{NumImg}.png"
    # complete_path = os.path.join('Tests/Res/96x72/8000', filename)
    # io.imsave(complete_path, image_new)
    
if __name__ == "__main__" :
    """
    To launch the main
    """
    start = time.time()
    
    
    main()
    
    end = time.time()
    print("The time used to execute this is : ")
    print(end - start)
    


