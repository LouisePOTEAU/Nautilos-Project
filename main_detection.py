# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:14:14 2024

@author: Louise
"""

from ultralytics import YOLO
import cv2

"""
We trained the model with YOLOV8 before and we just used the results now
model = YOLO('best.pt')

Load video
VideoPath = './test.mp4'

"""

"""
Manual counting - without saving the video
"""

# def detection(VideoPath, Model):


#     cap = cv2.VideoCapture(VideoPath)
   
#     Switch = True
#     # read frames
   
#     while Switch:
#         Switch, Frame = cap.read()
#         if Switch:
   
#             #Track objects
#             #Model.track is a YOLOV8 moduleR
#             results = Model.track(Frame, persist=True)
           
#             #Plot results
#             AnnotedFrame = results[0].plot()
   
#             #Visualize
#             cv2.imshow('frame', AnnotedFrame)
           
#             #If q is taped, the window will shut down
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break

# detection('Dataset/St53.mp4', YOLO('best.pt'))

###########################

"""
Automatic counting - without saving the video
"""

# import cv2
# from ultralytics import YOLO
# from ultralytics.solutions import object_counter

# VideoPath = 'Dataset/St53.mp4'

# model = YOLO('best.pt')
# cap = cv2.VideoCapture(VideoPath)

# # Init Object Counter
# counter = object_counter.ObjectCounter()

# # Define region points
# region_points = [(0, 144), (768, 144), (768, 600), (0, 600)]

# # counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, draw_tracks=True)
# counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, draw_tracks=True, line_dist_thresh=1)

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     # tracks = model.track(im0, persist=True, show=False)
#     tracks = model.track(im0, conf = 0.05, persist=True, show=False)
#     counter.start_counting(im0, tracks)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

###########################

"""
Automatic counting - saving the video
"""

import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

# Chemin de la vidéo d'entrée
VideoPath = 'Dataset/St53.mp4'
OutputPath = 'OutputVideo/Version3.6-Auto.mp4'  # Chemin de la vidéo de sortie

model = YOLO('best.pt')
cap = cv2.VideoCapture(VideoPath)

# Init Object Counter
counter = object_counter.ObjectCounter()

# Définir les points de la région
region_points = [(0, 144), (768, 144), (768, 600), (0, 600)]
counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, draw_tracks=True)

# Obtenez les dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialiser le VideoWriter pour enregistrer la vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choisir le codec approprié
out = cv2.VideoWriter(OutputPath, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
   
    # Tracking des objets
    tracks = model.track(im0, persist=True, show=False)
   
    # Démarrer le comptage d'objets
    counter.start_counting(im0, tracks)

    # Écrire la frame annotée dans le fichier de sortie
    out.write(im0)  # Note: Assurez-vous que im0 contient les annotations souhaitées

    # Afficher la frame annotée (facultatif)
    cv2.imshow('Annotated Frame', im0)  # Afficher la frame annotée
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

###########################

"""
Automatic counting - saving the video
"""

# import cv2
# from ultralytics import YOLO  # Assurez-vous que vous utilisez le bon import pour YOLOv8

# def detection(VideoPath, Model, OutputPath):
#     cap = cv2.VideoCapture(VideoPath)
   
#     # Obtenez les dimensions de la vidéo
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Initialiser le VideoWriter pour enregistrer la vidéo de sortie
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Vous pouvez changer le codec si nécessaire
#     out = cv2.VideoWriter(OutputPath, fourcc, fps, (frame_width, frame_height))
   
#     Switch = True
#     while Switch:
#         Switch, Frame = cap.read()
#         if Switch:
#             # Track objects
#             results = Model.track(Frame, persist=True)
           
#             # Plot results
#             AnnotedFrame = results[0].plot()
   
#             # Enregistrez la frame annotée
#             out.write(AnnotedFrame)

#             # Visualiser
#             cv2.imshow('frame', AnnotedFrame)
           
#             # Si 'q' est pressé, la fenêtre se ferme
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break

#     # Libérez les ressources
#     cap.release()
#     out.release()
    
#     cv2.destroyAllWindows()

# # Exemple d'utilisation
# detection('Dataset/St53.mp4', YOLO('best.pt'), 'OutputVideo/Version3.6-Manual.mp4')
