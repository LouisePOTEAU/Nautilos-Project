# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:55:24 2024

@author: Louise
"""

#importer ultralytics et torch
from ultralytics import YOLO

# Load a model, we use the nano version
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1000)  # train the model