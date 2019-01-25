# FYP - Human Following Robot 

This repository contains code for the detecting and tracking of humans in order to obtain their depth information using a Kinect-One sensor. 

Created and tested in Ubuntu 16.04 with Ros Kinetic.

### Three main methods were employed for detecting humans and objects:
- Object Detection using HSV and RGB Color Models
- Human Detection using Histogram of Oriented Gradients (HOG) with Support Vector Machines (SVM)
- Human Detection using Convoluted Neural Networks (YOLO), a state-of-the-art, real-time object detection system. 

### Tracking:
- Tracking was employed using Mean Shift Tracking algorithm using the Bhattacharyya Distance and a simple linear Kalman Filter. Both these tracking methods were employed using their existing OpenCV packages. 
