# Mono-Sceneflow
The code was written in 2014. If you can run this code while familiar with the paper "Live dense reconstruction of a single camera", you could find the truth that initial reconstruction by MSCSRBF is not really neccessary. 
If you have dense optical flow estimation, combined with some opengl tricks, you can re-implement the dense mono reconstruction by using monocular sceneflow. In my opinion, Brox's results should be good enough to use.
Besides, I left the interface of PCL's Poinsson Surface Reconstructio inside this project in order to compare with Sceneflow. 
