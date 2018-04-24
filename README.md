# Description 
This package is used for heart rate extraction from head's motion using skeleton detection, head segmentation wavelet processing. "time_serify" is a ROS-wrapped package treating following messages :

* **Subscribers** : "/centroid_XYZ" **Type** : Point
* **Publisher** : "time_serify/heart_rate" **Type** : Int64

# Prerequesites
This whole project can only run, online and offline, using ROS's middleware. ROS is a middleware enabling multiple user-friendly functionalities in robotics, vision and and signal processing. 

* **ROS middleware** Install ROS following this guideline :http://wiki.ros.org/kinetic/Installation/Ubuntu
* **Kinect driver and ROS package** Install the kinect package following **all** steps in the following tutorial : https://github.com/code-iai/iai_kinect2
**Tensorflow** Install tensorflow for python
```bash
pip install tensorflow
```
* **Openpose skeleton detectorimplemented on tensorflow** Install *tf-pose-estimation-1* following fork in in your catkin workspace:
```bash
cd ~/catkin_ws/src
git clone https://github.com/PFE2018/tf-pose-estimation-1.git
```

* **Head segmentation** Install following head_getter and head_filter packages in your catkin workspace for head segmentation in the point cloud :
```bash
git clone https://github.com/PFE2018/head_getter.git
git clone https://github.com/PFE2018/head_filter.git
```
* **Head position** Install following cluster_gen in your catkin workspace for head's position extraction
```bash
git clone https://github.com/PFE2018/cluster_gen.git
```
* **Heart rate extraction** Install following time_serify package in your catkin workspace for heart rate extraction
```bash
git clone https://github.com/PFE2018/time_serify.git
```
# System
This project aims to detect multiple humans in an 2D image and extract their heart rates. Using a tensorflow implementation of the 2D Pose convolutional machine "Open pose", the skeleton of multiple humans can be detected in an 2D image as long as the human is vertical in the image. Thereafter, the head's point cloud is extracted by selecting the point cloud around the nose. Eventually, more heuristics could be implemented to select a box above the shoulder if the human is not facing the camera. From the head's point cloud, its centroid is computed using pcl library tools. Finally, windows of 10 seconds are recorded in a First In Last Out (FILO) way and wavelet processing is applied to each window to extract the scale of the signal linked to the heart rate. Simple peak detection is used to compute the individual heart beats in the wavelets transform and the mean interbeat interval is extracted for the sliding 10-seconds window. 
![alt text](https://github.com/PFE2018/time_serify/blob/master/BlockSchemeSoftwareRelease.png?raw=true)

# Example
To directly run all the online heart rate extraction, run :
```bash
roslaunch time_serify  base.launch
```
