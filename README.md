# comp_vision_final_project

<div align="center">
<img width=460px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/portada.png" alt="explode"></a>
<img width=480px heigh=500px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/original.png" alt="explode"></a>
</div>

<br>
<div align="center">
<img width=100px src="https://img.shields.io/badge/status-finished-brightgreen" alt="explode"></a>
<img width=100px src="https://img.shields.io/badge/license-Apache-orange" alt="explode"></a>
</div>




## Table of Contents
- [Problem](#problem)
- [Option 1](#option-1)
- [Color spaces](#color-spaces)
- [Comunication between nodes](#comunication-between-nodes)
- [3D center proyection in 2D](#3d-center-proyection-in-2d)
- [2D center proyection in 3D](#2d-center-proyection-in-3d)
- [Person tracking](#person-tracking)
- [Option 2](#option-2)
- [Proyect all the spere from 3D to 2D](#proyect-all-the-spheres-from-3d-to-2d)
- [Detect the 3D spheres with K-means algorithm](#detect-the-3d-spheres-with-k-means-algorithm)
- [Video](#video)
- [trace_bar_node](#trace_bar_node)

## Problem
This practice involves using image filtering by OpenCV and 3D point cloud filtering together. The following are the problems encountered during the practice and their respective solutions. All the code was required to be in a single CPP file and divided into 2 nodes following the given structure.

-----------------------------------------------------------------------

## Option 1

The first step is to identify a person. If the robot locates the person, it will detect the pink spheres in both 2D (color filtering + Hough model) and 3D (color filtering in point cloud and RANSAC model).

<div align="center">
<img width=900px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/option1_all.png" alt="explode"></a>
</div>


Both the center and the projection of the ball will be shown in both dimensions, and then the 3D center will be shown in the 2D image, and the 2D center will be shown in the 3D image.

-----------------------------------------------------------------------

### Color spaces
Using different color spaces for image filtering and point cloud filtering can add more robustness to the detection process. HSV color space is often used for image processing because it separates hue, saturation, and value/brightness components, making it easier to work with colors. On the other hand, RGB color space is commonly used for point clouds because it represents the color of each point as a combination of red, green, and blue channels. By using both color spaces, we can take advantage of the strengths of each space and improve the accuracy of the detection.

-----------------------------------------------------------------------

### Comunication between nodes
We will use the following diagram to conect the nodes:

<div align="center">
<img heigh=400px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/topics.png" alt="explode"></a>
</div>

The OpenCV node sends data from the trace bars and the 3D sphere centers on two separate topics to which the PCL node subscribes. The PCL node only publishes the 3D centers along with the radii of the spheres.

Regarding the topics of the centers, the first element is the number of spheres, followed by the next 3 or 4 elements which correspond to x, y, z, and r in the case of the PCL node.

<!-- ### 2D Ball detection
 ### 3D Ball detection -->
 
 -----------------------------------------------------------------------

### 3D center proyection in 2D
First, we obtain the center coordinates from the callback of the subscriber in the cv node. The callback function is responsible for converting the 3D points from the topic to 2D points. First, we obtain the intrinsic matrix of the camera information. Since both dimensions in this case have the same reference system, the extrinsic matrix will be the identity matrix with zero translation. We multiply the intrinsic matrix by the extrinsic matrix and the 3D point to obtain the 2D point.


Note that this operation will give a 3-dimensional vector. We need to divide the x and y values by the third component. We store the vectors in a std::vector<cv::Vec3f>, where we will store x, y, and r (r will be explained later).Once these operations are performed, we only need to use the OpenCV circle function.

-----------------------------------------------------------------------

### 2D center proyection in 3D
As in the previous case, we obtain the coordinates from the callback. In this case, it is simpler since the topic publishes the coordinates in 3D. All we have to do is store the points in a std::vector<std::vector<float>>, which is an attribute of the node's class, and then draw them in the point cloud with the 'draw_square' function.


-----------------------------------------------------------------------

### Person tracking
OpenCV was used for person detection, using the hog.detectMultiScale function. Several parameters were adjusted, such as the detection window size, weight size, and confidence threshold. Then, the found vector is iterated over, comparing each region to discard duplicates.

```cpp
std::vector<cv::Rect> detect_people(cv::Mat img)
{
  cv::HOGDescriptor hog;
  hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  std::vector<cv::Rect> found, found_filtered;

  hog.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

  size_t i, j;
  for (i = 0; i < found.size(); i++) {

    cv::Rect r = found[i];

    for (j = 0; j < found.size(); j++)
      if (j != i && (r & found[j]) == r)
        break;
    
    if (j == found.size())
      found_filtered.push_back(r);
  }
  
  return found_filtered;
}

```

<div align="center">
<img width=400px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/person_track.png" alt="explode"></a>
<img width=400px heigh=500px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/without_person.png" alt="explode"></a>
</div>

-----------------------------------------------------------------------

## Option 2

#### Proyect all the spheres from 3D to 2D
Taking advantage of the fact that the RANSAC method used to calculate the center of the spheres in 3D also provided us with the radius, this optional feature was implemented. The radius was also passed as the 4th argument in the topic published by the pcl node.

In the callback function of the OpenCV node, a point of this radius was calculated by adding the radius to one component of the center. Then, the distance between the center and the new point generated in 2D was obtained to calculate the radius in 2D.

<br><div align="center">
<img width=800px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/3k.png" alt="explode"></a>
</div>

-----------------------------------------------------------------------

### Detect the 3D spheres with K-means algorithm

 Definition:
 
 > The K-means algorithm is a clustering algorithm that partitions a dataset into K clusters, where K is a user-defined number of clusters. The algorithm assigns each data point to the cluster whose centroid is closest, with the centroids being the means of the data points assigned to each cluster.<br><br>
 The algorithm works by initializing the centroids randomly and iteratively optimizing the assignment of points to clusters and the position of the centroids until convergence.

The implementation allowed the user to choose the desired number of clusters. If there are multiple objects and only one cluster, the center of these objects will be shown. If there are the same number of clusters as there are spheres, ideally the same number of centers as spheres will be shown. However, sometimes a cluster is generated far away and another one has both spheres, which can be resolved by increasing the number of clusters.
 
 Furthermore, even though it decreased performance, the point cloud was iterated again and each point was colored according to which cluster it belonged to, showing which areas were affected by each cluster.
 
<div align="center">
<img width=800px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/1k.png" alt="explode"></a>
</div>

<br><div align="center">
<img width=800px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/bug.png" alt="explode"></a>
</div><br>

Here we can see how two spheres are mistaken for the same one. (Purple)

-----------------------------------------------------------------------

## trace_bar_node
A node was created that displayed 6 tracebars and published their results to a topic, which was later subscribed to by the cv_pcl-node to properly adjust the color filters.
<div align="center">
<img width=200px src="https://github.com/madinabeip/comp_vision_final_project/blob/main/resources/trace_bars.png" alt="explode"></a>
</div>

## Video
https://user-images.githubusercontent.com/72991324/236585025-32731c21-1b69-46cd-b989-72af42d96daf.mp4

https://user-images.githubusercontent.com/72991324/236587309-74f57e88-8d55-4104-bf21-353611937734.mp4

In the second video, we can see how the k-means algorithm is not optimal for real-time detection, as clearing the initial positions of the random clusters may result in one cluster detecting multiple spheres, or multiple clusters detecting half of a sphere each.

In the first video, the performance appears slower due to the simulator client being open, which significantly slowed down the computer's processing. In the second video, it can be observed that without the client running, the processing speed is much faster.


## Licencia 
<a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Apache License" style="border-width:0" src="https://www.apache.org/img/asf-estd-1999-logo.jpg" /></a><br/> </a><br/>This work is licensed under a <a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache license 2.0








