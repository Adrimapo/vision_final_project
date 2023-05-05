# comp_vision_final_project

!!!!!!! Imagenn del tiago o del escenario

# Problem
This practice involves using image filtering by OpenCV and 3D point cloud filtering together. The following are the problems encountered during the practice and their respective solutions. All the code was required to be in a single CPP file and divided into 2 nodes following the given structure.

## Option 1
The first step is to identify a person. If the robot locates the person, it will detect the pink spheres in both 2D (color filtering + Hough model) and 3D (color filtering in point cloud and RANSAC model).

Both the center and the projection of the ball will be shown in both dimensions, and then the 3D center will be shown in the 2D image, and the 2D center will be shown in the 3D image.

### Comunication between nodes
We will use the following diagram to conect the nodes:

!!!!!!! Insertar imagen de la comunicación de nodos

!!!!!!! Explicar a partir de la imagen la comunicación

<!-- ### 2D Ball detection
 ### 3D Ball detection -->

### 3D center proyection in 2D
First, we obtain the center coordinates from the callback of the subscriber in the cv node. The callback function is responsible for converting the 3D points from the topic to 2D points. First, we obtain the intrinsic matrix of the camera information. Since both dimensions in this case have the same reference system, the extrinsic matrix will be the identity matrix with zero translation. We multiply the intrinsic matrix by the extrinsic matrix and the 3D point to obtain the 2D point.

!!!!!!!!!!!!!! INSERTAR IMAGEN DE LA DETECCIÓN

Note that this operation will give a 3-dimensional vector. We need to divide the x and y values by the third component. We store the vectors in a std::vector<cv::Vec3f>, where we will store x, y, and r (r will be explained later).Once these operations are performed, we only need to use the OpenCV circle function.

### 2D center proyection in 3D
As in the previous case, we obtain the coordinates from the callback. In this case, it is simpler since the topic publishes the coordinates in 3D. All we have to do is store the points in a std::vector<std::vector<float>>, which is an attribute of the node's class, and then draw them in the point cloud with the 'draw_square' function.

!!!!!!!!!!!!!! INSERTAR IMAGEN DE LA DETECCIÓN

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

## Option 2 (extras)

### Proyect all the spere from 3D to 2D
Taking advantage of the fact that the RANSAC method used to calculate the center of the spheres in 3D also provided us with the radius, this optional feature was implemented. The radius was also passed as the 4th argument in the topic published by the pcl node.

In the callback function of the OpenCV node, a point of this radius was calculated by adding the radius to one component of the center. Then, the distance between the center and the new point generated in 2D was obtained to calculate the radius in 2D.

!!!!!!!!!! Insertar imagen

### Detect the 3D spheres with K-means algorithm


## trace_bar_node
A node was created that displayed 6 tracebars and published their results to a topic, which was later subscribed to by the cv_pcl-node to properly adjust the color filters.

!!!!! Insertar imagen de los tracebar
