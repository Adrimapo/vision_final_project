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

### 2D Ball detection

### 3D Ball detection

### 3D center proyection in 2D

### 2D center proyection in 3D

## Option 2 (extras)

### Proyect all the spere from 3D to 2D

### Detect the 3D spheres with K-means algorithm


## trace_bar_node
A node was created that displayed 6 tracebars and published their results to a topic, which was later subscribed to by the cv_pcl-node to properly adjust the color filters.

!!!!! Insertar imagen de los tracebar
