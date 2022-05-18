cmake_minimum_required(VERSION 3.12)
project(adaptive_clustering_gpu)

find_package(PCL 1.11 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  std_msgs 
  sensor_msgs 
  visualization_msgs 
  geometry_msgs 
  pcl_conversions 
  pcl_ros)

find_package(Eigen3 REQUIRED)

include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS})

add_message_files(FILES ClusterArray.msg)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package()

# TODO: replace with a regex for the shared object name (i.e. *.1.11).
set(PCL_1_11_LIBRARIES
  pcl_common.so.1.11
  pcl_kdtree.so.1.11
  pcl_octree.so.1.11
  pcl_search.so.1.11
  pcl_sample_consensus.so.1.11
  pcl_filters.so.1.11
  pcl_io.so.1.11
  pcl_features.so.1.11
  pcl_ml.so.1.11
  pcl_segmentation.so.1.11
  pcl_visualization.so.1.11
  pcl_surface.so.1.11
  pcl_registration.so.1.11
  pcl_keypoints.so.1.11
  pcl_tracking.so.1.11
  pcl_recognition.so.1.11
  pcl_stereo.so.1.11
  pcl_apps.so.1.11
  pcl_outofcore.so.1.11
  pcl_gpu_containers.so.1.11
  pcl_gpu_utils.so.1.11
  pcl_gpu_octree.so.1.11
  pcl_gpu_features.so.1.11
  pcl_gpu_segmentation.so.1.11
  pcl_people.so.1.11
  /usr/lib/aarch64-linux-gnu/libboost_system.so
  /usr/lib/aarch64-linux-gnu/libboost_filesystem.so
  /usr/lib/aarch64-linux-gnu/libboost_date_time.so
  /usr/lib/aarch64-linux-gnu/libboost_iostreams.so
  /usr/lib/aarch64-linux-gnu/libboost_serialization.so
  /usr/lib/aarch64-linux-gnu/libboost_regex.so
  optimized
  /usr/lib/aarch64-linux-gnu/libqhull.so
  debug
  /usr/lib/aarch64-linux-gnu/libqhull.so
  /usr/lib/libOpenNI.so
  /usr/lib/libOpenNI2.so
  vtkChartsCore
  vtkCommonColor
  vtkCommonDataModel
  vtkCommonMath
  vtkCommonCore
  vtksys
  vtkCommonMisc
  vtkCommonSystem
  vtkCommonTransforms
  vtkInfovisCore
  vtkFiltersExtraction
  vtkCommonExecutionModel
  vtkFiltersCore
  vtkFiltersGeneral
  vtkCommonComputationalGeometry
  vtkFiltersStatistics
  vtkImagingFourier
  vtkImagingCore
  vtkalglib
  vtkRenderingContext2D
  vtkRenderingCore
  vtkFiltersGeometry
  vtkFiltersSources
  vtkRenderingFreeType
  /usr/lib/aarch64-linux-gnu/libfreetype.so
  /usr/lib/aarch64-linux-gnu/libz.so
  vtkftgl
  vtkFiltersModeling
  vtkImagingSources
  vtkInteractionStyle
  vtkInteractionWidgets
  vtkFiltersHybrid
  vtkImagingGeneral
  vtkImagingHybrid
  vtkIOImage
  vtkDICOMParser
  vtkIOCore
  vtkmetaio
  /usr/lib/aarch64-linux-gnu/libjpeg.so
  /usr/lib/aarch64-linux-gnu/libpng.so
  /usr/lib/aarch64-linux-gnu/libtiff.so
  vtkRenderingAnnotation
  vtkImagingColor
  vtkRenderingVolume
  vtkIOGeometry
  vtkIOLegacy
  vtkIOPLY
  vtkRenderingLOD
  vtkViewsContext2D
  vtkViewsCore
  vtkRenderingContextOpenGL
  vtkRenderingOpenGL
  vtkRenderingQt
  vtkFiltersTexture
  vtkGUISupportQt
  vtkRenderingLabel
  FLANN::FLANN
)


add_executable(adaptive_clustering_gpu src/adaptive_clustering_gpu.cpp)
target_link_libraries(adaptive_clustering_gpu ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${PCL_1_11_LIBRARIES})
if(catkin_EXPORTED_TARGETS)
  add_dependencies(adaptive_clustering_gpu ${catkin_EXPORTED_TARGETS})
endif()


install(TARGETS 
    ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY
    launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})