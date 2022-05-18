// Copyright (C) 2022  Zhi Yan

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "adaptive_clustering_gpu/ClusterArray.h"

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

// PCL-GPU
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>

//#define LOG

ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;

bool print_fps_;
float z_axis_min_;
float z_axis_max_;
int cluster_size_min_;
int cluster_size_max_;

const int region_max_ = 10; // Change this value to match how far you want to detect.
int regions_[100];


pcl::gpu::EuclideanClusterExtraction gpu_euclidian_cluster_extraction;

visualization_msgs::Marker ToMarker(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
    const std_msgs::Header& header,
    const int cluster_index)
{
    Eigen::Vector4f min, max;
    pcl::getMinMax3D(*cluster, min, max);
    
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "adaptive_clustering_gpu";
    marker.id = cluster_index;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    
    std::array<geometry_msgs::Point, 24> points;
    points[0].x = max[0];  points[0].y = max[1];  points[0].z = max[2];
    points[1].x = min[0];  points[1].y = max[1];  points[1].z = max[2];
    points[2].x = max[0];  points[2].y = max[1];  points[2].z = max[2];
    points[3].x = max[0];  points[3].y = min[1];  points[3].z = max[2];
    points[4].x = max[0];  points[4].y = max[1];  points[4].z = max[2];
    points[5].x = max[0];  points[5].y = max[1];  points[5].z = min[2];
    points[6].x = min[0];  points[6].y = min[1];  points[6].z = min[2];
    points[7].x = max[0];  points[7].y = min[1];  points[7].z = min[2];
    points[8].x = min[0];  points[8].y = min[1];  points[8].z = min[2];
    points[9].x = min[0];  points[9].y = max[1];  points[9].z = min[2];
    points[10].x = min[0]; points[10].y = min[1]; points[10].z = min[2];
    points[11].x = min[0]; points[11].y = min[1]; points[11].z = max[2];
    points[12].x = min[0]; points[12].y = max[1]; points[12].z = max[2];
    points[13].x = min[0]; points[13].y = max[1]; points[13].z = min[2];
    points[14].x = min[0]; points[14].y = max[1]; points[14].z = max[2];
    points[15].x = min[0]; points[15].y = min[1]; points[15].z = max[2];
    points[16].x = max[0]; points[16].y = min[1]; points[16].z = max[2];
    points[17].x = max[0]; points[17].y = min[1]; points[17].z = min[2];
    points[18].x = max[0]; points[18].y = min[1]; points[18].z = max[2];
    points[19].x = min[0]; points[19].y = min[1]; points[19].z = max[2];
    points[20].x = max[0]; points[20].y = max[1]; points[20].z = min[2];
    points[21].x = min[0]; points[21].y = max[1]; points[21].z = min[2];
    points[22].x = max[0]; points[22].y = max[1]; points[22].z = min[2];
    points[23].x = max[0]; points[23].y = min[1]; points[23].z = min[2];

    for (const auto& point : points)
    {
        marker.points.push_back(point);
    }
    
    marker.scale.x = 0.02;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.5;
    marker.lifetime = ros::Duration(0.1);

    return marker;
}

geometry_msgs::Pose ToPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    
    geometry_msgs::Pose pose;
    pose.position.x = centroid[0];
    pose.position.y = centroid[1];
    pose.position.z = centroid[2];
    pose.orientation.w = 1;

    return pose;
}


int frames; clock_t start_time; bool reset = true;//fps
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) 
{
    // fps
    if(print_fps_ && reset)
    {
        frames=0;
        start_time=clock();
        reset=false;
    }
    
    /*** Convert ROS message to PCL ***/
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
    
//     /*** Remove ground and ceiling ***/
//     std::vector<int> indices;
//     for(int i = 0; i < pcl_pc_in->size(); ++i) 
//     {
//         if(pcl_pc_in->points[i].z >= z_axis_min_ && pcl_pc_in->points[i].z <= z_axis_max_) 
//         {
//             indices.push_back(i);
//         }
//     }
//     pcl::copyPointCloud(*pcl_pc_in, indices, *pcl_pc_in);
    
    /*** Divide the point cloud into nested circular regions ***/
    boost::array<std::vector<int>, region_max_> indices_array;
    for(int i = 0; i < pcl_pc_in->size(); i++) 
    {
        float range = 0.0;
        for(int region_index = 0; region_index < region_max_; region_index++) 
        {
            float d2 = pcl_pc_in->points[i].x * pcl_pc_in->points[i].x + pcl_pc_in->points[i].y * pcl_pc_in->points[i].y + pcl_pc_in->points[i].z * pcl_pc_in->points[i].z;
            if(d2 > range * range && d2 <= (range+regions_[region_index]) * (range+regions_[region_index])) 
            {
                indices_array[region_index].push_back(i);
                break;
            }
            range += regions_[region_index];
        }
    }
    
    float tolerance = 0.0;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr > > clusters;
    
    for(int region_index = 0; region_index < region_max_; region_index++) 
    {
        tolerance += 0.1;
        
        if(indices_array[region_index].size() > cluster_size_min_) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*pcl_pc_in, indices_array[region_index], *cloud_filtered);
            
            pcl::gpu::Octree::PointCloud cloud_device;
            cloud_device.upload(cloud_filtered->points);
            
            pcl::gpu::Octree::Ptr octree_device(new pcl::gpu::Octree);
            octree_device->setCloud(cloud_device);
            octree_device->build();
            
            std::vector<pcl::PointIndices> cluster_indices_gpu;
            gpu_euclidian_cluster_extraction.setClusterTolerance(tolerance);
            gpu_euclidian_cluster_extraction.setMinClusterSize(cluster_size_min_);
            gpu_euclidian_cluster_extraction.setMaxClusterSize(cluster_size_max_);
            gpu_euclidian_cluster_extraction.setSearchMethod(octree_device);
            gpu_euclidian_cluster_extraction.setHostCloud(cloud_filtered);
            gpu_euclidian_cluster_extraction.extract(cluster_indices_gpu);
            
            for(const pcl::PointIndices& cluster : cluster_indices_gpu) 
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_gpu(new pcl::PointCloud<pcl::PointXYZ>);
                for(const auto& index : (cluster.indices)) 
                {
                    cloud_cluster_gpu->push_back((*cloud_filtered)[index]);
                }
                cloud_cluster_gpu->width = cloud_cluster_gpu->size();
                cloud_cluster_gpu->height = 1;
                cloud_cluster_gpu->is_dense = true;
                clusters.push_back(cloud_cluster_gpu);
            }
        }
    }

    /*** Output ***/
    if(cloud_filtered_pub_.getNumSubscribers() > 0) 
    {
        sensor_msgs::PointCloud2 ros_pc2_out;
        pcl::toROSMsg(*pcl_pc_in, ros_pc2_out);
        cloud_filtered_pub_.publish(ros_pc2_out);
    }
    
    adaptive_clustering_gpu::ClusterArray cluster_array;
    geometry_msgs::PoseArray pose_array;
    visualization_msgs::MarkerArray marker_array;
    
    for(int cluster_index = 0; cluster_index < clusters.size(); cluster_index++) 
    {
        if(cluster_array_pub_.getNumSubscribers() > 0) 
        {
            // add the cluster to the output.
            sensor_msgs::PointCloud2 ros_pc2_out;
            pcl::toROSMsg(*clusters[cluster_index], ros_pc2_out);
            cluster_array.clusters.push_back(ros_pc2_out);
        }
        
        if(pose_array_pub_.getNumSubscribers() > 0) 
        {
            const auto pose = ToPose(clusters[cluster_index]);
            pose_array.poses.push_back(pose);
        }
        
        if(marker_array_pub_.getNumSubscribers() > 0) 
        {
            const auto marker = ToMarker(clusters[cluster_index], ros_pc2_in->header, cluster_index);
            marker_array.markers.push_back(marker);
        }
    }
    
    if(cluster_array.clusters.size()) 
    {
        cluster_array.header = ros_pc2_in->header;
        cluster_array_pub_.publish(cluster_array);
    }

    if(pose_array.poses.size()) 
    {
        pose_array.header = ros_pc2_in->header;
        pose_array_pub_.publish(pose_array);
    }
    
    if(marker_array.markers.size()) 
    {
        marker_array_pub_.publish(marker_array);
    } 
    
    // fps
    if(print_fps_)
    {
        if(++frames>10)
        {
            ROS_DEBUG_STREAM("[adaptive_clustering_gpu] fps = " << float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC) << ", timestamp = " << clock()/CLOCKS_PER_SEC);
            reset = true;
        }
    }  
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering_gpu");
  
  ROS_WARN("This is the GPU version of Adaptive Clustering.");
  pcl::gpu::printCudaDeviceInfo();
  
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  
  /*** Subscribers ***/
  ros::Subscriber point_cloud_sub = private_nh.subscribe<sensor_msgs::PointCloud2>("input", 10, pointCloudCallback);

  /*** Publishers ***/
  cluster_array_pub_ = private_nh.advertise<adaptive_clustering_gpu::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
  
  /*** Parameters ***/
  std::string sensor_model;
  
  private_nh.param<std::string>("sensor_model", sensor_model, "VLP-16"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.8);
  private_nh.param<float>("z_axis_max", z_axis_max_, 2.0);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 3);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 2200000);
  
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"
  if(sensor_model.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  
  ros::spin();

  return 0;
}
