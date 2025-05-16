"""
Date: 2024-07-26
LastEditors: JJQ jj1623@ic.ac.uk
LastEditTime: 2024-07-26
FilePath: /coloradar/generate_bag.py
Description: 
"""
#!/usr/bin/env python3
import hashlib
import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion
import tf.transformations
import os
import glob
import argparse
import re

import ros_numpy
import pandas as pd

IMUTYPE = [
    ("timestamp", "float64"),
    ("ax", "float32"),
    ("ay", "float32"),
    ("az", "float32"),
    ("gx", "float32"),
    ("gy", "float32"),
    ("gz", "float32"),
]
# RADARTYEP = [('timestamp', 'float64'), ('type', 'U10'), ('x', 'float32'), ('y', 'float32'),
#              ('z', 'float32'), ('velocity', 'float32'), ('intensity', 'float32')]
RADARTYEP = [("timestamp", "float64"), ("file_path", "U200")]

ENABLE_FFT_CORRECTION = True # only used for the dataset provided bin files
ENABLE_FFT_CORRECTION = False

def numerical_sort(value):
    """
    Extracts numbers from a filename for sorting purposes.
    """
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])  # Convert captured strings to integers
    return parts


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process RADAR and IMU data for a given dataset."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True,
                      help="Name of the dataset folder.")
    parser.add_argument("--base-dir", type=str, 
                      default="./dataset",
                      help="Base directory where dataset is stored.")
    parser.add_argument("--imu-dir", type=str, default="imu",
                      help="Directory containing IMU data relative to dataset.")
    parser.add_argument("--imu-data-file", type=str, default="imu_data.txt",
                      help="Filename for IMU data.")
    parser.add_argument("--imu-time-file", type=str, default="timestamps.txt",
                      help="Filename for IMU timestamps.")
    parser.add_argument("--radar-dir", type=str, default="single_chip/pointclouds",
                      help="Directory containing radar data relative to dataset.")
    parser.add_argument("--gt-dir", type=str, default="groundtruth",
                      help="Directory containing ground truth data.")
    parser.add_argument("--gt-file-pattern", type=str, 
                      default="{dataset}_vicon2gt_states.csv",
                      help="Pattern for ground truth filename with {dataset} placeholder.")
    parser.add_argument("--output-dir", type=str, default=".",
                      help="Output directory for bag file.")
    parser.add_argument("--format", type=str, default="filter", choices=["filter", "optimize"],
                      help="Format type for bag generation (filter or optimize)")
    
    return parser.parse_args()


def read_imu_data(imu_data_filename, imu_time_filename):

    if not os.path.exists(imu_data_filename) or not os.path.exists(imu_time_filename):
        print(f"File {imu_data_filename} or {imu_time_filename} not found")
        return None, None

    imu_times = np.loadtxt(imu_time_filename)
    imu_data = np.zeros(len(imu_times), dtype=IMUTYPE)

    with open(imu_data_filename, "r") as file:
        for i, line in enumerate(file):
            vals = [float(s) for s in line.split()]
            # Check if the line has the correct number of values
            if len(vals) == 6:
                # Assign values to structured array
                imu_data[i] = (
                    imu_times[i],
                    vals[0],
                    vals[1],
                    vals[2],
                    vals[3],
                    vals[4],
                    vals[5],
                )
            else:
                print(
                    f"Line {i+1} in {imu_data_filename} does not have exactly 6 values, skipping this line."
                )

    return imu_data


def read_radar_data(radar_data_filename):
    if not os.path.exists(radar_data_filename):
        print(f"File {radar_data_filename} not found")
        return None, None

    with open(radar_data_filename, "rb") as f:
        radar_data = np.fromfile(f, dtype=np.float32).reshape(-1, 5)
    return radar_data


def read_all_radar_files(radar_data_path):
    radar_pcl_file_path = os.path.join(radar_data_path, "data")
    radar_files = sorted(
        glob.glob(os.path.join(radar_pcl_file_path, "*.bin")), key=numerical_sort
    )
    radar_time_filename = os.path.join(radar_data_path, "timestamps.txt")
    radar_times_list = np.loadtxt(radar_time_filename)

    radar_data_list = []

    # !: if the timestamp number is not equal to the file number, error exists
    radar_structured = np.zeros(len(radar_times_list), dtype=RADARTYEP)
    radar_structured["timestamp"] = radar_times_list
    radar_structured["file_path"] = radar_files

    return radar_structured


def create_index_array(imu_data, radar_structured, ground_data):
    # Calculate total length of valid data
    total_length = 0
    if imu_data is not None and len(imu_data) > 0:
        total_length += len(imu_data)
    if radar_structured is not None and len(radar_structured) > 0:
        total_length += len(radar_structured)
    if ground_data is not None and len(ground_data) > 0:
        total_length += len(ground_data)
    
    if total_length == 0:
        print("No data to process. Cannot create bag file.")
        return None
        
    # Create a structured array to hold timestamps, indexes, and a type indicator
    dtype = [("timestamp", "float64"), ("index", "int"), ("type", "U20")]
    index_array = np.zeros(total_length, dtype=dtype)
    
    offset = 0
    
    # Populate the array for IMU data
    if imu_data is not None and len(imu_data) > 0:
        for i, data in enumerate(imu_data):
            index_array[offset + i] = (data["timestamp"], i, "IMU")
        offset += len(imu_data)
    
    # Populate the array for Radar data
    if radar_structured is not None and len(radar_structured) > 0:
        for i, data in enumerate(radar_structured):
            index_array[offset + i] = (data["timestamp"], i, "RADAR")
        offset += len(radar_structured)
    
    # Populate the array for Ground truth data
    if ground_data is not None and len(ground_data) > 0:
        for i, data in enumerate(ground_data):
            index_array[offset + i] = (data["timestamp"], i, "GT")
    
    # Sort by timestamp
    index_array.sort(order="timestamp")
    
    return index_array

def transform_radar_data(original_data, format_type="filter"):
    # Assuming original_data is shaped (-1, 5) with columns [x, y, z, doppler, snr]
    
    # Initialize noise_db with a default value or compute it
    default_noise_db = 0.5  # Example default value for noise_db
    noise_db = np.full((original_data.shape[0], 1), default_noise_db, dtype=np.float32)
    
    # Select and rename columns
    x = original_data[:, 0:1]
    y = original_data[:, 1:2]
    z = original_data[:, 2:3]
    v_doppler = original_data[:, 3:4]
    snr_db = original_data[:, 4:5]
    if ENABLE_FFT_CORRECTION:
        # swap snr and v_doppler
        v_doppler = original_data[:, 4:5]
        snr_db = original_data[:, 3:4]

    # Concatenate columns in format-specific order
    if format_type == "optimize":
        transformed_data = np.hstack((x, y, z, v_doppler, snr_db, noise_db))
    else:
        transformed_data = np.hstack((x, y, z, snr_db, noise_db, v_doppler))
    
    return transformed_data

def create_pointcloud2_message(radar_data, timestamp, format_type="filter"):
    transformed_data = transform_radar_data(radar_data, format_type)

    if format_type == "optimize":
        transformed_data.dtype = {
            "names": ["x", "y", "z", "doppler", "snr", "noise"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8, 12, 16, 20],
            "itemsize": 24,
        }
        frame_id = "awr1843aop"
    else:
        transformed_data.dtype = {
            "names": ["x", "y", "z", "snr_db", "noise_db", "v_doppler_mps"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8, 12, 16, 20],
            "itemsize": 24,
        }
        frame_id = "radar"

    msg = ros_numpy.point_cloud2.array_to_pointcloud2(
        transformed_data, stamp=timestamp, frame_id=frame_id
    )

    return msg

def create_pointcloud2_message2(radar_data, timestamp):
    # Assuming radar_data is an array with shape (-1, 5) and columns are [x, y, z, snr_db, noise_db, v_doppler]
    
    msg = PointCloud2()
    msg.header.stamp = timestamp
    msg.header.frame_id = "radar"
    msg.height = 1  # Unstructured point cloud
    msg.width = len(radar_data)
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='velocity', offset=16, datatype=PointField.FLOAT32, count=1)
    ]
    
    msg.is_bigendian = False
    msg.point_step = 20  # The number of bytes in a point record
    msg.row_step = msg.point_step * msg.width
    msg.data = np.asarray(radar_data, dtype=np.float32).tobytes()
    msg.is_dense = True  # No invalid (NaN or Inf) points
    return msg

# Function to write data to a ROS bag
def write_to_rosbag(bag_filename, index_array, imu_data, radar_structured, ground_data, format_type="filter"):
    if index_array is None or len(index_array) == 0:
        print("No data to write to bag file.")
        return
    
    try:
        bag = rosbag.Bag(bag_filename, "w")
        data_counts = {"IMU": 0, "RADAR": 0, "GT": 0}
        
        for entry in index_array:
            timestamp = rospy.Time.from_sec(entry["timestamp"])
            
            if entry["type"] == "IMU" and imu_data is not None:
                # Fetch the actual IMU data using the index
                data = imu_data[entry["index"]]
                imu_msg = Imu()
                imu_msg.header.stamp = timestamp
                imu_msg.header.seq = entry["index"]
                
                if format_type == "optimize":
                    imu_msg.header.frame_id = "bmi088"
                    imu_topic = "/imu/data_raw"
                else:
                    imu_msg.header.frame_id = "base_link"
                    imu_topic = "/imu"
                    
                imu_msg.linear_acceleration.x = data["ax"]
                imu_msg.linear_acceleration.y = data["ay"]
                imu_msg.linear_acceleration.z = data["az"]
                imu_msg.angular_velocity.x = data["gx"]
                imu_msg.angular_velocity.y = data["gy"]
                imu_msg.angular_velocity.z = data["gz"]
                bag.write(imu_topic, imu_msg, imu_msg.header.stamp)
                data_counts["IMU"] += 1
                
            elif entry["type"] == "RADAR" and radar_structured is not None:
                # Fetch the actual Radar data using the index
                radar_file_path = radar_structured[entry["index"]]["file_path"]
                radar_data = read_radar_data(radar_file_path)
                if radar_data is not None and len(radar_data) > 0:
                    radar_msg = create_pointcloud2_message(radar_data, timestamp, format_type)
                    radar_msg.header.seq = entry["index"]
                    
                    if format_type == "optimize":
                        radar_topic = "/radar/cfar_detections"
                    else:
                        radar_topic = "/radar/pcl"
                        
                    bag.write(radar_topic, radar_msg, radar_msg.header.stamp)
                    data_counts["RADAR"] += 1
                else:
                    print(f"Error reading radar data from {radar_file_path}")
                    
            elif entry["type"] == "GT" and ground_data is not None:
                # Fetch the actual Ground truth data using the index
                data = ground_data[entry["index"]]
                
                # Create and write pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = timestamp
                pose_msg.header.frame_id = "world"
                pose_msg.pose.position.x = data["px"]
                pose_msg.pose.position.y = data["py"]
                pose_msg.pose.position.z = data["pz"]
                pose_msg.pose.orientation.x = data["qx"]
                pose_msg.pose.orientation.y = data["qy"]
                pose_msg.pose.orientation.z = data["qz"]
                pose_msg.pose.orientation.w = data["qw"]
                bag.write("/ground_truth/pose", pose_msg, pose_msg.header.stamp)

                # For "optimize" format, write an additional IMU message with orientation
                if format_type == "optimize":
                    imu_msg = Imu()
                    imu_msg.header.stamp = timestamp
                    imu_msg.header.frame_id = "bmi088"
                    imu_msg.orientation = pose_msg.pose.orientation
                    bag.write("/imu/data", imu_msg, imu_msg.header.stamp)

                # Create and write twist messages for default format
                twist_msg = TwistStamped()
                twist_msg.header.stamp = timestamp
                twist_msg.header.frame_id = "world"
                twist_msg.twist.linear.x = data["vx"]
                twist_msg.twist.linear.y = data["vy"]
                twist_msg.twist.linear.z = data["vz"]
                bag.write("/ground_truth/twist", twist_msg, twist_msg.header.stamp)

                # Calculate and write body frame twist
                global_vel = np.array([data["vx"], data["vy"], data["vz"]])
                quaternion = np.array([data["qx"], data["qy"], data["qz"], data["qw"]])
                rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
                body_vel = np.dot(rotation_matrix.T, global_vel)
                
                twist_body_msg = TwistStamped()
                twist_body_msg.header.stamp = timestamp
                twist_body_msg.header.frame_id = "base_link"
                twist_body_msg.twist.linear.x = body_vel[0]
                twist_body_msg.twist.linear.y = body_vel[1]
                twist_body_msg.twist.linear.z = body_vel[2]
                bag.write("/ground_truth/twist_body", twist_body_msg, twist_body_msg.header.stamp)
                
                data_counts["GT"] += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'bag' in locals():
            bag.close()
            print(f"ROS bag '{bag_filename}' created successfully.")
            print(f"Bag contents: {data_counts['IMU']} IMU messages, {data_counts['RADAR']} Radar messages, {data_counts['GT']} Ground Truth messages")

def read_gt_data(file_path):
    # Implementation of read_gt_data function goes here
    # Read the ground truth data from the csv file and return it as a structured array
    # The structured array should have the following fields:
    # time(ns)	px	py	pz	qw	qx	qy	qz	vx	vy	vz	bwx	bwy	bwz	bax	bay	baz

    df = pd.read_csv(file_path)

    dtype = [('timestamp', 'float64'), ('px', 'float32'), ('py', 'float32'), ('pz', 'float32'), ('qw', 'float32'), ('qx', 'float32'), ('qy', 'float32'), ('qz', 'float32'), ('vx', 'float32'), ('vy', 'float32'), ('vz', 'float32'), ('bwx', 'float32'), ('bwy', 'float32'), ('bwz', 'float32'), ('bax', 'float32'), ('bay', 'float32'), ('baz', 'float32')]

    # Convert DataFrame to numpy structured array
    ground_data = np.array([tuple(x) for x in df.to_records(index=False)], dtype=dtype)

    # convert timestamp from ns to s
    ground_data['timestamp'] = ground_data['timestamp'] / 1e9

    return ground_data

def main():
    args = parse_arguments()
    source_dir = os.path.join(args.base_dir, args.dataset)
    
    # Configure paths using command line arguments
    imu_dir = os.path.join(source_dir, args.imu_dir)
    imu_data_filename = os.path.join(imu_dir, args.imu_data_file)
    imu_time_filename = os.path.join(imu_dir, args.imu_time_file)
    
    radar_data_path = os.path.join(source_dir, args.radar_dir)
    
    gt_dir = os.path.join(source_dir, args.gt_dir)
    gt_data_filename = os.path.join(gt_dir, args.gt_file_pattern.format(dataset=args.dataset))
    
    bag_filename = os.path.join(args.output_dir, args.dataset + ".bag")

    # Load data with handling for missing files
    imu_data = None
    radar_data = None
    ground_data = None
    
    # Try to load IMU data
    if os.path.exists(imu_data_filename) and os.path.exists(imu_time_filename):
        print(f"Loading IMU data from {imu_data_filename}")
        imu_data = read_imu_data(imu_data_filename, imu_time_filename)
    else:
        print(f"IMU data not found at {imu_data_filename}")
    
    # Try to load Radar data
    if os.path.exists(radar_data_path):
        print(f"Loading Radar data from {radar_data_path}")
        radar_data = read_all_radar_files(radar_data_path)
    else:
        print(f"Radar data directory not found at {radar_data_path}")
    
    # Try to load Ground Truth data
    if os.path.exists(gt_data_filename):
        print(f"Loading Ground Truth data from {gt_data_filename}")
        ground_data = read_gt_data(gt_data_filename)
    else:
        print(f"Ground Truth data not found at {gt_data_filename}")

    # Create index array only with available data
    index_array = create_index_array(imu_data, radar_data, ground_data)

    # Write to bag file with specified format
    write_to_rosbag(bag_filename, index_array, imu_data, radar_data, ground_data, args.format)


if __name__ == "__main__":
    main()
