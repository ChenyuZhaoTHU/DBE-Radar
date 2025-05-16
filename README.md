# Digital Beamforming Enhancement for Radar Odometry

<!-- <img src="doc/Title.gif" alt="Video Demo"> -->

Digital Beamforming Enhancement (**DBE**) for Radar Odometry is an advanced radar signal processing pipeline that enhances radar-based odometry and SLAM systems by integrating spatial domain beamforming techniques. This project provides an alternative to traditional FFT-based radar processing, improving the precision of radar localization and mapping.


## Features

- **3D Direction of Arrival (DoA) Estimation** using digital beamforming.
- **Improved Radar Odometry Accuracy** by replacing standard FFT-based pipelines. (only support single-chip radar for now)
- **Enhanced Radar Point Cloud Quality**, reducing errors and improving SLAM performance.
- **Tested on Public Radar Datasets**, demonstrating superior accuracy.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/DBE-Radar-Odometry.git
cd DBE-Radar-Odometry
```


### Prerequisites

- Python 3.8+

Install the virtual environment and dependencies using the following commands:
  ```bash
  python -m venv venv
  source ./venv/bin/activate

  python -m pip install -r requirements.txt --extra-index-url https://rospypi.github.io/simple/
  ```


## Usage
**The following commands are examples of how to use the DBE-Radar-Odometry pipeline. This pipeline has been evaluated on the [ColoRadar Dataset](https://arpg.github.io/coloradar/), which provides raw millimeter-wave radar data for localization and mapping.
Adjust the parameters as needed for your specific dataset and requirements.**

**Note: The dataset should be downloaded and unzipped in the `dataset` folder. The folder structure should look like this:**
```txt
├── core
├── dataset
│   ├── 12_21_2020_ec_hallways_run0
│   ├── 12_21_2020_ec_hallways_run4
│   ├── 2_22_2021_longboard_run1
│   ├── 2_23_2021_edgar_army_run5
│   ├── 2_24_2021_aspen_run0
│   ├── 2_24_2021_aspen_run9
│   ├── 2_28_2021_outdoors_run0
│   ├── calib
│   └── dataset.json
├── __init__.py
├── main.py
├── README.md
└── requirements.txt
```
**For more details on the dataset structure, refer to the [Coloradar Dataset](https://github.com/azinke/coloradar) documentation.**

1. **Draw range-azimuth map**
   
   If you want to generate the range-azimuth map like the one shown in the demo video, you can use the following command.

   <img src="doc/ICRA2025-section1.gif" alt="Radar Map" />

   single frame processing
   ```bash
    python3 coloradar.py --dataset <datasetname> -i <index> --scradar --raw -bf --save-to <path_to_save>
    ```
    batch processing of radar data
   ```bash
   python3 coloradar.py --dataset <datasetname> --scradar --raw -bf --save-to <path_to_save>
   ```

2. **Run DBE-Based Processing for Radar Point Cloud**
   
   single frame processing
   ```bash
   python3 coloradar.py --dataset <datasetname> -i <index> --scradar --raw -bfpcl --save-to <path_to_save>
   ```
   batch processing of radar data
   ```bash
   python3 coloradar.py --dataset <datasetname> --scradar --raw -bfpcl --save-to <path_to_save>
   ```

<!-- 3. **Evaluate Odometry Results**
   ```bash
   python evaluate.py --input data/processed.mat --groundtruth data/gt.txt
   ``` -->

## Odometry Evaluation

<img src="doc/ICRA2025-Exp1.gif" alt="Odometry Evaluation" />

In this project, we use two different radar-inertial odometry algorithms to evaluate the performance of the DBE-Radar-Odometry pipeline.

1. **[Filter-based Odometry](https://github.com/christopherdoer/rio)**: This is an RIO toolbox for EKF-based Radar Inertial Odometry.
2. **[Graph-based Odometry](https://github.com/ethz-asl/rio)**: Graph-based, sparse radar-inertial odometry m-estimation with barometer support and zero-velocity tracking.

Note that in this project, we only keep the doppler update and IMU in both algorithms, which means that the functions like barometer update, zero-velocity tracking, and loop closure are not included in the evaluation. To run the odometry evaluation, follow these steps:
1. **Generate ROS bag file**: Generate a ROS bag file containing radar point cloud and IMU data. Note that the two odometry algorithms require different bag formats with specific topic names and data structures, so you'll need to create separate bag files for each algorithm.
We provide a script to generate the ROS bag file.

   ```bash
   # Generate ROS bag file
   python tools/generate_bag.py -d dataset_name
   # Or with custom paths
   python tools/generate_bag.py -d dataset_name --base-dir /path/to/data --output-dir /path/to/output
   # More options
   python tools/generate_bag.py -h
   ```

   In addition, we also provide some sample ROS bag files for the ColoRadar dataset. You can find them in the `bags` folder.

2. **Run the Odometry Algorithms**: Use the generated ROS bag file as input to run the odometry algorithms. This will produce the estimated trajectory and corresponding point cloud.
The necessary configuration files are located in the `config` folder.
Modify the launch files to use the appropriate config file for each odometry algorithm.

3. **Evaluate the Odometry Results**: Use the [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation) package to evaluate the odometry results.

## Point Cloud Evaluation
The radar point cloud generated by the DBE-Radar-Odometry pipeline can be evaluated using the [Cloud_Map_Evaluation](https://github.com/HKUSTGZ-IADC/Cloud_Map_Evaluation) project.
This project provides a comprehensive evaluation framework for point clouds, including metrics such as point cloud density, completeness, and accuracy.


Before running the point cloud evaluation, prepare the radar point cloud and the ground truth point cloud. To do this, follow these steps:
1. **Generate the Radar Point Cloud**: Use the DBE-Radar-Odometry pipeline to generate the radar point cloud. Or use the provided radar point cloud in the ColoRadar dataset. Then, project the radar point cloud to the world coordinate system based on the ground truth trajectory.
2. **Generate the Ground-truth Point Cloud**: Project the lidar point cloud to the world coordinate system based on the ground truth trajectory.
3. **Run the Evaluation**: Follow the **[Usage](https://github.com/HKUSTGZ-IADC/Cloud_Map_Evaluation?tab=readme-ov-file#usage)** 
section in the Cloud_Map_Evaluation project to run the evaluation.
  
<!-- An extra step is required to generate the ground truth point cloud for evaluation. The ground truth point cloud can be generated using the `generate_gt.py` script provided in the `eval` directory. This script takes the lidar point cloud and the corresponding ground truth trajectory as input and generates a ground truth point cloud for evaluation. -->


## Acknowledgements

This project uses components from the following open-source projects:

- **[ColoRadar](https://github.com/azinke/coloradar)** (MIT License) - A public dataset for evaluating radar-based odometry.
- **[OpenRadar](https://github.com/PreSenseRadar/OpenRadar)** (Apache 2.0 License) - A radar signal processing framework.

We acknowledge and appreciate the contributions of these projects. Please refer to their respective repositories for more details.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

Additionally, since this project includes code and dependencies from OpenRadar (Apache 2.0 License) and ColoRadar (MIT License), please ensure that you comply with their licensing terms.

## Citation

If you use this work, please cite:

```
@inproceedings{jiang2025DBE,
  title={Digital Beamforming Enhanced Radar Odometry},
  author={Jiang, Jingqi and Xu, Shida and Zhang, Kaicheng and Wei, Jiyuan and Wang, Jingyang and Wang, Sen},
  booktitle = {2025 {{IEEE International Conference}} on {{Robotics}} and {{Automation}} ({{ICRA}})},
  year={2025}
}
```

## Contact

For any questions or contributions, please contact [j.jiang23@imperial.ac.uk](j.jiang23@imperial.ac.uk).
