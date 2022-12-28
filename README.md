# Digital Beamforming Enhanced Radar Odometry

<img src="doc/ICRA25_2261_VI_i.gif" alt="Video Demo">

Digital Beamforming Enhanced Radar Odometry (**DBE-Radar-Odometry**) is an advanced radar signal processing pipeline that enhances radar-based odometry and SLAM systems by integrating spatial domain beamforming techniques. This project provides an alternative to traditional FFT-based radar processing, improving the precision of radar localization and mapping.


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
- Dependencies:
  ```bash
  python -m venv venv
  source ./venv/bin/activate

  python -m pip install -r requirements.txt --extra-index-url https://rospypi.github.io/simple/
  ```


## Usage
**The following commands are examples of how to use the DBE-Radar-Odometry pipeline. Adjust the parameters as needed for your specific dataset and requirements.**

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
   
   single frame processing
   ```bash
    python3 coloradar.py --dataset <datasetname> -i <index> --scradar --raw -bf --save-to ./dataset/bfpcltest/
    ```
    batch processing of radar data
   ```bash
   python3 coloradar.py --dataset <datasetname> --scradar --raw -bf --save-to ./dataset/bfpcltest/
   ```

2. **Run DBE-Based Processing for Radar Point Cloud**
   
   single frame processing
   ```bash
   python3 coloradar.py --dataset <datasetname> -i <index> --scradar --raw -bfpcl --save-to ./dataset/bfpcltest/
   ```
   batch processing of radar data
   ```bash
   python3 coloradar.py --dataset <datasetname> --scradar --raw -bfpcl --save-to ./dataset/bfpcltest/
   ```

<!-- 3. **Evaluate Odometry Results**
   ```bash
   python evaluate.py --input data/processed.mat --groundtruth data/gt.txt
   ``` -->

## Dataset

This project is evaluated on the [ColoRadar Dataset](https://arpg.github.io/coloradar/), which provides raw millimeter-wave radar data for localization and mapping.

The radar-inertial odometry algorithms used in this project are based on 
a [filter-based one](https://github.com/christopherdoer/rio) and a [graph-based one](https://github.com/ethz-asl/rio).

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