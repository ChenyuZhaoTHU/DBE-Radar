"""Record."""
import gc
from glob import glob
import sys
import os
import multiprocessing
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    pass

from core.config import ROOTDIR, DATASET
from core.lidar import Lidar
# 注意：这里删除了 radar 的 import，移到了 load 函数里

from .utils.common import error, info
from tqdm import tqdm

def get_mcap_frame_count(mcap_path: str, target_topic: str) -> int:
    """Helper function to count messages in an MCAP file for a specific topic."""
    count = 0
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=[target_topic]):
                count += 1
    except Exception as e:
        print(f"[ERROR] Error counting MCAP frames: {e}")
        return 0
    return count

class Record:
    def __init__(self, descriptor: dict[str, dict[str, str]],
                 calibration, codename: str, index: int) -> None:
        """Init."""
        self.calibration = calibration
        self.descriptor = descriptor
        self.index = index
        self.codename: str = codename
        subdir: str = ""
        for dataset in descriptor["folders"]:
            if dataset["codename"] == codename:
                subdir = dataset["path"]
                break
        if not subdir:
            error(f"Dataset codename '{codename}' not defined in '{DATASET}")
            sys.exit(1)
        self.descriptor["paths"]["rootdir"] = os.path.join(ROOTDIR, subdir)
        self.lidar = None
        self.scradar = None
        self.ccradar = None

    def load(self, sensor: str) -> None:
        """Load the data file for a given sensor."""
        # --- 延迟加载，防止循环引用 ---
        from core.radar import SCRadar, CCRadar
        # ---------------------------

        if sensor == "lidar":
            self.lidar = Lidar(self.descriptor, self.calibration, self.index)
        elif sensor == "scradar":
            self.scradar = SCRadar(self.descriptor, self.calibration, self.index)
        elif sensor == "ccradar":
            self.ccradar = CCRadar(self.descriptor, self.calibration, self.index)

    def process_and_save(self, sensor: str, **kwargs) -> None:
        """Process and save the result into an output folder."""
        self._dpi: int = 400
        self._kwargs = kwargs
        self._sensor = sensor

        output_dir: str = kwargs.get("output", "output")
        output_dir = f"{output_dir}/{self.codename}/{sensor}"
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        
        cpu_count: int = max(1, multiprocessing.cpu_count() - 2) 
        print(f"[SYSTEM] Processing on {cpu_count} CPU(s)")

        if sensor == "lidar":
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["data"]
            )
            nb_files: int = len(os.listdir(dataset_path)) - 1
            with multiprocessing.Pool(cpu_count, maxtasksperchild=10) as pool:
                pool.map(self._process_lidar, range(1, nb_files + 1), chunksize=10)

        elif (sensor == "ccradar") or (sensor == "scradar"):
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["raw"]["data"]
            )
            
            # Check for MCAP
            mcap_files = []
            if os.path.exists(dataset_path):
                mcap_files = [f for f in os.listdir(dataset_path) if f.endswith(".mcap")]
            
            processing_range = []
            
            if mcap_files:
                mcap_file_path = os.path.join(dataset_path, mcap_files[0])
                info(f"Detected MCAP file: {mcap_files[0]}")
                
                # IMPORTANT: 这里填你正确的 Topic
                target_topic = "/radar_0/raw_data" 
                
                nb_frames = get_mcap_frame_count(mcap_file_path, target_topic)
                
                if nb_frames > 0:
                    info(f"Found {nb_frames} frames in MCAP.")
                    processing_range = range(0, nb_frames)
                else:
                    error(f"Found 0 frames for topic '{target_topic}'. Check topic name via debug_mcap.py!")
                    return
            else:
                if os.path.exists(dataset_path):
                    nb_files = len(os.listdir(dataset_path)) - 1
                    processing_range = range(1, nb_files + 1)
                else:
                    error(f"Directory not found: {dataset_path}")
                    return

            if len(processing_range) == 0:
                return

            # 多进程处理
            with multiprocessing.Pool(cpu_count, maxtasksperchild=1) as pool:
                list(tqdm(
                    pool.imap_unordered(self._process_radar, processing_range, chunksize=1),
                    total=len(processing_range)
                ))

    def _process_radar(self, idx: int) -> int:
        """Handler of radar data processing."""
        self.index = idx
        self.load(self._sensor)
        
        sensor_obj = getattr(self, self._sensor)
        if sensor_obj is None or sensor_obj.raw is None:
            return idx

        SIZE: int = 20
        plt.figure(1, clear=True, dpi=self._dpi, figsize=(SIZE, SIZE))
        
        if self._kwargs.get("beamforming"):
            if self._sensor == "scradar":
                self.scradar.show2DRangeAzimuthMap(self._kwargs.get("polar"), show=False)
            elif self._sensor == "ccradar":
                self.ccradar.show2DRangeAzimuthMap(self._kwargs.get("polar"), show=False)
        elif self._kwargs.get("beamformingPCL"):
            if self._sensor == "scradar":
                if self._kwargs.get("pointcloud"):
                    pcl = self.scradar.showPointcloudFromRawBF(
                        polar=self._kwargs.get("polar"), show=False, novisible=True
                    )
                    if pcl is not None:
                        pcl.astype(np.float32).tofile(f"{self._output_dir}/radar_pointcloud_{idx}.bin")
                    return idx
                else:
                    self.scradar.showPointcloudFromRawBF(self._kwargs.get("polar"), show=False)
            elif self._sensor == "ccradar":
                error("Beamforming Pointcloud is not supported for CCRadar")
                return idx

        elif self._kwargs.get("heatmap_3d") == False:
            if self._sensor == "scradar":
                self.scradar.show2dHeatmap(True,False)
            elif self._sensor == "ccradar":
                self.ccradar.show2dHeatmap(True,False)
        elif self._kwargs.get("heatmap_3d"):
            self.ccradar.showHeatmapFromRaw(
                self._kwargs.get("threshold"), self._kwargs.get("no_sidelobe"),
                self._kwargs.get("velocity_view"), self._kwargs.get("polar"),
                (self._kwargs.get("min_range"), self._kwargs.get("max_range")),
                (self._kwargs.get("min_azimuth"), self._kwargs.get("max_azimuth")),
                show=False,
            )
        elif self._kwargs.get("pointcloud"):
             # ... Pointcloud save logic ...
             pass
        
        plt.savefig(f"{self._output_dir}/radar_{idx:04}.jpg", dpi=self._dpi)
        plt.close('all')
        plt.clf()
        gc.collect()
        return idx

    def _process_lidar(self, idx: int) -> int:
        """Handler of lidar data processing.

        Used as the handler for parallel processing. The context attributes
        needed by this method are only defined in the method `process_and_save`
        As so, only that method is supposed to call this one.

        NOTE: THIS METHOD IS NOT EXPECTED TO BE CALLED FROM OUTSIDE OF THIS
        CLASS

        Argument:
            idx: Index of the file to process
        """
        self.index = idx
        self.load(self._sensor)
        if self._kwargs.get("polar"):
            bev = self.lidar.getPolarBirdView(
                self._kwargs.get("resolution", 0.1),
                0.4,  # 0.5 degree angle resolution
                (1.0, 13.5),
                (-90, 90),
                # (-args.width/2, args.width/2),
                # (-args.height/2, args.height/2),
            )
            target_size = (2560, 1920)  
            bev = cv.resize(bev, target_size, interpolation=cv.INTER_LINEAR) 
            bev = cv.flip(bev, 0)  
            bev = cv.flip(bev, 1)

        else:
            bev = self.lidar.getBirdEyeView(
                self._kwargs.get("resolution", 0.1),
                self._kwargs.get("srange"),
                self._kwargs.get("frange"),
            )
        plt.imsave(f"{self._output_dir}/lidar_bev_{idx:04}.jpg", bev)

    def make_video(self, inputdir: str, ext: str = "jpg") -> None:
        """Make video out of pictures"""
        files = glob(inputdir + f"/*.{ext}")
        files = sorted(files)
        height, width, _ = plt.imread(files[0]).shape
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        video = cv.VideoWriter(inputdir + f"/{self.codename}.avi", fourcc, 10, (width, height))
        for idx, img in enumerate(files):
            print(
                f"[ ========= {100 * idx/len(files): 2.2f}% ========= ]\r",
                end=""
            )
            video.write(cv.imread(img))
        cv.destroyAllWindows()
        video.release()
