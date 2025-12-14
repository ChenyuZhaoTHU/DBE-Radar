"""Radar.
SCRadar & CCRadar: MCAP + Robust Calibration + Singular Matrix Fix.
"""
from typing import Optional, Any
import os

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except ImportError:
    pass

import numpy as np
import matplotlib.pyplot as plt

from core.calibration import Calibration, SCRadarCalibration
from core.utils.common import error, info
from core.utils import radardsp as rdsp
from .lidar import Lidar

from core.config import *
import core.dsp as dsp
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class SCRadar(Lidar):
    NUMBER_RECORDING_ATTRIBUTES: int = 5
    CFAR_WS: int = 16
    CFAR_GC: int = 8
    AZIMUTH_FOV: float = np.deg2rad(180)
    ELEVATION_FOV: float = np.deg2rad(20)
    
    HARDWARE_NUM_TX: int = 3
    HARDWARE_NUM_RX: int = 4

    def __init__(self, config: dict[str, str],
                 calib: Calibration, index: int) -> None:
        """Init."""
        self.sensor: str = self.__class__.__name__.lower()
        self.calibration: SCRadarCalibration = getattr(calib, self.sensor)
        self.config = config
        self.index = index

        self.cld = self._load(index, "pointcloud", np.float32, (-1, self.NUMBER_RECORDING_ATTRIBUTES))
        self.heatmap = self._load(index, "heatmap", np.float32, (
                self.calibration.heatmap.num_elevation_bins,
                self.calibration.heatmap.num_azimuth_bins,
                self.calibration.heatmap.num_range_bins, 2))

        self.raw = self._load_mcap_raw(index)

        if self.raw is None:
            raw = self._load(index, "raw", np.int16, (
                    self.calibration.waveform.num_tx,
                    self.calibration.waveform.num_rx,
                    self.calibration.waveform.num_chirps_per_frame,
                    self.calibration.waveform.num_adc_samples_per_chirp, 2))
            if raw is not None:
                I = np.float16(raw[:, :, :, :, 0])
                Q = np.float16(raw[:, :, :, :, 1])
                self.raw = I + 1j * Q
        
        if self.raw is not None:
            curr_tx, curr_rx, curr_loops, curr_samples = self.raw.shape
            if self.calibration.waveform.num_tx != curr_tx:
                self.calibration.waveform.num_tx = curr_tx
            if self.calibration.waveform.num_chirps_per_frame != curr_loops:
                self.calibration.waveform.num_chirps_per_frame = curr_loops
            if self.calibration.waveform.num_adc_samples_per_chirp != curr_samples:
                self.calibration.waveform.num_adc_samples_per_chirp = curr_samples

    def _load_mcap_raw(self, target_index: int) -> Optional[np.array]:
        try:
            raw_dir = os.path.join(self.config["paths"]["rootdir"], self.config["paths"][self.sensor]["raw"]["data"])
            if not os.path.exists(raw_dir): return None
            mcap_files = [f for f in os.listdir(raw_dir) if f.endswith(".mcap")]
            if not mcap_files: return None
            mcap_path = os.path.join(raw_dir, mcap_files[0])
        except Exception: return None

        target_topic = "/radar_0/raw_data" 
        radar_msg = None
        current_idx = 0

        try:
            with open(mcap_path, "rb") as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
                    if channel.topic == target_topic:
                        if current_idx == target_index:
                            radar_msg = decoded_msg
                            break
                        current_idx += 1
        except Exception as e:
            print(f"[ERROR] Reading MCAP Frame {target_index}: {e}")
            return None

        if radar_msg is None: return None
        return self._parse_ros_radar_msg(radar_msg)

    def _parse_ros_radar_msg(self, msg: Any) -> np.array:
        n_chirps_total = getattr(msg, 'num_chirps', 48)
        n_rx = getattr(msg, 'num_rx', 4)
        n_samples = getattr(msg, 'num_samples', 256)
        n_tx = self.HARDWARE_NUM_TX 
        
        raw_data_flat = np.array(msg.data, dtype=np.int16)
        try:
            raw_reshaped = raw_data_flat.reshape(n_chirps_total, n_rx, n_samples, 2)
        except ValueError:
            return None

        data_complex = raw_reshaped[:, :, :, 0] + 1j * raw_reshaped[:, :, :, 1]
        
        if n_tx > 0:
            n_loops = n_chirps_total // n_tx
            data_tdm = data_complex.reshape(n_loops, n_tx, n_rx, n_samples)
            data_final = data_tdm.transpose(1, 2, 0, 3) 
        else:
             data_final = data_complex
        return data_final

    def _safe_get_calibration(self, type_name: str):
        try:
            if type_name == "coupling": return self.calibration.get_coupling_calibration()
            elif type_name == "frequency": return self.calibration.get_frequency_calibration()
            elif type_name == "phase": return self.calibration.get_phase_calibration()
            return 0
        except ValueError: return 0

    def _calibrate(self) -> np.array:
        if self.raw is None: raise ValueError("No raw data available")
        adc_samples = self.raw.copy()
        adc_samples -= np.mean(adc_samples, axis=3)[:,:,:,np.newaxis]

        if self.sensor != "scradar":
            try:
                adc_samples *= self.calibration.get_frequency_calibration()
                adc_samples *= self.calibration.get_phase_calibration()
            except ValueError: pass
        return adc_samples

    def _generate_range_azimuth_heatmap(self) -> np.array:
        adc_samples = self._calibrate()
        ntx, nrx, nc, ns = adc_samples.shape
        _, _, Nc, Ns = self._get_fft_size(None, None, nc, ns)

        rfft = np.fft.fft(adc_samples, Ns, -1)
        
        try:
            coupling_calib = self.calibration.get_coupling_calibration()
            rfft -= coupling_calib
        except ValueError: pass
        
        _rfft = self._pre_process(rfft, False)
        
        if _rfft.ndim == 3:
            _rfft = np.expand_dims(_rfft, axis=0)
            
        _rfft = _rfft[0, :, :, :]
        _rfft = np.swapaxes(_rfft, 0, 1)

        BINS_PROCESSED = _rfft.shape[2]
        num_vec, steering_vec = dsp.gen_steering_vec(AZ_RANGE, ANGLE_RES_AZ, 8)
        range_azimuth = np.zeros((AZ_BINS, BINS_PROCESSED))
        
        # [关键修复]：Capon 失败时自动回退
        method = 'Capon'
        
        for i in range(BINS_PROCESSED):
            try:
                if method == 'Capon':
                    # 尝试 Capon，内部如果抛出 LinAlgError 会被捕获
                    # 注意：如果 dsp.aoa_capon 内部没有正则化，我们需要在这里加
                    # 为了不修改 dsp 库，我们这里只捕获异常
                    range_azimuth[:,i], _ = dsp.aoa_capon(_rfft[:, :, i].T, steering_vec, magnitude=True)
                elif method == 'Bartlett':
                    range_azimuth_multi = dsp.aoa_bartlett(steering_vec, _rfft[:, :, i], axis =1)
                    range_azimuth[:,i] = np.abs(range_azimuth_multi.sum(0)).squeeze()
            except np.linalg.LinAlgError:
                # 矩阵奇异，回退到 Bartlett (无需求逆，最稳定)
                # print(f"[WARN] Singular matrix at bin {i}, fallback to Bartlett")
                range_azimuth_multi = dsp.aoa_bartlett(steering_vec, _rfft[:, :, i], axis =1)
                range_azimuth[:,i] = np.abs(range_azimuth_multi.sum(0)).squeeze()
            except Exception:
                # 其他任何计算错误，填 0
                range_azimuth[:,i] = 0

        return range_azimuth

    def show2DRangeAzimuthMap(self, polar: bool = False, show: bool = True, method: str = "Capon") -> None:
        if self.raw is None: return
        RAmap = self._generate_range_azimuth_heatmap()
        
        # --- 优化显示逻辑 ---
        # 1. 转换为分贝 (dB)
        # 加一个极小值 1e-10 防止 log(0)
        RAmap_log = 20 * np.log10(np.abs(RAmap) + 1e-10)
        
        # 2. 动态归一化 (关键步骤)
        # 找出最大值
        max_val = np.max(RAmap_log)
        # 将最大值设为 0 dB，并将底噪设为 -35 dB (你可以调整这个 -35)
        # 这样可以过滤掉大部分背景噪声，只显示强的目标
        RAmap_norm = RAmap_log - max_val
        vmin = -35  # 显示范围下限 (-35dB)
        vmax = 0    # 显示范围上限 (0dB)
        
        # 3. 翻转与转置以匹配坐标系
        Nr = RAmap.shape[1]
        abins = np.linspace(-AZ_RANGE, AZ_RANGE, AZ_BINS) * (np.pi/180)
        rbins, _, _, _ = self._get_bins(Nr, None, None, None)
        
        # 注意：这里的数据排布可能需要转置，视具体情况而定
        # 通常是 (Azimuth, Range)，我们需要转置为 (Range, Azimuth) 用于绘图
        plot_data = RAmap_norm.T 
        
        if not polar:
            # 笛卡尔坐标系 (Range-Azimuth)
            # 使用 extent 参数直接映射物理坐标
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 使用 imshow 绘图，设置 origin='lower' 让 0 米在底部
            # aspect='auto' 填充整个图框
            im = ax.imshow(
                plot_data, 
                cmap="jet",  # 使用 jet 或 viridis
                vmin=vmin, vmax=vmax,
                origin='lower',
                extent=[np.rad2deg(abins[0]), np.rad2deg(abins[-1]), rbins[0], rbins[-1]],
                aspect='auto'
            )
            
            ax.set_xlabel("Azimuth (deg)")
            ax.set_ylabel("Range (m)")
            ax.set_title(f"Range-Azimuth Heatmap (Frame {self.index})")
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Power (dB)")
            
        else:
            # 极坐标图 (Polar Plot)
            # 这种画法更直观，像真实的雷达扫描
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='polar')
            
            # 创建网格
            az_grid, r_grid = np.meshgrid(abins, rbins)
            
            # 使用 pcolormesh 绘制极坐标热图
            # 注意: pcolormesh 需要 (Range, Azimuth) 对应的数据
            im = ax.pcolormesh(
                az_grid, r_grid, plot_data, 
                cmap="jet", 
                vmin=vmin, vmax=vmax,
                shading='auto'
            )
            
            ax.set_theta_zero_location("N") # 0度在正北（上方）
            ax.set_theta_direction(-1)      # 顺时针增加角度 (符合常规雷达习惯)
            # ax.set_rlabel_position(45)    # 距离标签位置
            
            ax.set_title(f"Polar Range-Azimuth Map (Frame {self.index})", va='bottom')
            
            cbar = plt.colorbar(im, ax=ax, pad=0.1)
            cbar.set_label("Power (dB)")

        # 4. 保存或显示
        if show: 
            plt.show()
        else:
            # 如果不显示，直接关闭，防止内存泄漏
            # (注意：这里的 plt.close 在 record.py 里也有调用，这里只是为了完整性)
            pass

    def _load(self, index: int, ftype: str, dtype: np.dtype, shape: tuple[int, ...]) -> Optional[np.array]:
        filename = self._filename(self.config["paths"][self.sensor][ftype]["filename_prefix"], index, "bin")
        filepath = os.path.join(self.config["paths"]["rootdir"], self.config["paths"][self.sensor][ftype]["data"], filename)
        try:
            data = np.fromfile(filepath, dtype)
            data = np.reshape(data, shape)
        except Exception: data = None
        return data

    def showHeatmap(self, threshold: float = 0.15, no_sidelobe: bool = False, render: bool = True) -> None: pass
    def showHeatmapBirdEyeView(self, threshold: float) -> None: pass 
    def _polar_to_cartesian(self, r_idx, az_idx, el_idx) -> np.array: return np.zeros(3)
    def _to_cartesian(self, hmap) -> np.array: return hmap
    def _heatmap_to_pointcloud(self, threshold, no_sidelobe) -> np.array: return np.zeros((1,5))

    def _get_fft_size(self, ne, na, nc, ns):
        ne = rdsp.fft_size(ne) if ne else NUMBER_ELEVATION_BINS_MIN
        na = rdsp.fft_size(na) if na else NUMBER_AZIMUTH_BINS_MIN
        nc = rdsp.fft_size(nc) if nc else NUMBER_DOPPLER_BINS_MIN
        ns = rdsp.fft_size(ns) if ns else NUMBER_RANGE_BINS_MIN
        return ne, na, nc, ns

    def _get_bins(self, ns, nc, na, ne):
        ntx = self.calibration.antenna.num_tx
        fs = self.calibration.waveform.adc_sample_frequency
        fslope = self.calibration.waveform.frequency_slope
        fstart = self.calibration.waveform.start_frequency
        te = self.calibration.waveform.ramp_end_time
        tc = self.calibration.waveform.idle_time + te
        rbins = rdsp.get_range_bins(ns, fs, fslope) if ns else []
        vbins = rdsp.get_velocity_bins(ntx, nc, fstart, tc) if nc else []
        abins, ebins = [], []
        if na: abins = -1 * np.arcsin(np.arange(-self.AZIMUTH_FOV, self.AZIMUTH_FOV, 2*self.AZIMUTH_FOV/na) / (2*np.pi*self.calibration.d))
        if ne: ebins = -1 * np.arcsin(np.arange(-self.ELEVATION_FOV, self.ELEVATION_FOV, 2*self.ELEVATION_FOV/ne) / (2*np.pi*self.calibration.d))
        return rbins, vbins, abins, ebins

    def _pre_process(self, adc_samples, apply_padding=True):
        virtual_array = rdsp.virtual_array(adc_samples, self.calibration.antenna.txl, self.calibration.antenna.rxl)
        if apply_padding:
            Ne, Na, Nc, Ns = self._get_fft_size(*virtual_array.shape)
            va_nel, va_naz, va_nc, va_ns = virtual_array.shape
            virtual_array = np.pad(virtual_array, ((0, Ne-va_nel), (0, Na-va_naz), (0, Nc-va_nc), (0, Ns-va_ns)), "constant")
        return virtual_array
    
    def getPointcloudFromRaw(self, polar=False): return np.array([])
    def showPointcloudFromRaw(self, *args, **kwargs): pass
    def showHeatmapFromRaw(self, *args, **kwargs): pass
    def show2dHeatmap(self, *args, **kwargs): pass
    def showPointcloudFromRawBF(self, *args, **kwargs): return None

class CCRadar(SCRadar):
    CFAR_WS: int = 32
    CFAR_GC: int = 16
    AZIMUTH_FOV: float = np.deg2rad(180)
    ELEVATION_FOV: float = np.deg2rad(20)
    def _phase_calibration(self) -> np.array:
        pm = self._safe_get_calibration("phase")
        if np.isscalar(pm): return self.raw
        return self.raw