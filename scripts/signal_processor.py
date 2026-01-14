import numpy as np
from collections import deque
import rclpy.logging
from radarstack_msgs.msg import RadarFrame

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = rclpy.logging.get_logger(__name__)

# Numba-accelerated helper functions for Cartesian conversion
@jit(nopython=True, cache=True)
def _convert_polar_to_cartesian_xy_jit(range_values, azimuth_angles, ra_2d,
                                       x_coords, y_coords, cartesian_xy, max_xy):
    """Numba-accelerated polar to Cartesian X-Y conversion."""
    for r_idx in range(len(range_values)):
        r = range_values[r_idx]
        if r > max_xy:
            continue
        for a_idx in range(len(azimuth_angles)):
            a = azimuth_angles[a_idx]
            x = r * np.sin(a)
            y = r * np.cos(a)
            
            if abs(x) > max_xy or y < 0.0 or y > max_xy:
                continue
            
            # Linear search for nearest grid point (fast with numba)
            x_idx = 0
            min_dist = abs(x_coords[0] - x)
            for i in range(1, len(x_coords)):
                dist = abs(x_coords[i] - x)
                if dist < min_dist:
                    min_dist = dist
                    x_idx = i
            
            y_idx = 0
            min_dist = abs(y_coords[0] - y)
            for i in range(1, len(y_coords)):
                dist = abs(y_coords[i] - y)
                if dist < min_dist:
                    min_dist = dist
                    y_idx = i
            
            if 0 <= y_idx < len(y_coords) and 0 <= x_idx < len(x_coords):
                val = ra_2d[r_idx, a_idx]
                if val > cartesian_xy[y_idx, x_idx]:
                    cartesian_xy[y_idx, x_idx] = val

@jit(nopython=True, cache=True)
def _convert_polar_to_cartesian_xz_jit(range_values, elevation_angles, re_2d,
                                      x_coords, z_coords, cartesian_xz, max_grid_x, max_grid_z):
    """Numba-accelerated polar to Cartesian X-Z conversion."""
    for r_idx in range(len(range_values)):
        r = range_values[r_idx]
        if r > max_grid_x:
            continue
        for e_idx in range(len(elevation_angles)):
            e = elevation_angles[e_idx]
            x = r * np.cos(e)
            z = r * np.sin(e)
            
            if x < 0.0 or x > max_grid_x or abs(z) > max_grid_z:
                continue
            
            # Linear search for nearest grid point (fast with numba)
            x_idx = 0
            min_dist = abs(x_coords[0] - x)
            for i in range(1, len(x_coords)):
                dist = abs(x_coords[i] - x)
                if dist < min_dist:
                    min_dist = dist
                    x_idx = i
            
            z_idx = 0
            min_dist = abs(z_coords[0] - z)
            for i in range(1, len(z_coords)):
                dist = abs(z_coords[i] - z)
                if dist < min_dist:
                    min_dist = dist
                    z_idx = i
            
            if 0 <= z_idx < len(z_coords) and 0 <= x_idx < len(x_coords):
                val = re_2d[r_idx, e_idx]
                if val > cartesian_xz[z_idx, x_idx]:
                    cartesian_xz[z_idx, x_idx] = val

class SignalProcessor:
    """
    Handles radar signal processing: reshaping, buffering, and algorithm execution.
    Designed to be configured via RadarInfo dictionaries.
    """
    
    def __init__(self, radar_params, buffer_size=1):
        """
        Args:
            radar_params (dict): Normalized radar parameters (from RadarInfo)
            buffer_size (int): Number of frames to buffer. Set to 1 to process each frame independently
        """
        self.params = radar_params
        self.buffer = deque(maxlen=buffer_size)
        
        # Pre-compute/Extract constants for speed
        self.n_chirps = radar_params['n_chirps']
        self.n_rx = radar_params['n_rx']
        self.n_samples = radar_params['n_samples']
        self.n_tx = radar_params['n_tx']
        self.rx_mask = np.array(radar_params['rx_mask_list'], dtype=np.int32)
        self.tx_mask = np.array(radar_params['tx_mask_list'], dtype=np.int32)
        
        # Handle phase bias
        # If list, convert to array. If None, create zeros.
        bias_list = radar_params.get('rx_phase_bias', [])
        if not bias_list:
            # Default: identity bias (1+0j)
            # We need one bias per virtual antenna (n_tx * n_rx possible combinations)
            # But wait, the DSP function expects specific mapping.
            # Let's assume a flat list matching the virtual array size.
            # If empty, we use 1.0 + 0.0j for all
             # total virtual antennas (max)
            max_virtual = len(self.rx_mask) * len(self.tx_mask) 
            self.rx_phase_bias = np.ones(max_virtual, dtype=np.complex64)
        else:
             # Convert [real, imag, real, imag...] to complex array
            self.rx_phase_bias = np.array([
                a + 1j * b for a, b in zip(bias_list[0::2], bias_list[1::2])
            ], dtype=np.complex64)

    def _convert_to_complex_cube(self, data: np.ndarray, num_chirps: int, num_rx: int, 
                                  num_samples: int, is_complex: bool, interleaved: bool) -> np.ndarray:
        """
        Convert raw int16 data to standardized complex64 radar cube.
        
        Args:
            data: Raw int16 array of ADC samples
            num_chirps: Number of chirps in frame
            num_rx: Number of RX antennas
            num_samples: Number of ADC samples per chirp
            is_complex: True for I/Q complex data, false for real-only
            interleaved: True for interleaved format (IIQQ), false for non-interleaved
            
        Returns:
            Complex64 array with shape [num_chirps, num_rx, num_samples]
            
        Raises:
            ValueError: If data length doesn't match expected dimensions
        """
        # Calculate expected data length
        C = 2 if is_complex else 1
        expected_length = num_chirps * num_rx * num_samples * C
        
        # Validate data length
        if len(data) != expected_length:
            raise ValueError(
                f"Data length mismatch: expected {expected_length}, got {len(data)}. "
                f"Dimensions: chirps={num_chirps}, rx={num_rx}, samples={num_samples}, "
                f"is_complex={is_complex}"
            )
        
        # Handle different data formats
        if not is_complex:
            # Real data: simple reshape to [chirps, rx, samples], then convert to complex
            cube = data.reshape(num_chirps, num_rx, num_samples)
            # Convert to complex (imaginary part = 0)
            cube = cube.astype(np.complex64)
        else:
            # Complex data: extract I and Q components
            if interleaved:
                # IIQQ format: DCA1000 2-lane LVDS interleaved data format
                # Data layout: [Q0, Q1, I0, I1, Q2, Q3, I2, I3, ...]
                # Reshape to [num_chirps, num_rx, num_samples * 2] to expose the interleaved dimension
                iiqq = data.reshape(num_chirps, num_rx, num_samples * 2)
                
                # De-interleave IIQQ format to reconstruct complex samples:
                # For even sample indices (0, 2, 4, ...): Q from position 0, I from position 2
                # For odd sample indices (1, 3, 5, ...): Q from position 1, I from position 3
                cube = np.zeros((num_chirps, num_rx, num_samples), dtype=np.complex64)
                cube[:, :, 0::2] = 1j * iiqq[:, :, 0::4].astype(np.float32) + iiqq[:, :, 2::4].astype(np.float32)
                cube[:, :, 1::2] = 1j * iiqq[:, :, 1::4].astype(np.float32) + iiqq[:, :, 3::4].astype(np.float32)
            else:
                # Non-interleaved format: [I0, I1, ..., Q0, Q1, ...]
                half = len(data) // 2
                I = data[:half].reshape(num_chirps, num_rx, num_samples)
                Q = data[half:].reshape(num_chirps, num_rx, num_samples)
                cube = I.astype(np.float32) + 1j * Q.astype(np.float32)
                cube = cube.astype(np.complex64)
        
        return cube

    def _construct_virtual_array(self, rd: np.ndarray) -> np.ndarray:
        """
        Construct TDM-MIMO virtual array from range-doppler spectrum.
        
        This function maps physical TX-RX antenna pairs to a virtual array geometry
        optimized for elevation and azimuth angle estimation. The virtual array is
        constructed based on the TDM-MIMO chirp sequence and antenna spacing.
        
        Args:
            rd: Range-doppler spectrum with shape [batch, doppler, tx, rx, range]
                where tx=3, rx=4 for AWR1843Boost
                
        Returns:
            MIMO virtual array with shape [batch, doppler, elevation, azimuth, range]
            where elevation=2, azimuth=8
            
        Virtual Array Geometry:
            The virtual array organizes antenna data into elevation and azimuth dimensions
            based on the physical antenna layout and TDM-MIMO chirp sequence:
            - Elevation row 0: Uses TX3 data (positions 2-5) for elevation estimation
            - Azimuth row 1: Uses TX1 data (positions 0-3) and TX2 data (positions 4-7) for azimuth estimation
            
        Note:
            This mapping is specific to AWR1843Boost antenna configuration with 3 TX and 4 RX antennas.
        """
        batch, doppler, tx, rx, range_bins = rd.shape
        if tx != 3 or rx != 4:
            raise ValueError(f"Expected (tx, rx)=(3, 4), got tx={tx} and rx={rx}.")
        
        mimo = np.zeros((batch, doppler, 2, 8, range_bins), dtype=np.complex64)
        # Map TX3 to elevation row 0 (positions 2-5)
        mimo[:, :, 0, 2:6, :] = rd[:, :, 1, :, :]
        # Map TX1 to azimuth row 1 (positions 0-3)
        mimo[:, :, 1, 0:4, :] = rd[:, :, 0, :, :]
        # Map TX2 to azimuth row 1 (positions 4-7)
        mimo[:, :, 1, 4:8, :] = rd[:, :, 2, :, :]
        return mimo

    def build_radar_cube(self, data: np.ndarray, num_chirps: int, num_rx: int, 
                         num_samples: int, is_complex: bool, interleaved: bool, 
                         tdm_mimo: bool, num_tx: int) -> np.ndarray:
        """
        Build radar cube from raw parameters.
        
        Args:
            data: int16 array of raw ADC data
            num_chirps: Number of chirps (384 in example)
            num_rx: Number of RX antennas (4)
            num_samples: Number of samples per chirp (128)
            is_complex: Whether data is complex (True)
            interleaved: Whether data is interleaved (True for IIQQ)
            tdm_mimo: Whether TDM-MIMO is enabled
            num_tx: Number of TX antennas (3 for AWR1843Boost)
        
        Returns:
            Complex64 array:
            - TDM-MIMO: [loops, tx, rx, samples] where loops = num_chirps / num_tx
            - Non-TDM-MIMO: [chirps, rx, samples]
        """
        # 1. Convert to complex cube [num_chirps, num_rx, num_samples]
        cube = self._convert_to_complex_cube(data, num_chirps, num_rx, num_samples, 
                                             is_complex, interleaved)
        
        # 2. Reshape based on TDM-MIMO
        if tdm_mimo:
            n_loops = num_chirps // num_tx
            if num_chirps % num_tx != 0:
                raise ValueError(f"num_chirps ({num_chirps}) must be divisible by num_tx ({num_tx})")
            return cube.reshape(n_loops, num_tx, num_rx, num_samples)
        else:
            return cube  # [chirps, rx, samples]

    def _apply_window(self, data: np.ndarray, axis: int, window: bool) -> np.ndarray:
        """Apply Hann window if requested."""
        if not window:
            return data
        hann = np.hanning(data.shape[axis] + 2).astype(np.float32)[1:-1]
        hann = hann / np.mean(hann)
        # Broadcast window to match data shape
        window_shape = [1] * data.ndim
        window_shape[axis] = -1
        return data * hann.reshape(*window_shape)
    
    def _fft_with_shift(self, data: np.ndarray, axes: tuple, size: tuple = None, shift: tuple = None) -> np.ndarray:
        """
        Compute FFT along specified axes with zero-padding and optional fftshift.
        Optimized to avoid unnecessary copies.
        
        Args:
            data: Input data
            axes: Axes along which to compute FFT
            size: Target size for each axis (for zero-padding), must match axes length
            shift: Axes to shift after FFT (if None, no shift).
                   - For range-doppler: shift=(1,) shifts doppler axis only, keeping range 0 at index 0
                   - For elevation-azimuth: shift=(2, 3) shifts both elevation and azimuth axes
        """
        result = data  # Don't copy unless we need to modify
        
        if size is not None:
            # Zero-pad if needed - only copy if modification is needed
            needs_copy = False
            for axis, target_size in zip(axes, size):
                if target_size != result.shape[axis]:
                    needs_copy = True
                    break
            
            if needs_copy:
                result = result.copy()  # Only copy if we need to modify
                for axis, target_size in zip(axes, size):
                    if target_size > result.shape[axis]:
                        pad_shape = list(result.shape)
                        pad_shape[axis] = target_size - result.shape[axis]
                        pad = np.zeros(pad_shape, dtype=result.dtype)
                        result = np.concatenate([result, pad], axis=axis)
                    elif target_size < result.shape[axis]:
                        # Truncate
                        slices = [slice(None)] * result.ndim
                        slices[axis] = slice(0, target_size)
                        result = result[tuple(slices)]
        
        # Compute FFT along each axis (in-place when possible)
        for axis in axes:
            result = np.fft.fft(result, axis=axis)
        
        # Apply fftshift only to specified axes
        if shift is not None:
            result = np.fft.fftshift(result, axes=shift)
        
        return result

    def _convert_to_cartesian_xy(self, ra_2d: np.ndarray, range_max: float, azimuth_bins: int = 128) -> np.ndarray:
        """
        Convert Range-Azimuth to Cartesian X-Y (top-down view) - Numba accelerated.
        
        Args:
            ra_2d: Range-Azimuth heatmap [range, azimuth]
            range_max: Maximum range in meters
            azimuth_bins: Number of azimuth bins (default 128)
        
        Returns:
            Cartesian 2D heatmap [y, x] where:
            - x: lateral position (left-right, negative=left, positive=right)
            - y: forward position (ahead of radar, 0=radar position)
        """
        range_bins, _ = ra_2d.shape
        grid_resolution = 0.1  # meters
        
        # Limit grid size to reasonable maximum (100m x 100m) to avoid memory issues
        max_grid_size = 100.0  # meters
        max_xy = min(range_max, max_grid_size) if range_max > 0 else max_grid_size
        
        # Create angle and range arrays
        azimuth_angles = np.linspace(-np.pi/2, np.pi/2, azimuth_bins, dtype=np.float32)
        range_values = np.linspace(0, range_max, range_bins, dtype=np.float32)
        
        # Create Cartesian grid with limited size
        x_coords = np.arange(-max_xy, max_xy + grid_resolution, grid_resolution, dtype=np.float32)
        y_coords = np.arange(0, max_xy + grid_resolution, grid_resolution, dtype=np.float32)
        
        # Initialize grid
        cartesian_xy = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)
        
        # Use numba-accelerated conversion
        if NUMBA_AVAILABLE:
            _convert_polar_to_cartesian_xy_jit(
                range_values, azimuth_angles, ra_2d,
                x_coords, y_coords, cartesian_xy, max_xy
            )
        else:
            # Fallback to Python implementation
            for r_idx, r in enumerate(range_values):
                if r > max_xy:
                    continue
                for a_idx, a in enumerate(azimuth_angles):
                    x = r * np.sin(a)
                    y = r * np.cos(a)
                    
                    if abs(x) > max_xy or y < 0 or y > max_xy:
                        continue
                    
                    x_idx = np.argmin(np.abs(x_coords - x))
                    y_idx = np.argmin(np.abs(y_coords - y))
                    
                    if 0 <= y_idx < len(y_coords) and 0 <= x_idx < len(x_coords):
                        cartesian_xy[y_idx, x_idx] = np.maximum(
                            cartesian_xy[y_idx, x_idx], ra_2d[r_idx, a_idx]
                        )
        
        return cartesian_xy

    def _convert_to_cartesian_xz(self, re_2d: np.ndarray, range_max: float, elevation_bins: int) -> np.ndarray:
        """
        Convert Range-Elevation to Cartesian X-Z (side view) - Numba accelerated.
        
        Args:
            re_2d: Range-Elevation heatmap [range, elevation]
            range_max: Maximum range in meters
            elevation_bins: Number of elevation bins
        
        Returns:
            Cartesian 2D heatmap [z, x] where:
            - x: forward range (distance ahead of radar, 0=radar position)
            - z: vertical position (up-down, positive=up, negative=down)
        """
        range_bins, _ = re_2d.shape
        grid_resolution = 0.1  # meters
        
        # Limit grid size to reasonable maximum (100m forward, 50m vertical) to avoid memory issues
        max_grid_x = min(range_max, 100.0) if range_max > 0 else 100.0  # meters forward
        max_grid_z = 50.0   # meters vertical
        
        # Estimate elevation angle range (typically -30° to +30° for AWR1843Boost)
        elevation_max_deg = 30.0  # degrees
        elevation_angles = np.linspace(-np.deg2rad(elevation_max_deg), 
                                       np.deg2rad(elevation_max_deg), 
                                       elevation_bins, dtype=np.float32)
        range_values = np.linspace(0, range_max, range_bins, dtype=np.float32)
        
        # Create Cartesian grid for side view with limited size
        x_coords = np.arange(0, max_grid_x + grid_resolution, grid_resolution, dtype=np.float32)
        z_coords = np.arange(-max_grid_z, max_grid_z + grid_resolution, grid_resolution, dtype=np.float32)
        
        # Initialize grid
        cartesian_xz = np.zeros((len(z_coords), len(x_coords)), dtype=np.float32)
        
        # Use numba-accelerated conversion
        if NUMBA_AVAILABLE:
            _convert_polar_to_cartesian_xz_jit(
                range_values, elevation_angles, re_2d,
                x_coords, z_coords, cartesian_xz, max_grid_x, max_grid_z
            )
        else:
            # Fallback to Python implementation
            for r_idx, r in enumerate(range_values):
                if r > max_grid_x:
                    continue
                for e_idx, e in enumerate(elevation_angles):
                    x = r * np.cos(e)
                    z = r * np.sin(e)
                    
                    if x < 0 or x > max_grid_x or abs(z) > max_grid_z:
                        continue
                    
                    x_idx = np.argmin(np.abs(x_coords - x))
                    z_idx = np.argmin(np.abs(z_coords - z))
                    
                    if 0 <= z_idx < len(z_coords) and 0 <= x_idx < len(x_coords):
                        cartesian_xz[z_idx, x_idx] = np.maximum(
                            cartesian_xz[z_idx, x_idx], re_2d[r_idx, e_idx]
                        )
        
        return cartesian_xz

    def process_frame(self, msg: RadarFrame):
        """
        Main signal processing pipeline: Convert -> Reshape -> FFT -> Virtual Array -> FFT -> Extract
        
        This method implements the complete radar signal processing chain:
        1. Convert raw interleaved data to complex cube
        2. Reshape to TDM-MIMO structure (separate TX antennas)
        3. Apply Range-Doppler FFT (along doppler and range dimensions)
        4. Construct virtual array from physical antenna data
        5. Apply Elevation-Azimuth FFT (along elevation and azimuth dimensions)
        6. Extract 2D heatmaps by averaging over appropriate dimensions
        
        Args:
            msg: RadarFrame message with raw ADC data
            
        Returns:
            Dictionary with 2D arrays:
            - 'range_doppler': [range, doppler] - velocity vs distance
            - 'range_azimuth': [range, azimuth] - angle vs distance
            - 'range_elevation': [range, elevation] - elevation angle vs distance
            - 'doppler_azimuth': [azimuth, doppler] - velocity vs angle
            - 'cartesian_xy': [y, x] - top-down bird's-eye view
            - 'cartesian_xz': [z, x] - side elevation view
            Returns None if processing fails
        """
        try:
            # 1. Convert raw int16 data to complex cube
            data = np.array(msg.data, dtype=np.int16)
            cube = self._convert_to_complex_cube(
                data, msg.num_chirps, msg.num_rx, msg.num_samples,
                msg.is_complex, msg.interleaved
            )
            # cube shape: [num_chirps, num_rx, num_samples]
            
            # 2. Reshape to TDM structure: [loops, tx, rx, samples]
            # For TDM-MIMO, chirps are interleaved by TX: TX1, TX2, TX3, TX1, TX2, TX3, ...
            # So we reshape to [loops, tx, rx, samples] where loops = num_chirps / num_tx
            if msg.tdm_mimo:
                n_loops = msg.num_chirps // self.n_tx
                if msg.num_chirps % self.n_tx != 0:
                    raise ValueError(f"num_chirps ({msg.num_chirps}) must be divisible by num_tx ({self.n_tx})")
                
                # Reshape: [num_chirps, num_rx, num_samples] -> [loops, tx, rx, samples]
                # The chirps are in order: TX0, TX1, TX2, TX0, TX1, TX2, ...
                tdm_cube = cube.reshape(n_loops, self.n_tx, self.n_rx, self.n_samples)
            else:
                # Non-TDM-MIMO: keep as [chirps, rx, samples]
                # Add dummy TX dimension for consistency: [chirps, 1, rx, samples]
                tdm_cube = cube[:, None, :, :]
                n_loops = msg.num_chirps
            
            # 3. Add batch dimension: [1, loops, tx, rx, samples]
            # Process each frame independently (no buffering/concatenation)
            iq = tdm_cube[None, ...]
            # iq shape: [1, loops, tx, rx, samples]
            
            # Process single frame independently
            processing_cube = iq
            batch, doppler, tx, rx, samples = processing_cube.shape
            
            # 4. Apply Range-Doppler FFT (along doppler=axis 1, range=axis 4)
            # Shift only doppler axis to center zero-velocity, keep range 0 at index 0
            # This preserves the physical meaning of range bins (0 = closest to radar)
            rd = self._fft_with_shift(processing_cube, axes=(1, 4), shift=(1,))
            # rd shape: [batch, doppler, tx, rx, range]
            
            # 5. Construct virtual array from physical antenna data
            # For TDM-MIMO: Maps 3 TX × 4 RX physical antennas to 2 elevation × 8 azimuth virtual array
            # For non-TDM-MIMO: Use simple virtual array (1 elevation, num_rx azimuth)
            if msg.tdm_mimo:
                mimo = self._construct_virtual_array(rd)
            else:
                # Non-TDM-MIMO: simple virtual array (no elevation/azimuth separation)
                # Use first TX antenna data (or all TX if only one)
                batch, doppler, tx, rx, range_bins = rd.shape
                # For non-TDM-MIMO, we typically have only one TX active
                # Create virtual array: [batch, doppler, 1, rx, range]
                mimo = rd[:, :, 0:1, :, :]  # Use first TX only
                # Reshape to match expected shape [batch, doppler, elevation, azimuth, range]
                # Add elevation dimension: [batch, doppler, 1, rx, range]
                mimo = mimo.reshape(batch, doppler, 1, rx, range_bins)
            # mimo shape: [batch, doppler, elevation, azimuth, range]
            
            # 6. Apply Elevation-Azimuth FFT (along elevation=axis 2, azimuth=axis 3)
            # Zero-pad azimuth dimension to 128 for higher angular resolution
            # Shift both elevation and azimuth axes to center zero-angle
            azimuth_size = 128  # Target azimuth resolution
            elevation_size = mimo.shape[2]  # Keep original elevation size
            drae = self._fft_with_shift(mimo, axes=(2, 3), size=(elevation_size, azimuth_size), shift=(2, 3))
            # drae shape: [batch, doppler, elevation, azimuth=128, range]
            
            # 7. Extract Range-Doppler and Range-Azimuth heatmaps
            # Take absolute value to get magnitude spectrum
            dear = np.abs(drae)
            
            # 8. Extract Range-Doppler heatmap: average over batch, elevation, azimuth
            # Result: [doppler, range] showing velocity vs distance
            rd_2d = np.mean(dear, axis=(0, 2, 3))
            # Swap axes to [range, doppler] for visualization (range on y-axis, doppler on x-axis)
            rd_2d = np.swapaxes(rd_2d, 0, 1)
            
            # 9. Extract Range-Azimuth heatmap: average over batch, doppler, elevation
            # Result: [azimuth, range] showing angle vs distance
            ra_2d = np.mean(dear, axis=(0, 1, 2))
            # Swap axes to [range, azimuth] for visualization (range on y-axis, azimuth on x-axis)
            ra_2d = np.swapaxes(ra_2d, 0, 1)
            
            # 10. Extract Range-Elevation heatmap: average over batch, doppler, azimuth
            # Result: [elevation, range] showing elevation angle vs distance
            re_2d = np.mean(dear, axis=(0, 1, 3))
            # Swap axes to [range, elevation] for visualization (range on y-axis, elevation on x-axis)
            re_2d = np.swapaxes(re_2d, 0, 1)
            
            # 11. Extract Doppler-Azimuth heatmap: average over batch, elevation, range
            # Result: [doppler, azimuth] showing velocity vs angle
            da_2d = np.mean(dear, axis=(0, 2, 4))
            # Swap axes to [azimuth, doppler] for visualization (azimuth on y-axis, doppler on x-axis)
            da_2d = np.swapaxes(da_2d, 0, 1)
            
            # 12. Convert to Cartesian coordinates
            range_max = self.params.get('range_max', 10.0)  # Default 10m if not set
            cartesian_xy = self._convert_to_cartesian_xy(ra_2d, range_max, azimuth_size)
            cartesian_xz = self._convert_to_cartesian_xz(re_2d, range_max, elevation_size)
            
            return {
                'range_doppler': rd_2d,
                'range_azimuth': ra_2d,
                'range_elevation': re_2d,
                'doppler_azimuth': da_2d,
                'cartesian_xy': cartesian_xy,
                'cartesian_xz': cartesian_xz
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None

