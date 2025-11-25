    metadata = ISPMetadata(
        # Camera settings
        iso=800,
        sensor_gain=2.0,
        lux_index=100.0,
        shutter_time=0.033,  # 1/30 second exposure
        
        # Exposure values (your actual values)
        i32ev=0,
        i32capture_ev=2170,
        ev_comp=0.0,
        
        # RAW data format
        bpp=10,  # 10-bit per pixel
        packed=True,  # Data is packed in 16-bit
        
        # Tone mapping curve (S-curve for contrast)
        gtm_curve=np.array([
            0.00, 0.05, 0.15, 0.30, 0.50, 
            0.70, 0.85, 0.95, 1.00
        ]),
        
        # Color correction matrix (example: daylight)
        ccm=np.array([
            [ 1.5, -0.3, -0.2],
            [-0.2,  1.4, -0.2],
            [-0.1, -0.4,  1.5]
        ]),
        
        # Sensor parameters - per channel black levels [R, Gr, Gb, B]
        black_level=np.array([64, 64, 64, 64]),
        white_level=1023,
        
        # White balance gains [R, Gr, Gb, B]
        wb_gains=np.array([2.0, 1.0, 1.0, 1.5]),
        
        # Bayer pattern
        bayer_pattern='RGGB',
        
        # Gamma
        gamma=2.2
    )"""
Complete ISP Pipeline for RAW Image Processing
Step-by-step implementation with detailed comments
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ============================================================================
# STEP 1: Define Metadata Structure
# ============================================================================

@dataclass
class ISPMetadata:
    """
    Container for all ISP processing metadata
    
    Parameters:
        iso: Camera ISO setting (e.g., 100, 400, 800, 1600)
        sensor_gain: Analog/digital sensor gain multiplier
        lux_index: Light level measurement (0-1000+)
        gtm_curve: Global tone mapping curve (1D array, normalized 0-1)
        ccm: Color correction matrix (3x3 array)
        black_level: Sensor black level offset per channel [R, Gr, Gb, B] or single value
        white_level: Sensor saturation level (typically 1023 for 10-bit)
        wb_gains: White balance gains [R, Gr, Gb, B] as array or tuple
        bayer_pattern: Bayer pattern type ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        gamma: Gamma correction value (typically 2.2)
        i32ev: Internal EV (Exposure Value) - typically 0 for normal exposure
        i32capture_ev: Capture EV in scaled units (e.g., 2170 = actual EV scaled by factor)
        ev_comp: Exposure compensation in stops (0 = no compensation)
        bpp: Bits per pixel (actual bit depth, e.g., 10, 12, 14)
        packed: Whether the RAW data is packed (True) or unpacked (False)
        shutter_time: Shutter/exposure time in seconds (e.g., 0.033 for 1/30s)
    """
    iso: int
    sensor_gain: float
    lux_index: float
    gtm_curve: np.ndarray
    ccm: np.ndarray
    black_level: np.ndarray  # Array of 4 values [R, Gr, Gb, B]
    white_level: int = 1023
    wb_gains: np.ndarray  # Array of 4 values [R, Gr, Gb, B]
    bayer_pattern: str = 'RGGB'
    gamma: float = 2.2
    i32ev: int = 0
    i32capture_ev: int = 0
    ev_comp: float = 0.0
    bpp: int = 10  # Bits per pixel
    packed: bool = True  # Whether data is packed
    shutter_time: float = 0.033  # Exposure time in seconds
    
    def __post_init__(self):
        """Convert inputs to numpy arrays if needed and calculate derived values"""
        if not isinstance(self.black_level, np.ndarray):
            if isinstance(self.black_level, (list, tuple)):
                self.black_level = np.array(self.black_level)
            else:
                # Single value - replicate for all channels
                self.black_level = np.array([self.black_level] * 4)
        
        if not isinstance(self.wb_gains, np.ndarray):
            self.wb_gains = np.array(self.wb_gains)
        
        # Validate shapes
        assert self.black_level.shape == (4,), "black_level must have 4 values [R, Gr, Gb, B]"
        assert self.wb_gains.shape == (4,), "wb_gains must have 4 values [R, Gr, Gb, B]"
        
        # Validate white level matches bpp
        expected_white_level = (2 ** self.bpp) - 1
        if self.white_level == 1023 and self.bpp != 10:
            print(f"Warning: white_level={self.white_level} but bpp={self.bpp}, " 
                  f"expected {expected_white_level}. Adjusting white_level.")
            self.white_level = expected_white_level
    
    def get_exposure_scale(self) -> float:
        """
        Calculate exposure scaling factor from EV values
        
        i32capture_ev is often scaled (e.g., by 1000 or 256)
        Common scaling: actual_EV = i32capture_ev / 256
        
        Returns:
            Exposure scale factor to apply during processing
        """
        if self.i32capture_ev == 0:
            return 1.0
        
        # Typical scaling factor for i32capture_ev
        actual_ev = self.i32capture_ev / 256.0
        
        # Apply EV compensation
        total_ev = actual_ev + self.ev_comp
        
        # Convert EV to linear scale: scale = 2^EV
        exposure_scale = 2.0 ** total_ev
        
        return exposure_scale
    
    def get_exposure_info(self) -> dict:
        """
        Get comprehensive exposure information
        
        Returns:
            Dictionary with exposure parameters and calculations
        """
        info = {
            'iso': self.iso,
            'shutter_time': self.shutter_time,
            'shutter_fraction': f"1/{int(1/self.shutter_time)}" if self.shutter_time > 0 else "N/A",
            'sensor_gain': self.sensor_gain,
            'lux_index': self.lux_index,
            'i32ev': self.i32ev,
            'i32capture_ev': self.i32capture_ev,
            'ev_comp': self.ev_comp
        }
        
        if self.i32capture_ev != 0:
            info['actual_ev'] = self.i32capture_ev / 256.0
            info['exposure_scale'] = self.get_exposure_scale()
        
        # Calculate total exposure (ISO × shutter time)
        info['total_exposure'] = self.iso * self.shutter_time
        
        return info


# ============================================================================
# STEP 2: ISP Pipeline Class
# ============================================================================

class ISPPipeline:
    """
    Complete Image Signal Processing Pipeline
    Processes RAW Bayer images to final RGB output
    """
    
    def __init__(self, metadata: ISPMetadata, debug: bool = False):
        """
        Initialize ISP pipeline with metadata
        
        Args:
            metadata: ISPMetadata object with all processing parameters
            debug: If True, saves intermediate results for debugging
        """
        self.meta = metadata
        self.debug = debug
        self.intermediate_results = {}
    
    # ========================================================================
    # STEP 0: Unpack RAW Data (if packed)
    # ========================================================================
    
    def unpack_raw(self, packed_raw: np.ndarray) -> np.ndarray:
        """
        Unpack 10-bit RAW data stored in 16-bit containers
        
        10-bit data is often packed to save space:
        - 4 pixels (40 bits) packed into 5 bytes
        - Or stored in 16-bit with 6 unused bits (LSB or MSB)
        
        Common packing schemes:
        1. MIPI RAW10: 4 pixels in 5 bytes
        2. Plain 16-bit: 10-bit values stored in uint16 (LSB aligned)
        
        Args:
            packed_raw: Packed RAW data
            
        Returns:
            Unpacked RAW data with actual 10-bit values
        """
        print("Step 0: Unpacking RAW Data")
        print(f"  Input shape: {packed_raw.shape}")
        print(f"  Input dtype: {packed_raw.dtype}")
        print(f"  BPP: {self.meta.bpp}")
        print(f"  Packed: {self.meta.packed}")
        
        if not self.meta.packed:
            print(f"  Data already unpacked, skipping")
            print(f"  ✓ Complete\n")
            return packed_raw
        
        # If data is in uint16 with 10-bit values
        if packed_raw.dtype == np.uint16:
            # Check if values are already in correct range
            max_val = packed_raw.max()
            expected_max = (2 ** self.meta.bpp) - 1
            
            if max_val <= expected_max:
                print(f"  Data appears already unpacked (max={max_val} <= {expected_max})")
                print(f"  ✓ Complete\n")
                return packed_raw
            
            # If values are bit-shifted, align them
            if max_val > expected_max:
                # Find how many bits to shift
                shift = 0
                temp_max = max_val
                while temp_max > expected_max:
                    temp_max >>= 1
                    shift += 1
                
                if shift > 0:
                    print(f"  Bit-shifting right by {shift} bits")
                    unpacked = packed_raw >> shift
                    unpacked = unpacked.astype(np.uint16)
                    print(f"  Output range: [{unpacked.min()}, {unpacked.max()}]")
                    print(f"  ✓ Complete\n")
                    return unpacked
        
        # For MIPI RAW10 packed format (4 pixels in 5 bytes)
        elif packed_raw.dtype == np.uint8:
            print(f"  Unpacking MIPI RAW10 format")
            return self._unpack_mipi_raw10(packed_raw)
        
        print(f"  Output range: [{packed_raw.min()}, {packed_raw.max()}]")
        print(f"  ✓ Complete\n")
        return packed_raw
    
    def _unpack_mipi_raw10(self, packed: np.ndarray) -> np.ndarray:
        """
        Unpack MIPI RAW10 format
        
        MIPI RAW10 packing:
        5 bytes contain 4 pixels (10 bits each)
        Byte layout: [P0[9:2]][P1[9:2]][P2[9:2]][P3[9:2]][P3[1:0]P2[1:0]P1[1:0]P0[1:0]]
        
        Args:
            packed: Packed uint8 array
            
        Returns:
            Unpacked uint16 array with 10-bit values
        """
        # Calculate dimensions
        total_bytes = packed.size
        num_groups = total_bytes // 5
        height = packed.shape[0]
        width = (num_groups * 4) // height
        
        # Reshape to groups of 5 bytes
        packed_groups = packed.reshape(-1, 5)
        
        # Extract 4 pixels from each group
        pixel0 = ((packed_groups[:, 0].astype(np.uint16) << 2) | 
                  ((packed_groups[:, 4] & 0x03) >> 0))
        pixel1 = ((packed_groups[:, 1].astype(np.uint16) << 2) | 
                  ((packed_groups[:, 4] & 0x0C) >> 2))
        pixel2 = ((packed_groups[:, 2].astype(np.uint16) << 2) | 
                  ((packed_groups[:, 4] & 0x30) >> 4))
        pixel3 = ((packed_groups[:, 3].astype(np.uint16) << 2) | 
                  ((packed_groups[:, 4] & 0xC0) >> 6))
        
        # Interleave pixels
        unpacked = np.empty(num_groups * 4, dtype=np.uint16)
        unpacked[0::4] = pixel0
        unpacked[1::4] = pixel1
        unpacked[2::4] = pixel2
        unpacked[3::4] = pixel3
        
        # Reshape to image dimensions
        unpacked = unpacked[:height * width].reshape(height, width)
        
        return unpacked
        
    # ========================================================================
    # STEP 3: Main Processing Pipeline
    # ========================================================================
    
    def process(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Execute complete ISP pipeline
        
        Pipeline order:
        0. Unpack RAW data (if packed)
        1. Black level correction
        2. Lens shading correction
        3. White balance
        4. Demosaicing
        5. Denoising
        6. Color correction
        7. Global tone mapping
        8. Gamma correction
        9. Sharpening
        10. Final output conversion
        
        Args:
            raw_image: RAW Bayer pattern image (H, W), can be packed or unpacked
            
        Returns:
            Processed RGB image (H, W, 3) in uint8 format
        """
        print("="*60)
        print("Starting ISP Pipeline Processing")
        print("="*60)
        print(f"Input shape: {raw_image.shape}")
        print(f"Input dtype: {raw_image.dtype}")
        print(f"Input range: [{raw_image.min()}, {raw_image.max()}]")
        
        # Display comprehensive exposure information
        print(f"\nExposure Information:")
        exp_info = self.meta.get_exposure_info()
        print(f"  ISO: {exp_info['iso']}")
        print(f"  Shutter time: {exp_info['shutter_time']:.6f}s ({exp_info['shutter_fraction']})")
        print(f"  Sensor gain: {exp_info['sensor_gain']}")
        print(f"  Lux index: {exp_info['lux_index']}")
        print(f"  Total exposure: {exp_info['total_exposure']:.2f}")
        print(f"  i32ev: {exp_info['i32ev']}")
        print(f"  i32capture_ev: {exp_info['i32capture_ev']}")
        print(f"  ev_comp: {exp_info['ev_comp']}")
        if 'actual_ev' in exp_info:
            print(f"  Actual EV: {exp_info['actual_ev']:.2f}")
            print(f"  Exposure scale: {exp_info['exposure_scale']:.4f}")
        print()
        
        # Store original for comparison
        if self.debug:
            self.intermediate_results['00_input'] = raw_image.copy()
        
        # Step 0: Unpack RAW data if needed
        img = self.unpack_raw(raw_image)
        
        # Step 1: Black Level Correction
        img = self.step1_black_level_correction(img)
        
        # Step 2: Lens Shading Correction
        img = self.step2_lens_shading_correction(img)
        
        # Step 3: White Balance
        img = self.step3_white_balance(img)
        
        # Step 4: Demosaicing
        rgb = self.step4_demosaic(img)
        
        # Step 5: Denoising
        rgb = self.step5_denoise(rgb)
        
        # Step 6: Color Correction
        rgb = self.step6_color_correction(rgb)
        
        # Step 7: Global Tone Mapping
        rgb = self.step7_global_tone_mapping(rgb)
        
        # Step 8: Gamma Correction
        rgb = self.step8_gamma_correction(rgb)
        
        # Step 9: Sharpening
        rgb = self.step9_sharpen(rgb)
        
        # Step 10: Final Output Conversion
        output = self.step10_output_conversion(rgb)
        
        print("="*60)
        print("Pipeline Complete!")
        print("="*60)
        
        return output
    
    # ========================================================================
    # STEP 1: Black Level Correction
    # ========================================================================
    
    def step1_black_level_correction(self, raw: np.ndarray) -> np.ndarray:
        """
        Remove black level offset and normalize to [0, 1]
        
        Black level is the sensor output when no light hits it.
        This needs to be subtracted to get true zero.
        Each Bayer channel can have different black levels.
        
        Also applies exposure scaling if i32capture_ev is provided.
        
        Formula: (raw - black_level[channel]) / (white_level - black_level[channel])
                 Then optionally multiply by exposure scale
        
        Args:
            raw: Input RAW image
            
        Returns:
            Normalized image in range [0, 1]
        """
        print("Step 1: Black Level Correction & Exposure Scaling")
        print(f"  Black levels [R, Gr, Gb, B]: {self.meta.black_level}")
        print(f"  White level: {self.meta.white_level}")
        
        corrected = raw.astype(np.float32)
        h, w = raw.shape
        
        # Map Bayer patterns to pixel positions and black level indices
        pattern_map = {
            'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)],  # R, Gr, Gb, B
            'BGGR': [(0, 0, 3), (0, 1, 2), (1, 0, 1), (1, 1, 0)],  # B, Gb, Gr, R
            'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 3), (1, 1, 2)],  # Gr, R, B, Gb
            'GBRG': [(0, 0, 2), (0, 1, 3), (1, 0, 0), (1, 1, 1)]   # Gb, B, R, Gr
        }
        
        # Subtract black level from each Bayer channel
        positions = pattern_map[self.meta.bayer_pattern]
        for y_offset, x_offset, bl_idx in positions:
            corrected[y_offset::2, x_offset::2] -= self.meta.black_level[bl_idx]
        
        # Clip negative values
        corrected = np.clip(corrected, 0, None)
        
        # Normalize to [0, 1] using average black level
        avg_black_level = np.mean(self.meta.black_level)
        max_value = self.meta.white_level - avg_black_level
        corrected = corrected / max_value
        
        # Apply exposure scaling if capture EV is provided
        if self.meta.i32capture_ev != 0:
            exposure_scale = self.meta.get_exposure_scale()
            print(f"  Applying exposure scale: {exposure_scale:.4f}")
            corrected = corrected * exposure_scale
            corrected = np.clip(corrected, 0, 1)
        
        print(f"  Output range: [{corrected.min():.4f}, {corrected.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['01_black_level'] = corrected.copy()
        
        return corrected
    
    # ========================================================================
    # STEP 2: Lens Shading Correction
    # ========================================================================
    
    def step2_lens_shading_correction(self, raw: np.ndarray) -> np.ndarray:
        """
        Correct for vignetting (lens shading)
        
        Vignetting causes corners to be darker than the center.
        We apply a radial gain map to compensate.
        
        Args:
            raw: Input normalized RAW image
            
        Returns:
            Corrected image
        """
        print("Step 2: Lens Shading Correction")
        
        h, w = raw.shape
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance
        
        # Apply polynomial vignetting model
        # Gain increases toward corners
        gain_map = 1.0 + 0.3 * normalized_distance**2
        
        corrected = raw * gain_map
        corrected = np.clip(corrected, 0, 1)
        
        print(f"  Gain range: [{gain_map.min():.4f}, {gain_map.max():.4f}]")
        print(f"  Output range: [{corrected.min():.4f}, {corrected.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['02_lens_shading'] = corrected.copy()
            self.intermediate_results['02_gain_map'] = gain_map
        
        return corrected
    
    # ========================================================================
    # STEP 3: White Balance
    # ========================================================================
    
    def step3_white_balance(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply white balance gains to Bayer pattern
        
        Different color channels have different sensitivities.
        White balance gains compensate for this and color temperature.
        
        Bayer pattern (RGGB example):
        R  Gr
        Gb B
        
        Args:
            raw: Input normalized RAW image
            
        Returns:
            White balanced image
        """
        print("Step 3: White Balance")
        print(f"  Bayer pattern: {self.meta.bayer_pattern}")
        print(f"  WB gains (R, Gr, Gb, B): {self.meta.wb_gains}")
        
        wb_image = raw.copy()
        h, w = raw.shape
        
        # Map Bayer patterns to pixel positions and gain indices
        pattern_map = {
            'RGGB': [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)],  # R, Gr, Gb, B
            'BGGR': [(0, 0, 3), (0, 1, 2), (1, 0, 1), (1, 1, 0)],  # B, Gb, Gr, R
            'GRBG': [(0, 0, 1), (0, 1, 0), (1, 0, 3), (1, 1, 2)],  # Gr, R, B, Gb
            'GBRG': [(0, 0, 2), (0, 1, 3), (1, 0, 0), (1, 1, 1)]   # Gb, B, R, Gr
        }
        
        # Apply gains to each Bayer channel
        positions = pattern_map[self.meta.bayer_pattern]
        for y_offset, x_offset, gain_idx in positions:
            wb_image[y_offset::2, x_offset::2] *= self.meta.wb_gains[gain_idx]
        
        wb_image = np.clip(wb_image, 0, 1)
        
        print(f"  Output range: [{wb_image.min():.4f}, {wb_image.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['03_white_balance'] = wb_image.copy()
        
        return wb_image
    
    # ========================================================================
    # STEP 4: Demosaicing
    # ========================================================================
    
    def step4_demosaic(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert Bayer pattern to full RGB image
        
        Demosaicing interpolates missing color values at each pixel.
        Each pixel in Bayer has only one color (R, G, or B).
        
        Methods:
        - Bilinear interpolation (fast, good quality)
        - Edge-aware methods (better, slower)
        
        Args:
            raw: White-balanced Bayer image
            
        Returns:
            Full RGB image (H, W, 3)
        """
        print("Step 4: Demosaicing (Bayer to RGB)")
        
        # Map Bayer pattern to OpenCV codes
        bayer_codes = {
            'RGGB': cv2.COLOR_BAYER_BG2RGB,
            'BGGR': cv2.COLOR_BAYER_RG2RGB,
            'GRBG': cv2.COLOR_BAYER_GB2RGB,
            'GBRG': cv2.COLOR_BAYER_GR2RGB
        }
        
        # Convert to 16-bit for OpenCV
        raw_16 = (raw * 65535).astype(np.uint16)
        
        # Demosaic using OpenCV
        rgb_16 = cv2.cvtColor(raw_16, bayer_codes[self.meta.bayer_pattern])
        
        # Convert back to float [0, 1]
        rgb = rgb_16.astype(np.float32) / 65535.0
        
        print(f"  Input shape: {raw.shape}")
        print(f"  Output shape: {rgb.shape}")
        print(f"  Output range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['04_demosaic'] = rgb.copy()
        
        return rgb
    
    # ========================================================================
    # STEP 5: Denoising
    # ========================================================================
    
    def step5_denoise(self, rgb: np.ndarray) -> np.ndarray:
        """
        Reduce noise in the image
        
        Noise increases with ISO/gain and low light conditions.
        We apply adaptive denoising based on ISO and shutter time.
        
        Methods used:
        - Non-local means denoising (preserves edges)
        - Strength adapts to ISO level and exposure time
        
        Args:
            rgb: Input RGB image
            
        Returns:
            Denoised RGB image
        """
        print("Step 5: Denoising")
        print(f"  ISO: {self.meta.iso}")
        print(f"  Sensor gain: {self.meta.sensor_gain}")
        print(f"  Shutter time: {self.meta.shutter_time:.6f}s")
        
        # Calculate adaptive denoising strength
        # Higher ISO = more noise
        iso_factor = self.meta.iso / 400.0
        
        # Shorter exposure = more noise (less light gathered)
        # Normalize to 1/30s = 1.0
        exposure_factor = (1.0 / 30.0) / max(self.meta.shutter_time, 0.001)
        
        # Combine factors
        strength = min(15.0, iso_factor * exposure_factor)
        
        print(f"  ISO factor: {iso_factor:.2f}")
        print(f"  Exposure factor: {exposure_factor:.2f}")
        print(f"  Denoising strength: {strength:.2f}")
        
        if strength > 1.0:
            # Convert to 8-bit for OpenCV denoising
            rgb_8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            
            # Apply non-local means denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                rgb_8,
                None,
                h=strength,
                hColor=strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            # Convert back to float
            rgb = denoised.astype(np.float32) / 255.0
            print(f"  Denoising applied")
        else:
            print(f"  Skipped (low noise conditions)")
        
        print(f"  Output range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['05_denoise'] = rgb.copy()
        
        return rgb
    
    # ========================================================================
    # STEP 6: Color Correction
    # ========================================================================
    
    def step6_color_correction(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply Color Correction Matrix (CCM)
        
        The CCM transforms sensor RGB to a standard color space.
        It corrects for sensor spectral sensitivities.
        
        Formula: RGB_out = RGB_in × CCM^T
        
        Args:
            rgb: Input RGB image
            
        Returns:
            Color-corrected RGB image
        """
        print("Step 6: Color Correction (CCM)")
        print(f"  CCM shape: {self.meta.ccm.shape}")
        print(f"  CCM:\n{self.meta.ccm}")
        
        h, w, c = rgb.shape
        
        # Reshape to (N, 3) for matrix multiplication
        rgb_flat = rgb.reshape(-1, 3)
        
        # Apply CCM: each pixel multiplied by CCM
        corrected = rgb_flat @ self.meta.ccm.T
        
        # Reshape back to image
        corrected = corrected.reshape(h, w, 3)
        
        print(f"  Input range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  Output range: [{corrected.min():.4f}, {corrected.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['06_color_correction'] = corrected.copy()
        
        return corrected
    
    # ========================================================================
    # STEP 7: Global Tone Mapping
    # ========================================================================
    
    def step7_global_tone_mapping(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply Global Tone Mapping (GTM) curve
        
        GTM compresses high dynamic range to display range.
        Common curves: S-curve, log curve, filmic curve
        
        The curve can be any length - it will be interpolated to match
        the input values. A 514-point curve is very common and provides
        smooth, high-resolution tone mapping.
        
        Args:
            rgb: Input RGB image
            
        Returns:
            Tone-mapped RGB image
        """
        print("Step 7: Global Tone Mapping")
        
        if self.meta.gtm_curve is None or len(self.meta.gtm_curve) == 0:
            print(f"  No GTM curve provided, skipping")
            print(f"  ✓ Complete\n")
            return rgb
        
        print(f"  GTM curve length: {len(self.meta.gtm_curve)} points")
        print(f"  GTM curve range: [{self.meta.gtm_curve.min():.4f}, {self.meta.gtm_curve.max():.4f}]")
        
        # Create input values for interpolation
        # Map curve indices to [0, 1] range
        x_curve = np.linspace(0, 1, len(self.meta.gtm_curve))
        
        # Apply curve to each channel independently
        mapped = np.zeros_like(rgb)
        for i in range(3):
            # Interpolate: for each pixel value, find corresponding curve output
            mapped[:, :, i] = np.interp(rgb[:, :, i], x_curve, self.meta.gtm_curve)
        
        print(f"  Input range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  Output range: [{mapped.min():.4f}, {mapped.max():.4f}]")
        
        # Show curve characteristics
        curve_type = self._analyze_gtm_curve(self.meta.gtm_curve)
        print(f"  Curve type: {curve_type}")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['07_tone_mapping'] = mapped.copy()
            # Also save the curve for visualization
            self.intermediate_results['07_gtm_curve'] = self.meta.gtm_curve.copy()
        
        return mapped
    
    def _analyze_gtm_curve(self, curve: np.ndarray) -> str:
        """
        Analyze the GTM curve characteristics
        
        Args:
            curve: GTM curve array
            
        Returns:
            String describing the curve type
        """
        # Check if curve is linear
        linear = np.linspace(0, 1, len(curve))
        if np.allclose(curve, linear, atol=0.01):
            return "Linear (no tone mapping)"
        
        # Check for S-curve characteristics
        mid_idx = len(curve) // 2
        quarter_idx = len(curve) // 4
        three_quarter_idx = 3 * len(curve) // 4
        
        # S-curve typically has:
        # - Gentle slope at start (shadows)
        # - Steep slope in middle (midtones)
        # - Gentle slope at end (highlights)
        
        slope_start = curve[quarter_idx] - curve[0]
        slope_mid = curve[three_quarter_idx] - curve[quarter_idx]
        slope_end = curve[-1] - curve[three_quarter_idx]
        
        if slope_mid > slope_start and slope_mid > slope_end:
            if curve[mid_idx] > linear[mid_idx]:
                return "S-curve (contrast boost, brightening)"
            else:
                return "S-curve (contrast boost)"
        elif slope_start < slope_mid < slope_end:
            return "Expansive curve (highlight protection)"
        elif slope_start > slope_mid > slope_end:
            return "Compressive curve (shadow boost)"
        else:
            return "Custom curve"
    
    # ========================================================================
    # STEP 8: Gamma Correction
    # ========================================================================
    
    def step8_gamma_correction(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction
        
        Gamma correction compensates for display non-linearity.
        Standard gamma is 2.2 for most displays.
        
        Formula: output = input^(1/gamma)
        
        Args:
            rgb: Input RGB image
            
        Returns:
            Gamma-corrected RGB image
        """
        print("Step 8: Gamma Correction")
        print(f"  Gamma: {self.meta.gamma}")
        
        # Clip to valid range before gamma
        rgb_clipped = np.clip(rgb, 0, 1)
        
        # Apply gamma correction
        corrected = np.power(rgb_clipped, 1.0 / self.meta.gamma)
        
        print(f"  Input range: [{rgb_clipped.min():.4f}, {rgb_clipped.max():.4f}]")
        print(f"  Output range: [{corrected.min():.4f}, {corrected.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['08_gamma'] = corrected.copy()
        
        return corrected
    
    # ========================================================================
    # STEP 9: Sharpening
    # ========================================================================
    
    def step9_sharpen(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply image sharpening
        
        Sharpening enhances edges and fine details.
        We use adaptive sharpening based on lighting conditions.
        
        Method: Unsharp masking via convolution kernel
        
        Args:
            rgb: Input RGB image
            
        Returns:
            Sharpened RGB image
        """
        print("Step 9: Sharpening")
        print(f"  Lux index: {self.meta.lux_index}")
        
        # Adaptive sharpening: less in low light (more noise)
        strength = np.clip(1.0 - (self.meta.lux_index / 1000.0), 0.3, 1.0)
        print(f"  Sharpening strength: {strength:.2f}")
        
        # Sharpening kernel (Laplacian-based)
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1 + 4*strength, -strength],
            [0, -strength, 0]
        ])
        
        # Apply to each channel
        sharpened = np.zeros_like(rgb)
        for i in range(3):
            sharpened[:, :, i] = cv2.filter2D(rgb[:, :, i], -1, kernel)
        
        print(f"  Output range: [{sharpened.min():.4f}, {sharpened.max():.4f}]")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['09_sharpen'] = sharpened.copy()
        
        return sharpened
    
    # ========================================================================
    # STEP 10: Final Output Conversion
    # ========================================================================
    
    def step10_output_conversion(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert to final 8-bit output format
        
        Clips values to [0, 1] and converts to uint8 [0, 255]
        
        Args:
            rgb: Input RGB image in float [0, 1]
            
        Returns:
            Final RGB image in uint8 [0, 255]
        """
        print("Step 10: Output Conversion")
        
        # Clip to valid range
        rgb_clipped = np.clip(rgb, 0, 1)
        
        # Convert to 8-bit
        output = (rgb_clipped * 255).astype(np.uint8)
        
        print(f"  Input range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        print(f"  Output range: [{output.min()}, {output.max()}]")
        print(f"  Output dtype: {output.dtype}")
        print(f"  ✓ Complete\n")
        
        if self.debug:
            self.intermediate_results['10_output'] = output.copy()
        
        return output
    
    # ========================================================================
    # Utility: Save Debug Results
    # ========================================================================
    
    def save_debug_results(self, output_dir: str = 'isp_debug'):
        """
        Save all intermediate results for debugging
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, img in self.intermediate_results.items():
            # Handle GTM curve separately
            if name == '07_gtm_curve':
                self._save_gtm_curve_plot(img, output_dir)
                continue
            
            filename = os.path.join(output_dir, f'{name}.png')
            
            if img.ndim == 2:  # Grayscale
                # Normalize for visualization
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                cv2.imwrite(filename, img_norm)
            elif img.ndim == 3:  # RGB
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img_8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img_8 = img
                cv2.imwrite(filename, cv2.cvtColor(img_8, cv2.COLOR_RGB2BGR))
        
        print(f"\nDebug results saved to: {output_dir}/")
    
    def _save_gtm_curve_plot(self, curve: np.ndarray, output_dir: str):
        """
        Save a plot of the GTM curve
        
        Args:
            curve: GTM curve array
            output_dir: Directory to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            # Plot the curve
            x = np.linspace(0, 1, len(curve))
            plt.plot(x, curve, 'b-', linewidth=2, label='GTM Curve')
            
            # Plot reference linear curve
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5, label='Linear (no mapping)')
            
            plt.xlabel('Input Value', fontsize=12)
            plt.ylabel('Output Value', fontsize=12)
            plt.title(f'Global Tone Mapping Curve ({len(curve)} points)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # Save
            filename = os.path.join(output_dir, '07_gtm_curve_plot.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  GTM curve plot saved: {filename}")
        except Exception as e:
            print(f"  Could not save GTM curve plot: {e}")


# ============================================================================
# STEP 11: Helper Functions for Loading RAW Files
# ============================================================================

def load_raw_file(file_path: str, height: int, width: int, 
                  bpp: int = 10, dtype: str = 'uint16') -> np.ndarray:
    """
    Load RAW image file from disk
    
    RAW files are typically stored as binary data without headers.
    You need to know the dimensions and bit depth to read them correctly.
    
    Args:
        file_path: Path to the RAW file
        height: Image height in pixels
        width: Image width in pixels
        bpp: Bits per pixel (8, 10, 12, 14, 16)
        dtype: Data type ('uint8', 'uint16')
        
    Returns:
        RAW image as numpy array
        
    Example:
        raw = load_raw_file('image.raw', 1080, 1920, bpp=10, dtype='uint16')
    """
    print(f"Loading RAW file: {file_path}")
    print(f"  Dimensions: {height}x{width}")
    print(f"  BPP: {bpp}")
    print(f"  Data type: {dtype}")
    
    # Map dtype string to numpy dtype
    dtype_map = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'int16': np.int16,
        'uint32': np.uint32,
    }
    
    np_dtype = dtype_map.get(dtype, np.uint16)
    
    try:
        # Read binary file
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np_dtype)
        
        print(f"  Read {len(raw_data)} values")
        print(f"  Expected: {height * width} pixels")
        
        # Reshape to image dimensions
        if len(raw_data) >= height * width:
            raw_image = raw_data[:height * width].reshape(height, width)
            print(f"  Successfully reshaped to ({height}, {width})")
            print(f"  Value range: [{raw_image.min()}, {raw_image.max()}]")
            return raw_image
        else:
            raise ValueError(f"File contains {len(raw_data)} values but expected {height * width}")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"RAW file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading RAW file: {e}")


def load_raw_mipi10(file_path: str, height: int, width: int) -> np.ndarray:
    """
    Load MIPI RAW10 packed format
    
    MIPI RAW10 stores 4 pixels (10-bit each) in 5 bytes.
    Total file size = (height * width * 10) / 8 bytes
    
    Args:
        file_path: Path to the MIPI RAW10 file
        height: Image height in pixels
        width: Image width in pixels
        
    Returns:
        Unpacked RAW image as uint16 array
    """
    print(f"Loading MIPI RAW10 file: {file_path}")
    print(f"  Dimensions: {height}x{width}")
    
    # Calculate expected file size
    num_pixels = height * width
    num_bytes = (num_pixels * 10) // 8
    
    if num_pixels % 4 != 0:
        # MIPI RAW10 requires multiple of 4 pixels
        raise ValueError("Width * Height must be divisible by 4 for MIPI RAW10")
    
    try:
        # Read binary file
        with open(file_path, 'rb') as f:
            packed_data = np.fromfile(f, dtype=np.uint8, count=num_bytes)
        
        print(f"  Read {len(packed_data)} bytes")
        print(f"  Expected: {num_bytes} bytes")
        
        # Unpack MIPI RAW10
        unpacked = _unpack_mipi_raw10_buffer(packed_data, height, width)
        
        print(f"  Successfully unpacked to ({height}, {width})")
        print(f"  Value range: [{unpacked.min()}, {unpacked.max()}]")
        
        return unpacked
        
    except FileNotFoundError:
        raise FileNotFoundError(f"RAW file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading MIPI RAW10 file: {e}")


def _unpack_mipi_raw10_buffer(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Unpack MIPI RAW10 format from byte buffer
    
    Args:
        packed: Packed uint8 array
        height: Image height
        width: Image width
        
    Returns:
        Unpacked uint16 array with 10-bit values
    """
    num_pixels = height * width
    num_groups = num_pixels // 4
    
    # Reshape to groups of 5 bytes
    packed_groups = packed[:num_groups * 5].reshape(-1, 5)
    
    # Extract 4 pixels from each group
    pixel0 = ((packed_groups[:, 0].astype(np.uint16) << 2) | 
              ((packed_groups[:, 4] & 0x03) >> 0))
    pixel1 = ((packed_groups[:, 1].astype(np.uint16) << 2) | 
              ((packed_groups[:, 4] & 0x0C) >> 2))
    pixel2 = ((packed_groups[:, 2].astype(np.uint16) << 2) | 
              ((packed_groups[:, 4] & 0x30) >> 4))
    pixel3 = ((packed_groups[:, 3].astype(np.uint16) << 2) | 
              ((packed_groups[:, 4] & 0xC0) >> 6))
    
    # Interleave pixels
    unpacked = np.empty(num_pixels, dtype=np.uint16)
    unpacked[0::4] = pixel0
    unpacked[1::4] = pixel1
    unpacked[2::4] = pixel2
    unpacked[3::4] = pixel3
    
    # Reshape to image dimensions
    return unpacked.reshape(height, width)


def save_raw_file(raw_image: np.ndarray, file_path: str):
    """
    Save RAW image to binary file
    
    Args:
        raw_image: RAW image array
        file_path: Output file path
    """
    print(f"Saving RAW file: {file_path}")
    print(f"  Shape: {raw_image.shape}")
    print(f"  Dtype: {raw_image.dtype}")
    
    with open(file_path, 'wb') as f:
        raw_image.tofile(f)
    
    file_size = raw_image.nbytes
    print(f"  Saved {file_size} bytes ({file_size / (1024*1024):.2f} MB)")


# ============================================================================
# STEP 12: Complete Example with File Loading
# ============================================================================

def create_test_raw_image(height: int = 512, width: int = 512) -> np.ndarray:
    """
    Create a synthetic RAW test image with Bayer pattern
    
    Args:
        height: Image height (must be even)
        width: Image width (must be even)
        
    Returns:
        Synthetic RAW image in 10-bit format
    """
    # Create gradient pattern
    y, x = np.ogrid[:height, :width]
    
    # Create interesting pattern: gradient + radial
    gradient = x / width
    radial = np.sqrt((x - width/2)**2 + (y - height/2)**2) / (width/2)
    pattern = gradient * 0.5 + radial * 0.5
    
    # Scale to 10-bit range
    raw = (pattern * (1023 - 64) + 64).astype(np.uint16)
    
    # Add some noise
    noise = np.random.normal(0, 10, (height, width))
    raw = np.clip(raw + noise, 64, 1023).astype(np.uint16)
    
    return raw


def main():
    """
    Main function demonstrating complete pipeline usage
    """
    print("\n" + "="*60)
    print("ISP Pipeline - Complete Example")
    print("="*60 + "\n")
    
    # ========================================================================
    # Configure Metadata
    # ========================================================================
    
    print("Configuring ISP Metadata...")
    
    metadata = ISPMetadata(
        # Camera settings
        iso=800,
        sensor_gain=2.0,
        lux_index=100.0,
        
        # Tone mapping curve (S-curve for contrast)
        gtm_curve=np.array([
            0.00, 0.05, 0.15, 0.30, 0.50, 
            0.70, 0.85, 0.95, 1.00
        ]),
        
        # Color correction matrix (example: daylight)
        ccm=np.array([
            [ 1.5, -0.3, -0.2],
            [-0.2,  1.4, -0.2],
            [-0.1, -0.4,  1.5]
        ]),
        
        # Sensor parameters - per channel black levels [R, Gr, Gb, B]
        black_level=np.array([64, 64, 64, 64]),
        white_level=1023,
        
        # White balance gains [R, Gr, Gb, B]
        wb_gains=np.array([2.0, 1.0, 1.0, 1.5]),
        
        # Bayer pattern
        bayer_pattern='RGGB',
        
        # Gamma
        gamma=2.2
    )
    
    print("✓ Metadata configured\n")
    
    # ========================================================================
    # Create or Load RAW Image
    # ========================================================================
    
    print("Creating test RAW image...")
    raw_image = create_test_raw_image(height=512, width=512)
    print(f"✓ RAW image created: {raw_image.shape}, {raw_image.dtype}\n")
    
    # ========================================================================
    # Process Image
    # ========================================================================
    
    # Create pipeline instance with debug mode
    isp = ISPPipeline(metadata, debug=True)
    
    # Process the image
    processed = isp.process(raw_image)
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    print("\nSaving results...")
    
    # Save final output
    cv2.imwrite('output_final.png', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    print("✓ Saved: output_final.png")
    
    # Save all intermediate results
    isp.save_debug_results('isp_debug')
    
    # ========================================================================
    # Display Summary
    # ========================================================================
    
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Input:  {raw_image.shape} {raw_image.dtype}")
    print(f"Output: {processed.shape} {processed.dtype}")
    print(f"Output range: [{processed.min()}, {processed.max()}]")
    print(f"\nAll intermediate results saved to: isp_debug/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
