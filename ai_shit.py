"""
FFT-Based Signal Processing Pipeline with AWGN Channel & Spectral Analysis
- 6-stage cascaded upsampling using 8-point FFT-based digital upconversion
- 14-bit fixed-point quantization (12-bit fractional width)
- 16-point FFT-based upconversion after filtering
- AWGN channel simulation
- Frequency domain spectral analysis
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from input_gen import int_to_bin, float_to_fixed_binary
from fft import bin_to_int
from upconverter_16 import upconverter_16

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================
FS = 40e6  # Sampling frequency: 40 MHz
SIGNAL_FREQ = 20e6  # Signal frequency: 20 MHz
N_FFT_INPUT = 1024  # FFT size for frequency analysis
N_SAMPLES_INPUT = 2000  # Input samples

# Quantization parameters
BIT_WIDTH = 14  # Total bits
FRAC_WIDTH = 12  # Fractional bits
Q_FACTOR = 2 ** FRAC_WIDTH  # Quantization factor

# ============================================================================
# 1. INPUT SIGNAL GENERATION
# ============================================================================

def generate_lut(nsam, bin_freq, n_fft=1024):
    """
    Generate quantized sine and cosine lookup tables.
    
    Parameters:
    -----------
    nsam : int
        Number of samples
    bin_freq : int
        Bin frequency (normalized to FFT size)
    n_fft : int
        FFT size (default: 1024)
    
    Returns:
    --------
    sin_lut, cos_lut : arrays of quantized values
    """
    sin_lut = []
    cos_lut = []
    
    for i in range(1, nsam + 1):
        angle = 2 * np.pi * bin_freq * i / n_fft
        sin_lut.append(np.sin(angle))
        cos_lut.append(np.cos(angle))
    
    return np.array(sin_lut), np.array(cos_lut)


def feeder(xi, xq):
    """
    Convert floating-point I/Q values to 14-bit fixed-point binary representation.
    
    Parameters:
    -----------
    xi, xq : arrays
        Floating-point I and Q components
    
    Returns:
    --------
    xi_bin, xq_bin : arrays of fixed-point binary values
    """
    BIT_WIDTH = 14
    FRAC_WIDTH = 12
    Q_FACTOR = 2 ** FRAC_WIDTH
    
    xi_fixed = []
    xq_fixed = []
    
    for i in range(len(xi)):
        # Convert to fixed-point and quantize
        xi_fixed.append(int(np.round(xi[i])))
        xq_fixed.append(int(np.round(xq[i])))
    
    return np.array(xi_fixed), np.array(xq_fixed)


def generate_input_signal():
    """
    Generate input signal: 2000 samples of 20 MHz sine/cosine pair at 40 MHz sampling.
    
    Returns:
    --------
    xi_bin, xq_bin : 14-bit quantized I/Q components
    xi_float, xq_float : floating-point I/Q for reference
    """
    # Calculate bin frequency
    t = np.arange(N_SAMPLES_INPUT) / FS
    
    # Generate 20 MHz signal
    xi_float = np.sin(2 * np.pi * SIGNAL_FREQ * t)
    xq_float = np.cos(2 * np.pi * SIGNAL_FREQ * t)
    
    # Apply amplitude scaling and quantize
    scale = Q_FACTOR * 0.9  # 0.9 to leave headroom
    xi_float_scaled = xi_float * scale / Q_FACTOR
    xq_float_scaled = xq_float * scale / Q_FACTOR
    
    xi_bin, xq_bin = feeder(xi_float_scaled * Q_FACTOR, xq_float_scaled * Q_FACTOR)
    
    return xi_bin / Q_FACTOR, xq_bin / Q_FACTOR, xi_float, xq_float


# ============================================================================
# 2. ELLIPTIC IIR FILTER DESIGN
# ============================================================================

def design_elliptic_filter(order=6, cutoff_norm=0.5):
    """
    Design 6th-order elliptic low-pass IIR filter.
    
    Parameters:
    -----------
    order : int
        Filter order (default: 6)
    cutoff_norm : float
        Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
    
    Returns:
    --------
    b, a : filter coefficients
    """
    # Design elliptic filter with 0.5 dB ripple in passband, 50 dB stopband attenuation
    b, a = signal.ellip(order, 0.5, 50, cutoff_norm)
    return b, a


# ============================================================================
# 3. DIGITAL UPCONVERTER (8-Point FFT-based)
# ============================================================================

def digital_upconverter_fft8(xi, xq):
    """
    Digital upconverter using 8-point FFT/IFFT.
    Performs frequency domain upconversion via FFT processing.
    
    Parameters:
    -----------
    xi, xq : arrays
        I and Q components
    
    Returns:
    --------
    xi_up, xq_up : upconverted I/Q components
    """
    # Use actual signal length for FFT processing, pad to next power of 2
    n_input = len(xi)
    n_fft = max(8, 2 ** int(np.ceil(np.log2(n_input))))
    
    # Pad to FFT size
    xi_padded = np.zeros(n_fft)
    xq_padded = np.zeros(n_fft)
    
    # Place input
    xi_padded[:n_input] = xi
    xq_padded[:n_input] = xq
    
    # FFT
    Xi = np.fft.fft(xi_padded)
    Xq = np.fft.fft(xq_padded)
    
    # Frequency domain upconversion (circular shift by 1 bin)
    Xi_shifted = np.roll(Xi, 1)
    Xq_shifted = np.roll(Xq, 1)
    
    # IFFT
    xi_up = np.fft.ifft(Xi_shifted).real[:n_input]
    xq_up = np.fft.ifft(Xq_shifted).real[:n_input]
    
    return xi_up, xq_up


def digital_upconverter_16point_hardware(xi, xq, shift=1):
    """
    Digital upconverter using 16-point hardware FFT/IFFT.
    Performs frequency domain upconversion via hardware-accurate FFT processing.
    
    Parameters:
    -----------
    xi, xq : arrays
        I and Q components (floating point values in range -1 to 1)
    shift : int
        Spectral shift amount (default: 1)
    
    Returns:
    --------
    xi_up, xq_up : upconverted I/Q components
    """
    n_input = len(xi)
    
    # Pad to multiple of 16
    pad_length = ((n_input + 15) // 16) * 16
    xi_padded = np.zeros(pad_length)
    xq_padded = np.zeros(pad_length)
    xi_padded[:n_input] = xi
    xq_padded[:n_input] = xq
    
    # Convert to 14-bit fixed-point binary representation
    xi_binary = float_to_fixed_binary(xi_padded, total_bits=14, frac_bits=13)
    xq_binary = float_to_fixed_binary(xq_padded, total_bits=14, frac_bits=13)
    
    # Apply hardware upconverter
    ifft_real, ifft_imag = upconverter_16(xi_binary, xq_binary, shift)
    
    # Convert back to floating point
    scale_factor = 2 ** 13
    xi_up = np.array(ifft_real[:n_input]) / scale_factor
    xq_up = np.array(ifft_imag[:n_input]) / scale_factor
    
    return xi_up, xq_up


# ============================================================================
# 4. SINGLE UPSAMPLING STAGE
# ============================================================================

def upsample_stage(xi, xq, stage_num=1):
    """
    Single upsampling stage: digital upconversion + 2x interpolation + IIR filtering.
    
    Parameters:
    -----------
    xi, xq : arrays
        Input I/Q components
    stage_num : int
        Stage number for tracking
    
    Returns:
    --------
    xi_out, xq_out : Output I/Q components (2x upsampled)
    """
    # Step 1: Digital Upconverter (8-point FFT-based)
    xi_upconv, xq_upconv = digital_upconverter_fft8(xi, xq)
    
    # Step 2: 2x Upsampling with Linear Interpolation
    n_input = len(xi_upconv)
    xi_interp = np.zeros(n_input * 2)
    xq_interp = np.zeros(n_input * 2)
    
    # Zero-stuffing: place samples at even indices
    xi_interp[::2] = xi_upconv
    xq_interp[::2] = xq_upconv
    
    # Linear interpolation between samples
    for i in range(1, len(xi_interp), 2):
        xi_interp[i] = (xi_interp[i-1] + xi_interp[(i+1) % len(xi_interp)]) / 2
        xq_interp[i] = (xq_interp[i-1] + xq_interp[(i+1) % len(xq_interp)]) / 2
    
    # Step 3: Elliptic IIR Filter (6th-order, normalized cutoff at 0.4)
    # Adapt filter order based on signal length to avoid filtfilt error
    filter_order = min(6, max(1, len(xi_interp) // 10))
    b, a = signal.ellip(filter_order, 0.5, 50, 0.4)
    
    # Use lfilter instead of filtfilt for smaller signals to avoid padding issues
    if len(xi_interp) > 2 * len(b):
        xi_filtered = signal.filtfilt(b, a, xi_interp)
        xq_filtered = signal.filtfilt(b, a, xq_interp)
    else:
        xi_filtered = signal.lfilter(b, a, xi_interp)
        xq_filtered = signal.lfilter(b, a, xq_interp)
    
    # Quantize to 14-bit fixed-point
    xi_filtered = np.clip(xi_filtered, -1, 1)
    xq_filtered = np.clip(xq_filtered, -1, 1)
    xi_filtered = np.round(xi_filtered * Q_FACTOR) / Q_FACTOR
    xq_filtered = np.round(xq_filtered * Q_FACTOR) / Q_FACTOR
    
    return xi_filtered, xq_filtered


# ============================================================================
# 5. CASCADED 6-STAGE PIPELINE
# ============================================================================

def cascaded_upsampling_pipeline(xi, xq, num_stages=6):
    """
    Cascade upsampling stages for total 2^num_stages upsampling factor.
    
    Parameters:
    -----------
    xi, xq : arrays
        Input I/Q components
    num_stages : int
        Number of stages (default: 6 for 64x upsampling)
    
    Returns:
    --------
    xi_final, xq_final : Output I/Q components
    stage_outputs : List of (xi, xq) at each stage for analysis
    """
    stage_outputs = [(xi.copy(), xq.copy())]  # Store stage 0 (input)
    
    xi_current = xi.copy()
    xq_current = xq.copy()
    
    for stage in range(1, num_stages + 1):
        print(f"Processing Stage {stage}... Input size: {len(xi_current)}", flush=True)
        xi_current, xq_current = upsample_stage(xi_current, xq_current, stage_num=stage)
        stage_outputs.append((xi_current.copy(), xq_current.copy()))
        print(f"  Output size: {len(xi_current)}", flush=True)
    
    return xi_current, xq_current, stage_outputs


# ============================================================================
# 6. AWGN CHANNEL SIMULATION
# ============================================================================

def add_awgn_noise(xi, xq, snr_db=20):
    """
    Add AWGN noise to the signal with specified SNR.
    
    Parameters:
    -----------
    xi, xq : arrays
        Input I/Q components
    snr_db : float
        Signal-to-Noise Ratio in dB
    
    Returns:
    --------
    xi_noisy, xq_noisy : Noisy I/Q components
    """
    # Calculate signal power
    signal_power_i = np.mean(xi ** 2)
    signal_power_q = np.mean(xq ** 2)
    signal_power = (signal_power_i + signal_power_q) / 2
    
    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate AWGN
    noise_i = np.random.normal(0, np.sqrt(noise_power), len(xi))
    noise_q = np.random.normal(0, np.sqrt(noise_power), len(xq))
    
    # Add noise
    xi_noisy = xi + noise_i
    xq_noisy = xq + noise_q
    
    return xi_noisy, xq_noisy


# ============================================================================
# 7. SPECTRAL ANALYSIS & VISUALIZATION
# ============================================================================

def compute_spectrum(xi, xq, fs_effective):
    """
    Compute frequency domain spectrum of I/Q signal.
    
    Parameters:
    -----------
    xi, xq : arrays
        I/Q components
    fs_effective : float
        Effective sampling frequency at this stage
    
    Returns:
    --------
    freqs : Frequency array
    magnitude_db : Magnitude spectrum in dB
    """
    # Combine I/Q into complex signal
    signal_complex = xi + 1j * xq
    
    # Compute FFT
    fft_size = min(len(signal_complex), 4096)
    fft_result = np.fft.fft(signal_complex, n=fft_size)
    magnitude = np.abs(fft_result) / fft_size
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Frequency array (normalized)
    freqs = np.fft.fftfreq(fft_size, 1/fs_effective)
    
    return freqs, magnitude_db, fft_size


def plot_spectral_analysis(stage_outputs, upconv_outputs, snr_db=20):
    """
    Plot frequency spectra at each stage, after 16-bit upconversion, and after channel simulation.
    
    Parameters:
    -----------
    stage_outputs : list
        List of (xi, xq) tuples at each upsampling stage
    upconv_outputs : list
        List of (xi, xq) tuples after 16-bit upconversion and filtering
    snr_db : float
        SNR level for channel simulation
    """
    num_stages = len(stage_outputs) - 1
    num_upconv_stages = len(upconv_outputs)
    n_plots = len(stage_outputs) + num_upconv_stages + 2  # All stages + upconv stages + AWGN + comparison
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    elif n_plots > 1:
        axes = list(axes)
    
    fs_values = [FS * (2 ** i) for i in range(num_stages + 1)]
    
    # Plot spectra for each upsampling stage
    for stage in range(len(stage_outputs)):
        xi, xq = stage_outputs[stage]
        freqs, magnitude_db, fft_size = compute_spectrum(xi, xq, fs_values[stage])
        
        # Plot one-sided spectrum
        one_sided_idx = freqs >= 0
        axes[stage].plot(freqs[one_sided_idx] / 1e6, magnitude_db[one_sided_idx], 'b-', linewidth=0.8)
        axes[stage].set_title(f'Upsampling Stage {stage}: Input size = {len(xi)} samples, Fs = {fs_values[stage]/1e6:.1f} MHz', fontsize=11, fontweight='bold')
        axes[stage].set_ylabel('Magnitude (dB)', fontsize=10)
        axes[stage].grid(True, alpha=0.3)
        axes[stage].set_xlim([0, fs_values[stage] / 2 / 1e6])
    
    # Plot spectra for 16-bit upconversion stages
    stage_offset = len(stage_outputs)
    for idx, (xi, xq) in enumerate(upconv_outputs):
        freqs, magnitude_db, fft_size = compute_spectrum(xi, xq, fs_values[-1])
        
        one_sided_idx = freqs >= 0
        axes[stage_offset + idx].plot(freqs[one_sided_idx] / 1e6, magnitude_db[one_sided_idx], 'g-', linewidth=0.8)
        title = f'After 16-bit Upconversion' if idx == 0 else f'After Filter Pass {idx}'
        axes[stage_offset + idx].set_title(f'{title}: size = {len(xi)} samples, Fs = {fs_values[-1]/1e6:.1f} MHz', fontsize=11, fontweight='bold')
        axes[stage_offset + idx].set_ylabel('Magnitude (dB)', fontsize=10)
        axes[stage_offset + idx].grid(True, alpha=0.3)
        axes[stage_offset + idx].set_xlim([0, fs_values[-1] / 2 / 1e6])
    
    # Plot after AWGN channel
    xi_final, xq_final = upconv_outputs[-1]
    xi_noisy, xq_noisy = add_awgn_noise(xi_final, xq_final, snr_db=snr_db)
    freqs, magnitude_db_noisy, _ = compute_spectrum(xi_noisy, xq_noisy, fs_values[-1])
    
    stage_idx = stage_offset + num_upconv_stages
    one_sided_idx = freqs >= 0
    axes[stage_idx].plot(freqs[one_sided_idx] / 1e6, magnitude_db_noisy[one_sided_idx], 'r-', linewidth=0.8, label=f'SNR = {snr_db} dB')
    axes[stage_idx].set_title(f'After AWGN Channel (SNR = {snr_db} dB)', fontsize=11, fontweight='bold')
    axes[stage_idx].set_ylabel('Magnitude (dB)', fontsize=10)
    axes[stage_idx].set_xlabel('Frequency (MHz)', fontsize=10)
    axes[stage_idx].grid(True, alpha=0.3)
    axes[stage_idx].set_xlim([0, fs_values[-1] / 2 / 1e6])
    axes[stage_idx].legend(fontsize=10)
    
    # Before and After comparison
    xi_final_clean = upconv_outputs[-1][0]
    xq_final_clean = upconv_outputs[-1][1]
    freqs_clean, magnitude_db_clean, _ = compute_spectrum(xi_final_clean, xq_final_clean, fs_values[-1])
    
    stage_idx = stage_offset + num_upconv_stages + 1
    one_sided_idx = freqs_clean >= 0
    axes[stage_idx].plot(freqs_clean[one_sided_idx] / 1e6, magnitude_db_clean[one_sided_idx], 'b-', linewidth=0.8, label='Before Channel')
    axes[stage_idx].plot(freqs[one_sided_idx] / 1e6, magnitude_db_noisy[one_sided_idx], 'r-', linewidth=0.8, label=f'After AWGN ({snr_db} dB)')
    axes[stage_idx].set_title('Pre vs Post Channel Comparison', fontsize=11, fontweight='bold')
    axes[stage_idx].set_ylabel('Magnitude (dB)', fontsize=10)
    axes[stage_idx].set_xlabel('Frequency (MHz)', fontsize=10)
    axes[stage_idx].grid(True, alpha=0.3)
    axes[stage_idx].set_xlim([0, fs_values[-1] / 2 / 1e6])
    axes[stage_idx].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('spectral_analysis.png', dpi=150, bbox_inches='tight')
    print("Spectral analysis plot saved: spectral_analysis.png")
    plt.show()


# ============================================================================
# 8. MAIN SIMULATION
# ============================================================================

def main():
    """Run complete signal processing pipeline simulation."""
    print("=" * 70)
    print("FFT-Based Signal Processing Pipeline Simulation")
    print("=" * 70)
    
    # Step 1: Generate input signal
    print("\n[1] Generating input signal (2000 samples @ 40 MHz, 20 MHz signal)...")
    xi, xq, xi_ref, xq_ref = generate_input_signal()
    print(f"    Input I component range: [{xi.min():.4f}, {xi.max():.4f}]")
    print(f"    Input Q component range: [{xq.min():.4f}, {xq.max():.4f}]")
    
    # Step 2: Cascaded upsampling (6 stages)
    print("\n[2] Running 6-stage cascaded upsampling pipeline (64x total)...")
    xi_final, xq_final, stage_outputs = cascaded_upsampling_pipeline(xi, xq, num_stages=6)
    final_samples = len(xi_final)
    print(f"    Final output size: {final_samples} samples")
    print(f"    Final output I range: [{xi_final.min():.4f}, {xi_final.max():.4f}]")
    print(f"    Final output Q range: [{xq_final.min():.4f}, {xq_final.max():.4f}]")
    
    # Step 3: 16-bit upconversion on upsampled and filtered signal
    print("\n[3] Applying 16-point hardware FFT-based upconversion...")
    xi_upconv, xq_upconv = digital_upconverter_16point_hardware(xi_final, xq_final, shift=1)
    print(f"    After 16-bit upconversion: size = {len(xi_upconv)} samples")
    print(f"    I range: [{xi_upconv.min():.4f}, {xi_upconv.max():.4f}]")
    print(f"    Q range: [{xq_upconv.min():.4f}, {xq_upconv.max():.4f}]")
    
    # Step 4: Pass through filter again
    print("\n[4] Passing through elliptic IIR filter again...")
    filter_order = 6
    b, a = signal.ellip(filter_order, 0.5, 50, 0.4)
    
    if len(xi_upconv) > 2 * len(b):
        xi_filtered = signal.filtfilt(b, a, xi_upconv)
        xq_filtered = signal.filtfilt(b, a, xq_upconv)
    else:
        xi_filtered = signal.lfilter(b, a, xi_upconv)
        xq_filtered = signal.lfilter(b, a, xq_upconv)
    
    # Quantize to 14-bit fixed-point
    xi_filtered = np.clip(xi_filtered, -1, 1)
    xq_filtered = np.clip(xq_filtered, -1, 1)
    xi_filtered = np.round(xi_filtered * Q_FACTOR) / Q_FACTOR
    xq_filtered = np.round(xq_filtered * Q_FACTOR) / Q_FACTOR
    
    print(f"    After filtering: size = {len(xi_filtered)} samples")
    print(f"    I range: [{xi_filtered.min():.4f}, {xi_filtered.max():.4f}]")
    print(f"    Q range: [{xq_filtered.min():.4f}, {xq_filtered.max():.4f}]")
    
    # Store upconversion outputs for visualization
    upconv_outputs = [
        (xi_upconv.copy(), xq_upconv.copy()),
        (xi_filtered.copy(), xq_filtered.copy())
    ]
    
    # Step 5: AWGN channel simulation at multiple SNR levels
    print("\n[5] Simulating AWGN channel at multiple SNR levels...")
    snr_levels = [10, 20, 30]
    for snr in snr_levels:
        xi_noisy, xq_noisy = add_awgn_noise(xi_filtered, xq_filtered, snr_db=snr)
        print(f"    SNR = {snr} dB: Signal power = {np.mean(xi_filtered**2 + xq_filtered**2):.6f}, Noise power = {np.mean((xi_noisy - xi_filtered)**2 + (xq_noisy - xq_filtered)**2):.6f}")
    
    # Step 6: Spectral analysis and visualization
    print("\n[6] Computing spectral analysis at each stage...")
    plot_spectral_analysis(stage_outputs, upconv_outputs, snr_db=20)
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
