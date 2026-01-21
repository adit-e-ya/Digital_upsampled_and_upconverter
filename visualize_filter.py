"""
Visualization script for the Elliptical IIR Filter.

This script demonstrates the filter's behavior by:
1. Showing input signal before filtering
2. Showing output signal after filtering
3. Displaying magnitude response of the filter
4. Displaying phase response of the filter
"""

import numpy as np
import matplotlib.pyplot as plt
from elliptical_iir import FixedElliptical
from input_gen import Sin_gen, Cos_gen, float_to_fixed_binary
from upsampler import zoh_interpolation


def binary_to_int(binary_str):
    """Convert binary string (two's complement) to integer."""
    val = int(binary_str, 2)
    if val >= 2 ** (len(binary_str) - 1):
        val -= 2 ** len(binary_str)
    return val


def visualize_filter(save_path='filter_visualization.png'):
    """
    Create comprehensive visualization of the elliptical IIR filter.
    
    Parameters
    ----------
    save_path : str
        Path to save the visualization image
    """
    # Configuration
    bin_freq = 50
    nsam = 128
    nfft = 1024
    total_bits = 14
    frac_bits = 13
    
    print("Generating test signal...")
    # Generate I and Q components
    i_signal = Sin_gen(bin_freq, nsam, nfft)
    q_signal = Cos_gen(bin_freq, nsam, nfft)
    
    # Convert to fixed-point binary
    i_binary = float_to_fixed_binary(i_signal, total_bits, frac_bits)
    q_binary = float_to_fixed_binary(q_signal, total_bits, frac_bits)
    
    # Convert to integers
    i_int = [binary_to_int(b) for b in i_binary]
    q_int = [binary_to_int(b) for b in q_binary]
    
    # Upsample
    print("Upsampling signal...")
    upsampled_i, upsampled_q = zoh_interpolation(i_int, q_int, factor=2)
    
    # Apply filter
    print("Applying elliptical IIR filter...")
    filt = FixedElliptical(
        bit_width=14,
        filter_bits=32,
        order=6,
        rp=0.1,
        rs=100.0,
        wp=0.5,
        fs=2.0
    )
    
    filtered_i, filtered_q = filt.process(upsampled_i, upsampled_q)
    
    # Get frequency response of filter
    print("Computing filter frequency response...")
    w, mag, phase = filt.get_frequency_response(worN=512)
    
    # Compute FFT of input and output signals
    print("Computing signal frequency spectra...")
    n_fft = len(upsampled_i)
    freq = np.fft.fftfreq(n_fft, d=1.0/filt.fs)
    freq_positive = freq[:n_fft//2]
    
    # Input signal FFT (I channel)
    input_fft_i = np.fft.fft(upsampled_i)
    input_mag_i = 20 * np.log10(np.abs(input_fft_i[:n_fft//2]) + 1e-10)
    
    # Output signal FFT (I channel)
    output_fft_i = np.fft.fft(filtered_i)
    output_mag_i = 20 * np.log10(np.abs(output_fft_i[:n_fft//2]) + 1e-10)
    
    # Input signal FFT (Q channel)
    input_fft_q = np.fft.fft(upsampled_q)
    input_mag_q = 20 * np.log10(np.abs(input_fft_q[:n_fft//2]) + 1e-10)
    
    # Output signal FFT (Q channel)
    output_fft_q = np.fft.fft(filtered_q)
    output_mag_q = 20 * np.log10(np.abs(output_fft_q[:n_fft//2]) + 1e-10)
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Input Signal (I channel) - Time Domain
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(upsampled_i[:100], 'b-', linewidth=1.5)
    ax1.set_title('Input Signal (I Channel)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # 2. Input Signal (I channel) - Frequency Domain
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(freq_positive, input_mag_i, 'b-', linewidth=1.5)
    ax2.set_title('Input Spectrum (I Channel)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frequency (normalized)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, filt.fs/2)
    
    # 3. Input vs Output Comparison (I channel) - Time Domain
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(upsampled_i[:100], 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax3.plot(filtered_i[:100], 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax3.set_title('I Channel Comparison (Time)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # 4. Output Signal (I channel) - Time Domain
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(filtered_i[:100], 'r-', linewidth=1.5)
    ax4.set_title('Output Signal (I Channel)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    
    # 5. Output Signal (I channel) - Frequency Domain
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(freq_positive, output_mag_i, 'r-', linewidth=1.5)
    ax5.set_title('Output Spectrum (I Channel)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Frequency (normalized)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, filt.fs/2)
    
    # 6. Input vs Output Comparison (I channel) - Frequency Domain
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(freq_positive, input_mag_i, 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax6.plot(freq_positive, output_mag_i, 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax6.set_title('I Channel Comparison (Freq)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Frequency (normalized)')
    ax6.set_ylabel('Magnitude (dB)')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, filt.fs/2)
    
    # 7. Input Signal (Q channel) - Time Domain
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(upsampled_q[:100], 'b-', linewidth=1.5)
    ax7.set_title('Input Signal (Q Channel)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('Amplitude')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 100)
    
    # 8. Input Signal (Q channel) - Frequency Domain
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(freq_positive, input_mag_q, 'b-', linewidth=1.5)
    ax8.set_title('Input Spectrum (Q Channel)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Frequency (normalized)')
    ax8.set_ylabel('Magnitude (dB)')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, filt.fs/2)
    
    # 9. Input vs Output Comparison (Q channel) - Time Domain
    ax9 = plt.subplot(4, 3, 9)
    ax9.plot(upsampled_q[:100], 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax9.plot(filtered_q[:100], 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax9.set_title('Q Channel Comparison (Time)', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Sample Index')
    ax9.set_ylabel('Amplitude')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, 100)
    
    # 10. Filter Magnitude Response
    ax10 = plt.subplot(4, 3, 10)
    ax10.plot(w, mag, 'g-', linewidth=2)
    ax10.axvline(filt.wp, color='orange', linestyle='--', linewidth=1.5, label=f'Cutoff (Wp={filt.wp})')
    ax10.axhline(-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Passband Ripple (±0.1 dB)')
    ax10.axhline(-100, color='purple', linestyle=':', linewidth=1, alpha=0.5, label='Stopband Atten (100 dB)')
    ax10.set_title('Filter Magnitude Response', fontsize=11, fontweight='bold')
    ax10.set_xlabel('Frequency (normalized)')
    ax10.set_ylabel('Magnitude (dB)')
    ax10.grid(True, alpha=0.3)
    ax10.legend(loc='upper right', fontsize=7)
    ax10.set_ylim(-120, 5)
    
    # 11. Filter Phase Response
    ax11 = plt.subplot(4, 3, 11)
    ax11.plot(w, np.degrees(phase), 'm-', linewidth=2)
    ax11.axvline(filt.wp, color='orange', linestyle='--', linewidth=1.5, label=f'Cutoff (Wp={filt.wp})')
    ax11.set_title('Filter Phase Response', fontsize=11, fontweight='bold')
    ax11.set_xlabel('Frequency (normalized)')
    ax11.set_ylabel('Phase (degrees)')
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc='upper right', fontsize=8)
    
    # 12. Q Channel Frequency Comparison
    ax12 = plt.subplot(4, 3, 12)
    ax12.plot(freq_positive, input_mag_q, 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax12.plot(freq_positive, output_mag_q, 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax12.set_title('Q Channel Comparison (Freq)', fontsize=11, fontweight='bold')
    ax12.set_xlabel('Frequency (normalized)')
    ax12.set_ylabel('Magnitude (dB)')
    ax12.legend(loc='upper right', fontsize=8)
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim(0, filt.fs/2)
    
    # Add overall title
    fig.suptitle('Elliptical IIR Filter Visualization (Time & Frequency Domain)\n' + 
                 f'Order={filt.order}, Rp={filt.rp}dB, Rs={filt.rs}dB, Wp={filt.wp}',
                 fontsize=14, fontweight='bold', y=0.997)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    # Display filter statistics
    print("\n" + "="*60)
    print("FILTER STATISTICS")
    print("="*60)
    print(f"Filter Order: {filt.order}")
    print(f"Number of SOS Sections: {filt.num_sections}")
    print(f"Passband Ripple: {filt.rp} dB")
    print(f"Stopband Attenuation: {filt.rs} dB")
    print(f"Cutoff Frequency: {filt.wp} (normalized)")
    print(f"Sampling Frequency: {filt.fs} (normalized)")
    
    # Compute actual response characteristics
    passband_idx = w < filt.wp
    stopband_idx = w > filt.wp
    passband_mag = mag[passband_idx]
    stopband_mag = mag[stopband_idx]
    
    print(f"\nMeasured Characteristics:")
    print(f"  Passband ripple: {np.max(passband_mag) - np.min(passband_mag):.3f} dB")
    print(f"  Stopband attenuation: {-np.max(stopband_mag):.1f} dB")
    
    print("\nSignal Processing:")
    print(f"  Input samples: {len(upsampled_i)}")
    print(f"  Output samples: {len(filtered_i)}")
    print(f"  Input range (I): [{min(upsampled_i)}, {max(upsampled_i)}]")
    print(f"  Output range (I): [{min(filtered_i)}, {max(filtered_i)}]")
    print("="*60)
    
    return fig


if __name__ == "__main__":
    print("="*60)
    print("Elliptical IIR Filter Visualization")
    print("="*60)
    visualize_filter()
    print("\n✓ Visualization complete!")
