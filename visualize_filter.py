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
    
    # Get frequency response
    print("Computing frequency response...")
    w, mag, phase = filt.get_frequency_response(worN=512)
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Input Signal (I channel)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(upsampled_i[:100], 'b-', linewidth=1.5)
    ax1.set_title('Input Signal (I Channel)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # 2. Output Signal (I channel)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(filtered_i[:100], 'r-', linewidth=1.5)
    ax2.set_title('Output Signal (I Channel)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # 3. Input vs Output Comparison (I channel)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(upsampled_i[:100], 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax3.plot(filtered_i[:100], 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax3.set_title('Input vs Output Comparison (I Channel)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # 4. Q Channel Comparison
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(upsampled_q[:100], 'b-', alpha=0.7, linewidth=1.5, label='Input')
    ax4.plot(filtered_q[:100], 'r-', alpha=0.7, linewidth=1.5, label='Output')
    ax4.set_title('Input vs Output Comparison (Q Channel)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 100)
    
    # 5. Magnitude Response
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(w, mag, 'g-', linewidth=2)
    ax5.axvline(filt.wp, color='orange', linestyle='--', linewidth=1.5, label=f'Cutoff (Wp={filt.wp})')
    ax5.axhline(-0.1, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Passband Ripple (±0.1 dB)')
    ax5.axhline(-100, color='purple', linestyle=':', linewidth=1, alpha=0.5, label='Stopband Atten (100 dB)')
    ax5.set_title('Magnitude Response', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Frequency (normalized)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_ylim(-120, 5)
    
    # 6. Phase Response
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(w, np.degrees(phase), 'm-', linewidth=2)
    ax6.axvline(filt.wp, color='orange', linestyle='--', linewidth=1.5, label=f'Cutoff (Wp={filt.wp})')
    ax6.set_title('Phase Response', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Frequency (normalized)')
    ax6.set_ylabel('Phase (degrees)')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=8)
    
    # Add overall title
    fig.suptitle('Elliptical IIR Filter Visualization\n' + 
                 f'Order={filt.order}, Rp={filt.rp}dB, Rs={filt.rs}dB, Wp={filt.wp}',
                 fontsize=14, fontweight='bold', y=0.995)
    
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
