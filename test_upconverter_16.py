"""
Test script for 16-point upconverter

Demonstrates the 16-point FFT-based upconverter functionality.
"""

import numpy as np
from input_gen import float_to_fixed_binary, int_to_bin
from fft import bin_to_int
from upconverter_16 import upconverter_16

def test_upconverter_16():
    """Test the 16-point upconverter with a simple signal."""
    print("=" * 70)
    print("16-Point Upconverter Test")
    print("=" * 70)
    
    # Generate a simple test signal (32 samples = 2 blocks of 16)
    n_samples = 32
    t = np.arange(n_samples)
    
    # Create a simple sine wave
    freq = 2 * np.pi / 16
    i_signal = np.sin(freq * t) * 0.5
    q_signal = np.cos(freq * t) * 0.5
    
    print(f"\nInput signal: {n_samples} samples")
    print(f"I signal range: [{i_signal.min():.4f}, {i_signal.max():.4f}]")
    print(f"Q signal range: [{q_signal.min():.4f}, {q_signal.max():.4f}]")
    
    # Convert to 14-bit fixed-point binary
    i_binary = float_to_fixed_binary(i_signal, total_bits=14, frac_bits=13)
    q_binary = float_to_fixed_binary(q_signal, total_bits=14, frac_bits=13)
    
    # Apply upconverter with shift=1
    print("\nApplying 16-point upconverter (shift=1)...")
    ifft_real, ifft_imag = upconverter_16(i_binary, q_binary, shift=1)
    
    # Convert back to floating point
    scale_factor = 2 ** 13
    i_output = np.array(ifft_real) / scale_factor
    q_output = np.array(ifft_imag) / scale_factor
    
    print(f"\nOutput signal: {len(i_output)} samples")
    print(f"I output range: [{i_output.min():.4f}, {i_output.max():.4f}]")
    print(f"Q output range: [{q_output.min():.4f}, {q_output.max():.4f}]")
    
    # Verify signal energy is preserved (approximately)
    input_energy = np.sum(i_signal**2 + q_signal**2)
    output_energy = np.sum(i_output**2 + q_output**2)
    energy_ratio = output_energy / input_energy
    
    print(f"\nSignal energy:")
    print(f"  Input:  {input_energy:.6f}")
    print(f"  Output: {output_energy:.6f}")
    print(f"  Ratio:  {energy_ratio:.4f}")
    
    # Check if energy is reasonably preserved (within 50% due to hardware approximations)
    if 0.5 <= energy_ratio <= 1.5:
        print("\n✓ Signal energy is reasonably preserved")
        return True
    else:
        print("\n✗ Signal energy changed significantly")
        return False

if __name__ == "__main__":
    success = test_upconverter_16()
    if success:
        print("\n16-Point Upconverter test PASSED.")
    else:
        print("\n16-Point Upconverter test FAILED.")
