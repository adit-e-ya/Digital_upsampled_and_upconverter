"""
16-Point IFFT Hardware Simulation

This module implements a 16-point Radix-2 Decimation-in-Time (DIT) IFFT
that simulates the hardware architecture, mirroring the 16-point FFT implementation.

Key features:
- Bit-reversed input ordering
- 4-stage pipeline (14->15->16->17->18 bits)
- Hardware-style twiddle factor approximation using shift-and-add
- Conjugate twiddle factors compared to FFT (positive exponents)
- Scaling by 1/16 at the output

Mathematical Relationship:
IFFT(X[k]) = (1/N) * conjugate(FFT(conjugate(X[k])))
Where N is the number of points (16 in this case)
"""

import os
import numpy as np
from input_gen import int_to_bin
from fft import bin_to_int, half_round_up, approx_0_7071
from fft_16 import approx_0_9239, approx_0_3827
from ifft import twiddle_multiply_w8_minus1, twiddle_multiply_w8_minus2, twiddle_multiply_w8_minus3


def twiddle_multiply_w16_minus1(real, imag):
    """
    Multiply by W_16^(-1) = e^(j*2*pi*(-1)/16) = e^(j*pi/8) = cos(pi/8) + j*sin(pi/8) = 0.9239 + j*0.3827
    
    This is the conjugate of W_16^1 used in FFT.
    (a + jb) * (0.9239 + j*0.3827) = (0.9239*a - 0.3827*b) + j*(0.9239*b + 0.3827*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = approx_0_9239(real) - approx_0_3827(imag)
    new_imag = approx_0_9239(imag) + approx_0_3827(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_minus3(real, imag):
    """
    Multiply by W_16^(-3) = e^(j*2*pi*(-3)/16) = e^(j*3*pi/8) = cos(3*pi/8) + j*sin(3*pi/8) = 0.3827 + j*0.9239
    
    This is the conjugate of W_16^3 used in FFT.
    (a + jb) * (0.3827 + j*0.9239) = (0.3827*a - 0.9239*b) + j*(0.3827*b + 0.9239*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = approx_0_3827(real) - approx_0_9239(imag)
    new_imag = approx_0_3827(imag) + approx_0_9239(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_minus5(real, imag):
    """
    Multiply by W_16^(-5) = e^(j*2*pi*(-5)/16) = e^(j*5*pi/8) = cos(5*pi/8) + j*sin(5*pi/8) = -0.3827 + j*0.9239
    
    This is the conjugate of W_16^5 used in FFT.
    (a + jb) * (-0.3827 + j*0.9239) = (-0.3827*a - 0.9239*b) + j*(-0.3827*b + 0.9239*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = -approx_0_3827(real) - approx_0_9239(imag)
    new_imag = -approx_0_3827(imag) + approx_0_9239(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_minus7(real, imag):
    """
    Multiply by W_16^(-7) = e^(j*2*pi*(-7)/16) = e^(j*7*pi/8) = cos(7*pi/8) + j*sin(7*pi/8) = -0.9239 + j*0.3827
    
    This is the conjugate of W_16^7 used in FFT.
    (a + jb) * (-0.9239 + j*0.3827) = (-0.9239*a - 0.3827*b) + j*(-0.9239*b + 0.3827*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = -approx_0_9239(real) - approx_0_3827(imag)
    new_imag = -approx_0_9239(imag) + approx_0_3827(real)
    
    return new_real, new_imag


def ifft_16point_hardware(input_i_binary_list, input_q_binary_list):
    """
    16-Point IFFT Hardware Simulation.
    
    Implements the 4-stage Radix-2 DIT IFFT with:
    - Bit-reversed input ordering
    - Hardware-style conjugate twiddle factor approximation
    - Scaling by 1/16 at the output (right shift by 4 bits)
    
    The IFFT is implemented using the relationship:
    IFFT = (1/N) * conjugate(FFT(conjugate(input)))
    But structured to match the FFT's 4-stage architecture with conjugate twiddles.
    
    Parameters:
    -----------
    input_i_binary_list : list of str
        List of 16 binary strings, each 14-bit wide (two's complement).
        These represent the real part of the frequency domain data.
        
    input_q_binary_list : list of str
        List of 16 binary strings, each 14-bit wide (two's complement).
        These represent the imaginary part of the frequency domain data.
        
    Returns:
    --------
    tuple of lists : (real_out_list, imag_out_list)
        Each list contains 16 integers representing the real and imaginary parts
        of the time-domain signal after IFFT
    """
    if len(input_i_binary_list) != 16 or len(input_q_binary_list) != 16:
        raise ValueError("Input must contain exactly 16 samples")
    
    # Convert binary strings to integers
    x_real = [bin_to_int(b) for b in input_i_binary_list]
    x_imag = [bin_to_int(b) for b in input_q_binary_list]
    
    # Stage 1: First butterfly operations with bit-reversed input order
    stage1_real = [0] * 16
    stage1_imag = [0] * 16
    
    stage1_real[0] = x_real[0] + x_real[8]
    stage1_imag[0] = x_imag[0] + x_imag[8]
    stage1_real[1] = x_real[0] - x_real[8]
    stage1_imag[1] = x_imag[0] - x_imag[8]
    
    stage1_real[2] = x_real[4] + x_real[12]
    stage1_imag[2] = x_imag[4] + x_imag[12]
    stage1_real[3] = x_real[4] - x_real[12]
    stage1_imag[3] = x_imag[4] - x_imag[12]
    
    stage1_real[4] = x_real[2] + x_real[10]
    stage1_imag[4] = x_imag[2] + x_imag[10]
    stage1_real[5] = x_real[2] - x_real[10]
    stage1_imag[5] = x_imag[2] - x_imag[10]
    
    stage1_real[6] = x_real[6] + x_real[14]
    stage1_imag[6] = x_imag[6] + x_imag[14]
    stage1_real[7] = x_real[6] - x_real[14]
    stage1_imag[7] = x_imag[6] - x_imag[14]
    
    stage1_real[8] = x_real[1] + x_real[9]
    stage1_imag[8] = x_imag[1] + x_imag[9]
    stage1_real[9] = x_real[1] - x_real[9]
    stage1_imag[9] = x_imag[1] - x_imag[9]
    
    stage1_real[10] = x_real[5] + x_real[13]
    stage1_imag[10] = x_imag[5] + x_imag[13]
    stage1_real[11] = x_real[5] - x_real[13]
    stage1_imag[11] = x_imag[5] - x_imag[13]
    
    stage1_real[12] = x_real[3] + x_real[11]
    stage1_imag[12] = x_imag[3] + x_imag[11]
    stage1_real[13] = x_real[3] - x_real[11]
    stage1_imag[13] = x_imag[3] - x_imag[11]
    
    stage1_real[14] = x_real[7] + x_real[15]
    stage1_imag[14] = x_imag[7] + x_imag[15]
    stage1_real[15] = x_real[7] - x_real[15]
    stage1_imag[15] = x_imag[7] - x_imag[15]
    
    # Apply conjugate twiddle factors for stage 1 (W_16^(-4) = +j instead of -j)
    stage1_real[3], stage1_imag[3] = twiddle_multiply_w8_minus2(stage1_real[3], stage1_imag[3])
    stage1_real[7], stage1_imag[7] = twiddle_multiply_w8_minus2(stage1_real[7], stage1_imag[7])
    stage1_real[11], stage1_imag[11] = twiddle_multiply_w8_minus2(stage1_real[11], stage1_imag[11])
    stage1_real[15], stage1_imag[15] = twiddle_multiply_w8_minus2(stage1_real[15], stage1_imag[15])
    
    # Stage 2: Second butterfly stage
    stage2_real = [0] * 16
    stage2_imag = [0] * 16
    
    stage2_real[0] = stage1_real[0] + stage1_real[2]
    stage2_imag[0] = stage1_imag[0] + stage1_imag[2]
    stage2_real[1] = stage1_real[1] + stage1_real[3]
    stage2_imag[1] = stage1_imag[1] + stage1_imag[3]
    stage2_real[2] = stage1_real[0] - stage1_real[2]
    stage2_imag[2] = stage1_imag[0] - stage1_imag[2]
    stage2_real[3] = stage1_real[1] - stage1_real[3]
    stage2_imag[3] = stage1_imag[1] - stage1_imag[3]
    
    stage2_real[4] = stage1_real[4] + stage1_real[6]
    stage2_imag[4] = stage1_imag[4] + stage1_imag[6]
    stage2_real[5] = stage1_real[5] + stage1_real[7]
    stage2_imag[5] = stage1_imag[5] + stage1_imag[7]
    stage2_real[6] = stage1_real[4] - stage1_real[6]
    stage2_imag[6] = stage1_imag[4] - stage1_imag[6]
    stage2_real[7] = stage1_real[5] - stage1_real[7]
    stage2_imag[7] = stage1_imag[5] - stage1_imag[7]
    
    stage2_real[8] = stage1_real[8] + stage1_real[10]
    stage2_imag[8] = stage1_imag[8] + stage1_imag[10]
    stage2_real[9] = stage1_real[9] + stage1_real[11]
    stage2_imag[9] = stage1_imag[9] + stage1_imag[11]
    stage2_real[10] = stage1_real[8] - stage1_real[10]
    stage2_imag[10] = stage1_imag[8] - stage1_imag[10]
    stage2_real[11] = stage1_real[9] - stage1_real[11]
    stage2_imag[11] = stage1_imag[9] - stage1_imag[11]
    
    stage2_real[12] = stage1_real[12] + stage1_real[14]
    stage2_imag[12] = stage1_imag[12] + stage1_imag[14]
    stage2_real[13] = stage1_real[13] + stage1_real[15]
    stage2_imag[13] = stage1_imag[13] + stage1_imag[15]
    stage2_real[14] = stage1_real[12] - stage1_real[14]
    stage2_imag[14] = stage1_imag[12] - stage1_imag[14]
    stage2_real[15] = stage1_real[13] - stage1_real[15]
    stage2_imag[15] = stage1_imag[13] - stage1_imag[15]
    
    # Apply conjugate twiddle factors for stage 2
    stage2_real[5], stage2_imag[5] = twiddle_multiply_w8_minus1(stage2_real[5], stage2_imag[5])
    stage2_real[6], stage2_imag[6] = twiddle_multiply_w8_minus2(stage2_real[6], stage2_imag[6])
    stage2_real[7], stage2_imag[7] = twiddle_multiply_w8_minus3(stage2_real[7], stage2_imag[7])
    
    stage2_real[13], stage2_imag[13] = twiddle_multiply_w8_minus1(stage2_real[13], stage2_imag[13])
    stage2_real[14], stage2_imag[14] = twiddle_multiply_w8_minus2(stage2_real[14], stage2_imag[14])
    stage2_real[15], stage2_imag[15] = twiddle_multiply_w8_minus3(stage2_real[15], stage2_imag[15])
    
    # Stage 3: Third butterfly stage
    stage3_real = [0] * 16
    stage3_imag = [0] * 16
    
    stage3_real[0] = stage2_real[0] + stage2_real[4]
    stage3_imag[0] = stage2_imag[0] + stage2_imag[4]
    stage3_real[1] = stage2_real[1] + stage2_real[5]
    stage3_imag[1] = stage2_imag[1] + stage2_imag[5]
    stage3_real[2] = stage2_real[2] + stage2_real[6]
    stage3_imag[2] = stage2_imag[2] + stage2_imag[6]
    stage3_real[3] = stage2_real[3] + stage2_real[7]
    stage3_imag[3] = stage2_imag[3] + stage2_imag[7]
    
    stage3_real[4] = stage2_real[0] - stage2_real[4]
    stage3_imag[4] = stage2_imag[0] - stage2_imag[4]
    stage3_real[5] = stage2_real[1] - stage2_real[5]
    stage3_imag[5] = stage2_imag[1] - stage2_imag[5]
    stage3_real[6] = stage2_real[2] - stage2_real[6]
    stage3_imag[6] = stage2_imag[2] - stage2_imag[6]
    stage3_real[7] = stage2_real[3] - stage2_real[7]
    stage3_imag[7] = stage2_imag[3] - stage2_imag[7]
    
    stage3_real[8] = stage2_real[8] + stage2_real[12]
    stage3_imag[8] = stage2_imag[8] + stage2_imag[12]
    stage3_real[9] = stage2_real[9] + stage2_real[13]
    stage3_imag[9] = stage2_imag[9] + stage2_imag[13]
    stage3_real[10] = stage2_real[10] + stage2_real[14]
    stage3_imag[10] = stage2_imag[10] + stage2_imag[14]
    stage3_real[11] = stage2_real[11] + stage2_real[15]
    stage3_imag[11] = stage2_imag[11] + stage2_imag[15]
    
    stage3_real[12] = stage2_real[8] - stage2_real[12]
    stage3_imag[12] = stage2_imag[8] - stage2_imag[12]
    stage3_real[13] = stage2_real[9] - stage2_real[13]
    stage3_imag[13] = stage2_imag[9] - stage2_imag[13]
    stage3_real[14] = stage2_real[10] - stage2_real[14]
    stage3_imag[14] = stage2_imag[10] - stage2_imag[14]
    stage3_real[15] = stage2_real[11] - stage2_real[15]
    stage3_imag[15] = stage2_imag[11] - stage2_imag[15]
    
    # Apply conjugate twiddle factors for stage 3
    stage3_real[9], stage3_imag[9] = twiddle_multiply_w16_minus1(stage3_real[9], stage3_imag[9])
    stage3_real[10], stage3_imag[10] = twiddle_multiply_w8_minus1(stage3_real[10], stage3_imag[10])
    stage3_real[11], stage3_imag[11] = twiddle_multiply_w16_minus3(stage3_real[11], stage3_imag[11])
    stage3_real[12], stage3_imag[12] = twiddle_multiply_w8_minus2(stage3_real[12], stage3_imag[12])
    stage3_real[13], stage3_imag[13] = twiddle_multiply_w16_minus5(stage3_real[13], stage3_imag[13])
    stage3_real[14], stage3_imag[14] = twiddle_multiply_w8_minus3(stage3_real[14], stage3_imag[14])
    stage3_real[15], stage3_imag[15] = twiddle_multiply_w16_minus7(stage3_real[15], stage3_imag[15])
    
    # Stage 4: Final butterfly stage
    stage4_real = [0] * 16
    stage4_imag = [0] * 16
    
    stage4_real[0] = stage3_real[0] + stage3_real[8]
    stage4_imag[0] = stage3_imag[0] + stage3_imag[8]
    stage4_real[1] = stage3_real[1] + stage3_real[9]
    stage4_imag[1] = stage3_imag[1] + stage3_imag[9]
    stage4_real[2] = stage3_real[2] + stage3_real[10]
    stage4_imag[2] = stage3_imag[2] + stage3_imag[10]
    stage4_real[3] = stage3_real[3] + stage3_real[11]
    stage4_imag[3] = stage3_imag[3] + stage3_imag[11]
    stage4_real[4] = stage3_real[4] + stage3_real[12]
    stage4_imag[4] = stage3_imag[4] + stage3_imag[12]
    stage4_real[5] = stage3_real[5] + stage3_real[13]
    stage4_imag[5] = stage3_imag[5] + stage3_imag[13]
    stage4_real[6] = stage3_real[6] + stage3_real[14]
    stage4_imag[6] = stage3_imag[6] + stage3_imag[14]
    stage4_real[7] = stage3_real[7] + stage3_real[15]
    stage4_imag[7] = stage3_imag[7] + stage3_imag[15]
    
    stage4_real[8] = stage3_real[0] - stage3_real[8]
    stage4_imag[8] = stage3_imag[0] - stage3_imag[8]
    stage4_real[9] = stage3_real[1] - stage3_real[9]
    stage4_imag[9] = stage3_imag[1] - stage3_imag[9]
    stage4_real[10] = stage3_real[2] - stage3_real[10]
    stage4_imag[10] = stage3_imag[2] - stage3_imag[10]
    stage4_real[11] = stage3_real[3] - stage3_real[11]
    stage4_imag[11] = stage3_imag[3] - stage3_imag[11]
    stage4_real[12] = stage3_real[4] - stage3_real[12]
    stage4_imag[12] = stage3_imag[4] - stage3_imag[12]
    stage4_real[13] = stage3_real[5] - stage3_real[13]
    stage4_imag[13] = stage3_imag[5] - stage3_imag[13]
    stage4_real[14] = stage3_real[6] - stage3_real[14]
    stage4_imag[14] = stage3_imag[6] - stage3_imag[14]
    stage4_real[15] = stage3_real[7] - stage3_real[15]
    stage4_imag[15] = stage3_imag[7] - stage3_imag[15]
    
    # Scale by 1/16 (right shift by 4 bits) for IFFT normalization
    output_real = [val >> 4 for val in stage4_real]
    output_imag = [val >> 4 for val in stage4_imag]
    
    # Output in natural order
    return output_real, output_imag


def ifft_compare_demo(seed=0, scale=8192, tolerance_pct=35, bit_width=14):
    """
    Run one IFFT comparison between the hardware-style algorithm and NumPy.
    Also verifies the round-trip property: IFFT(FFT(x)) ≈ x

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    scale : int
        Maximum absolute value for generated I/Q samples (must fit bit_width).
    tolerance_pct : float
        Allowed relative magnitude error percentage for PASS/FAIL.
    bit_width : int
        Bit width used when converting integers to binary strings.

    Returns
    -------
    bool
        True when all bins meet the tolerance.
    """
    from fft_16 import fft_16point_hardware
    
    # Generate random frequency-domain data for IFFT testing
    rng = np.random.default_rng(seed)
    freq_real = rng.integers(-scale, scale + 1, size=16)
    freq_imag = rng.integers(-scale, scale + 1, size=16)

    # Convert to binary representation
    freq_real_bin = [int_to_bin(int(v), bit_width) for v in freq_real]
    freq_imag_bin = [int_to_bin(int(v), bit_width) for v in freq_imag]

    # Run hardware IFFT
    hw_real, hw_imag = ifft_16point_hardware(freq_real_bin, freq_imag_bin)
    
    # Run NumPy IFFT for reference
    freq_complex = freq_real + 1j * freq_imag
    ref = np.fft.ifft(freq_complex)

    # Display results
    header = f"{'Bin':>3} {'HW Real':>9} {'HW Imag':>9} {'NP Real':>11} {'NP Imag':>11} {'|err|':>9} {'rel %':>7} {'Status':>7}"
    print("\n=== 16-Point IFFT Comparison ===")
    print(f"Seed: {seed} | Scale: {scale} | Tolerance: {tolerance_pct}%")
    print(f"Frequency Real: {freq_real.tolist()}")
    print(f"Frequency Imag: {freq_imag.tolist()}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for idx in range(16):
        hw_c = complex(hw_real[idx], hw_imag[idx])
        ref_c = ref[idx]
        err_mag = abs(hw_c - ref_c)
        ref_mag = abs(ref_c)
        rel_pct = (err_mag / ref_mag * 100.0) if ref_mag else 0.0
        status = "PASS" if rel_pct <= tolerance_pct else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{idx:>3} {hw_real[idx]:>9} {hw_imag[idx]:>9} {ref_c.real:>11.2f} {ref_c.imag:>11.2f} {err_mag:>9.2f} {rel_pct:>6.2f}% {status:>7}")

    print("-" * len(header))
    print(f"IFFT Result: {'PASSED' if all_pass else 'FAILED'}")
    
    # Test round-trip: IFFT(FFT(x)) should equal x
    print("\n=== Round-Trip Test: IFFT(FFT(x)) ===")
    
    # Generate time-domain signal
    time_real = rng.integers(-scale, scale + 1, size=16)
    time_imag = rng.integers(-scale, scale + 1, size=16)
    time_real_bin = [int_to_bin(int(v), bit_width) for v in time_real]
    time_imag_bin = [int_to_bin(int(v), bit_width) for v in time_imag]
    
    # FFT then IFFT
    FFT_OUTPUT_BITS = 18  # 16-point FFT expands from 14-bit to 18-bit
    fft_real, fft_imag = fft_16point_hardware(time_real_bin, time_imag_bin)
    fft_real_bin = [int_to_bin(int(v), FFT_OUTPUT_BITS) for v in fft_real]
    fft_imag_bin = [int_to_bin(int(v), FFT_OUTPUT_BITS) for v in fft_imag]
    
    ifft_real, ifft_imag = ifft_16point_hardware(fft_real_bin, fft_imag_bin)
    
    print(f"Original Time: Real={time_real.tolist()}")
    print(f"               Imag={time_imag.tolist()}")
    print(f"After IFFT(FFT): Real={ifft_real}")
    print(f"                 Imag={ifft_imag}")
    
    # Calculate round-trip error
    ROUND_TRIP_ERROR_TOLERANCE = 1500  # Maximum acceptable error in integer units (increased for 16-point approximations)
    round_trip_pass = True
    max_error = 0
    for idx in range(16):
        err_real = abs(time_real[idx] - ifft_real[idx])
        err_imag = abs(time_imag[idx] - ifft_imag[idx])
        max_error = max(max_error, err_real, err_imag)
        if err_real > ROUND_TRIP_ERROR_TOLERANCE or err_imag > ROUND_TRIP_ERROR_TOLERANCE:
            round_trip_pass = False
    
    print(f"Round-trip max error: {max_error}")
    print(f"Round-trip result: {'PASSED' if round_trip_pass else 'FAILED'}")
    
    return all_pass and round_trip_pass


if __name__ == "__main__":
    seed_env = os.environ.get("IFFT_SEED", "0")
    try:
        seed_value = int(seed_env)
    except ValueError:
        seed_value = 0
    success = ifft_compare_demo(seed=seed_value)
    if success:
        print("\n16-Point IFFT hardware simulation test PASSED.")
    else:
        print("\n16-Point IFFT hardware simulation test FAILED.")
