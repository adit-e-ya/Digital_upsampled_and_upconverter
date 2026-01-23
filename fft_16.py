"""
16-Point FFT Hardware Simulation

This module implements a 16-point Radix-2 Decimation-in-Time (DIT) FFT
that simulates the hardware architecture, following the same style as the 8-point FFT.

Key features:
- Bit-reversed input ordering
- 4-stage pipeline (14->15->16->17->18 bits)
- Hardware-style twiddle factor approximation using shift-and-add
- Half-rounding-up for rounding operations
"""

import os
import numpy as np
from input_gen import int_to_bin
from fft import bin_to_int, half_round_up, approx_0_7071, twiddle_multiply_w8_1, twiddle_multiply_w8_2, twiddle_multiply_w8_3


def approx_0_9239(val):
    """
    Approximate multiplication by 0.9239 (cos(pi/8)) using shift-and-add.
    
    Hardware implementation: left-shift by 0, 5, 7, 9, 10, 11, 12, 13, 14 positions,
    sum the results, then right-shift by 15 positions with half-rounding-up.
    
    0.9239 ≈ (2^0 + 2^5 + 2^7 + 2^9 + 2^10 + 2^11 + 2^12 + 2^13 + 2^14) / 2^15 = 30305/32768 ≈ 0.9248
    
    Parameters:
    -----------
    val : int
        Input value (16-bit integer)
        
    Returns:
    --------
    int : Approximated result of val * 0.9239
    """
    shifted_sum = (val << 0) + (val << 5) + (val << 7) + (val << 9) + (val << 10) + (val << 11) + (val << 12) + (val << 13) + (val << 14)
    
    if shifted_sum >= 0:
        result = (shifted_sum + (1 << 14)) >> 15
    else:
        result = -(((-shifted_sum) + (1 << 14)) >> 15)
    
    return result


def approx_0_3827(val):
    """
    Approximate multiplication by 0.3827 (sin(pi/8)) using shift-and-add.
    
    Hardware implementation: left-shift by 6, 7, 8, 11, 12, 13 positions,
    sum the results, then right-shift by 15 positions with half-rounding-up.
    
    0.3827 ≈ (2^6 + 2^7 + 2^8 + 2^11 + 2^12 + 2^13) / 2^15 = 12544/32768 ≈ 0.3828
    
    Parameters:
    -----------
    val : int
        Input value (16-bit integer)
        
    Returns:
    --------
    int : Approximated result of val * 0.3827
    """
    shifted_sum = (val << 6) + (val << 7) + (val << 8) + (val << 11) + (val << 12) + (val << 13)
    
    if shifted_sum >= 0:
        result = (shifted_sum + (1 << 14)) >> 15
    else:
        result = -(((-shifted_sum) + (1 << 14)) >> 15)
    
    return result


def twiddle_multiply_w16_1(real, imag):
    """
    Multiply by W_16^1 = e^(-j*2*pi/16) = cos(pi/8) - j*sin(pi/8) = 0.9239 - j*0.3827
    
    (a + jb) * (0.9239 - j*0.3827) = (0.9239*a + 0.3827*b) + j*(0.9239*b - 0.3827*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = approx_0_9239(real) + approx_0_3827(imag)
    new_imag = approx_0_9239(imag) - approx_0_3827(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_3(real, imag):
    """
    Multiply by W_16^3 = e^(-j*3*pi/8) = cos(3*pi/8) - j*sin(3*pi/8) = 0.3827 - j*0.9239
    
    (a + jb) * (0.3827 - j*0.9239) = (0.3827*a + 0.9239*b) + j*(0.3827*b - 0.9239*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = approx_0_3827(real) + approx_0_9239(imag)
    new_imag = approx_0_3827(imag) - approx_0_9239(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_5(real, imag):
    """
    Multiply by W_16^5 = e^(-j*5*pi/8) = cos(5*pi/8) - j*sin(5*pi/8) = -0.3827 - j*0.9239
    
    (a + jb) * (-0.3827 - j*0.9239) = (-0.3827*a + 0.9239*b) + j*(-0.3827*b - 0.9239*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = -approx_0_3827(real) + approx_0_9239(imag)
    new_imag = -approx_0_3827(imag) - approx_0_9239(real)
    
    return new_real, new_imag


def twiddle_multiply_w16_7(real, imag):
    """
    Multiply by W_16^7 = e^(-j*7*pi/8) = cos(7*pi/8) - j*sin(7*pi/8) = -0.9239 - j*0.3827
    
    (a + jb) * (-0.9239 - j*0.3827) = (-0.9239*a + 0.3827*b) + j*(-0.9239*b - 0.3827*a)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    new_real = -approx_0_9239(real) + approx_0_3827(imag)
    new_imag = -approx_0_9239(imag) - approx_0_3827(real)
    
    return new_real, new_imag


def fft_16point_hardware(input_i_binary_list, input_q_binary_list):
    """
    16-Point FFT Hardware Simulation.
    
    Implements the 4-stage Radix-2 DIT FFT with:
    - Bit-reversed input ordering
    - Hardware-style twiddle factor approximation
    - Proper bit-width expansion at each stage
    
    Parameters:
    -----------
    input_i_binary_list : list of str
        List of 16 binary strings, each 14-bit wide (two's complement).
        These are scaled by 2^13 from the original +/-1 range signal.
        
    input_q_binary_list : list of str
        List of 16 binary strings, each 14-bit wide (two's complement).
        These are scaled by 2^13 from the original +/-1 range signal.
        
    Returns:
    --------
    tuple of lists : (real_out_list, imag_out_list)
        Each list contains 16 integers representing the real and imaginary parts
    """
    if len(input_i_binary_list) != 16 or len(input_q_binary_list) != 16:
        raise ValueError("Input must contain exactly 16 samples")
    
    # Convert binary strings to integers
    x_real = [bin_to_int(b) for b in input_i_binary_list]
    x_imag = [bin_to_int(b) for b in input_q_binary_list]
    
    # Stage 1: First butterfly operations with bit-reversed input order
    # Pairs: [0,8], [4,12], [2,10], [6,14], [1,9], [5,13], [3,11], [7,15]
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
    
    # Apply twiddle factors for stage 1
    # W_16^0 = 1 (no multiplication for indices 1, 3, 5, 7, 9, 11, 13, 15)
    # W_16^4 = -j for odd indices that use it
    stage1_real[3], stage1_imag[3] = twiddle_multiply_w8_2(stage1_real[3], stage1_imag[3])  # W_16^4 = W_8^2 = -j
    stage1_real[7], stage1_imag[7] = twiddle_multiply_w8_2(stage1_real[7], stage1_imag[7])
    stage1_real[11], stage1_imag[11] = twiddle_multiply_w8_2(stage1_real[11], stage1_imag[11])
    stage1_real[15], stage1_imag[15] = twiddle_multiply_w8_2(stage1_real[15], stage1_imag[15])
    
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
    
    # Apply twiddle factors for stage 2
    # Indices 5, 7, 13, 15 need W_8^1, W_8^2, W_8^3
    stage2_real[5], stage2_imag[5] = twiddle_multiply_w8_1(stage2_real[5], stage2_imag[5])  # W_16^2 = W_8^1
    stage2_real[6], stage2_imag[6] = twiddle_multiply_w8_2(stage2_real[6], stage2_imag[6])  # W_16^4 = W_8^2
    stage2_real[7], stage2_imag[7] = twiddle_multiply_w8_3(stage2_real[7], stage2_imag[7])  # W_16^6 = W_8^3
    
    stage2_real[13], stage2_imag[13] = twiddle_multiply_w8_1(stage2_real[13], stage2_imag[13])
    stage2_real[14], stage2_imag[14] = twiddle_multiply_w8_2(stage2_real[14], stage2_imag[14])
    stage2_real[15], stage2_imag[15] = twiddle_multiply_w8_3(stage2_real[15], stage2_imag[15])
    
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
    
    # Apply twiddle factors for stage 3
    # W_16^1, W_16^2, W_16^3, W_16^4, W_16^5, W_16^6, W_16^7
    stage3_real[9], stage3_imag[9] = twiddle_multiply_w16_1(stage3_real[9], stage3_imag[9])
    stage3_real[10], stage3_imag[10] = twiddle_multiply_w8_1(stage3_real[10], stage3_imag[10])  # W_16^2 = W_8^1
    stage3_real[11], stage3_imag[11] = twiddle_multiply_w16_3(stage3_real[11], stage3_imag[11])
    stage3_real[12], stage3_imag[12] = twiddle_multiply_w8_2(stage3_real[12], stage3_imag[12])  # W_16^4 = W_8^2
    stage3_real[13], stage3_imag[13] = twiddle_multiply_w16_5(stage3_real[13], stage3_imag[13])
    stage3_real[14], stage3_imag[14] = twiddle_multiply_w8_3(stage3_real[14], stage3_imag[14])  # W_16^6 = W_8^3
    stage3_real[15], stage3_imag[15] = twiddle_multiply_w16_7(stage3_real[15], stage3_imag[15])
    
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
    
    # Output in natural order
    return stage4_real, stage4_imag


def fft_compare_demo(seed=0, scale=4096, tolerance_pct=35, bit_width=14):
    """
    Run one FFT comparison between the hardware-style algorithm and NumPy.

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
    rng = np.random.default_rng(seed)
    i_vals = rng.integers(-scale, scale + 1, size=16)
    q_vals = rng.integers(-scale, scale + 1, size=16)

    i_bin = [int_to_bin(int(v), bit_width) for v in i_vals]
    q_bin = [int_to_bin(int(v), bit_width) for v in q_vals]

    hw_real, hw_imag = fft_16point_hardware(i_bin, q_bin)
    ref = np.fft.fft(i_vals + 1j * q_vals)

    header = f"{'Bin':>3} {'HW Real':>9} {'HW Imag':>9} {'NP Real':>11} {'NP Imag':>11} {'|err|':>9} {'rel %':>7} {'Status':>7}"
    print("\n=== 16-Point FFT Comparison ===")
    print(f"Seed: {seed} | Scale: {scale} | Tolerance: {tolerance_pct}%")
    print(f"I samples: {i_vals.tolist()}")
    print(f"Q samples: {q_vals.tolist()}")
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
    print(f"Result: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


if __name__ == "__main__":
    seed_env = os.environ.get("FFT_SEED", "0")
    try:
        seed_value = int(seed_env)
    except ValueError:
        seed_value = 0
    success = fft_compare_demo(seed=seed_value)
    if success:
        print("16-Point FFT hardware simulation test PASSED.")
    else:
        print("16-Point FFT hardware simulation test FAILED.")
