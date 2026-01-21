"""
8-Point FFT Hardware Simulation

This module implements an 8-point Radix-2 Decimation-in-Time (DIT) FFT
that simulates the hardware architecture described in the specification.

Key features:
- Bit-reversed input ordering
- 3-stage pipeline (14->15->16->17 bits)
- Hardware-style twiddle factor approximation using shift-and-add
- Half-rounding-up for rounding operations
"""

import os

import numpy as np
from input_gen import int_to_bin

def half_round_up(value):
    """
    Half-rounding-up technique used in hardware.
    
    For positive numbers: rounds to next higher integer if fractional part >= 0.5
    For negative numbers: rounds towards zero if fractional part is -0.5 or greater
    
    Note: This function documents the hardware rounding behavior. The actual
    rounding in approx_0_7071() is implemented using bit manipulation for
    efficiency, but follows this same logic.
    
    Parameters:
    -----------
    value : float
        The value to round
        
    Returns:
    --------
    int : The rounded integer value
    """
    if value >= 0:
        return int(np.floor(value + 0.5))
    else:
        # For negative numbers, round towards zero
        # -1.5 -> -1, -1.6 -> -2
        return int(np.ceil(value - 0.5))


def bin_to_int(binary_str, signed=True):
    """
    Convert a binary string to a signed integer (two's complement).
    
    Parameters:
    -----------
    binary_str : str
        Binary string representation
    signed : bool
        If True, interpret as two's complement signed integer
        
    Returns:
    --------
    int : The integer value
    """
    total_bits = len(binary_str)
    value = int(binary_str, 2)
    
    if signed and binary_str[0] == '1':
        # Two's complement: subtract 2^n for negative numbers
        value -= (1 << total_bits)
    
    return value



def approx_0_7071(val):
    """
    Approximate multiplication by 0.7071 using shift-and-add.
    
    Hardware implementation: left-shift by 1, 7, 9, 11, 13, and 14 positions,
    sum the results, then right-shift by 15 positions with half-rounding-up.
    
    Parameters:
    -----------
    val : int
        Input value (16-bit integer)
        
    Returns:
    --------
    int : Approximated result of val * 0.7071
    """
    # Left shifts: 1, 7, 9, 11, 12, 14
    # Sum these shifted versions, then right-shift by 15
    shifted_sum = (val << 1) + (val << 7) + (val << 9) + (val << 11) + (val << 12) + (val << 14)
    
    # Right shift by 15 with half-rounding-up
    # Add 2^14 (half of 2^15) before shifting for rounding
    if shifted_sum >= 0:
        result = (shifted_sum + (1 << 14)) >> 15
    else:
        # For negative numbers, round towards zero
        result = -(((-shifted_sum) + (1 << 14)) >> 15)
    
    return result


def twiddle_multiply_w8_1(data_real, data_imag):
    """
    Multiply by W_8^1 = e^(-j*2*pi/8) = cos(pi/4) - j*sin(pi/4) = 0.7071 - j*0.7071
    
    (a + jb) * (0.7071 - j*0.7071) = 0.7071*(a + b) + j*0.7071*(b - a)
    
    Parameters:
    -----------
    data_real : int
        Real part of the input (16-bit integer)
    data_imag : int
        Imaginary part of the input (16-bit integer)
        
    Returns:
    --------
    tuple : (real_out, imag_out) - approximated result
    """
    sum_real_imag = data_real + data_imag
    diff_imag_real = data_imag - data_real
    
    real_out = approx_0_7071(sum_real_imag)
    imag_out = approx_0_7071(diff_imag_real)
    
    return real_out, imag_out



def twiddle_multiply_w8_2(real, imag):
    """
    Multiply complex number by -j.
    
    (a + jb) * (-j) = b - ja
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    return (imag, -real)


def twiddle_multiply_w8_3(real, imag):
    """
    Multiply by W_8^3 = e^(-j*3*pi/4) = cos(3*pi/4) - j*sin(3*pi/4) = -0.7071 - j*0.7071
    
    (a + jb) * (-0.7071 - j*0.7071) = 0.7071*(b - a) - j*0.7071*(a + b)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    diff_b_a = imag - real
    sum_a_b = real + imag
    
    new_real = approx_0_7071(diff_b_a)
    new_imag = -approx_0_7071(sum_a_b)
    
    return new_real, new_imag


def fft_8point_hardware(input_i_binary_list, input_q_binary_list):
    """
    8-Point FFT Hardware Simulation.
    
    Implements the 3-stage Radix-2 DIT FFT with:
    - Bit-reversed input ordering
    - Hardware-style twiddle factor approximation
    - (Not implemented) Proper bit-width expansion at each stage
    
    Parameters:
    -----------
    input_i_binary_list : list of str
        List of 8 binary strings, each 14-bit wide (two's complement).
        These are scaled by 2^13 from the original +/-1 range signal.
        
    input_q_binary_list : list of str
        List of 8 binary strings, each 14-bit wide (two's complement).
        These are scaled by 2^13 from the original +/-1 range signal.
        
    Returns:
    --------
    tuple of lists : (real_out_list, imag_out_list)
        Each list contains 8 integers representing the real and imaginary parts
    """
    if len(input_i_binary_list) != 8 or len(input_q_binary_list) != 8:
        raise ValueError("Input must contain exactly 8 samples")
    
    # Convert binary strings to integers
    x_real = [bin_to_int(b) for b in input_i_binary_list]
    x_imag = [bin_to_int(b) for b in input_q_binary_list]
    
    # Initialize real and imaginary parts (inputs are real-only)
    #stage1
    stage1_real=[0]*8
    stage1_imag=[0]*8

    stage1_real[0] = x_real[0] + x_real[4]   
    stage1_imag[0] = x_imag[0] + x_imag[4]

    stage1_real[1] = x_real[0] - x_real[4] 
    stage1_imag[1] = x_imag[0] - x_imag[4]

    stage1_real[2] = x_real[2] + x_real[6]
    stage1_imag[2] = x_imag[2] + x_imag[6]

    stage1_real[3] = x_real[2] - x_real[6]
    stage1_imag[3] = x_imag[2] - x_imag[6]

    stage1_real[4] = x_real[1] + x_real[5]
    stage1_imag[4] = x_imag[1] + x_imag[5]

    stage1_real[5] = x_real[1] - x_real[5]
    stage1_imag[5] = x_imag[1] - x_imag[5]

    stage1_real[6] = x_real[3] + x_real[7]
    stage1_imag[6] = x_imag[3] + x_imag[7]

    stage1_real[7] = x_real[3] - x_real[7]
    stage1_imag[7] = x_imag[3] - x_imag[7]

    #multiplication by twiddle factors
    stage1_real[3], stage1_imag[3] = twiddle_multiply_w8_2(stage1_real[3], stage1_imag[3])
    stage1_real[7], stage1_imag[7] = twiddle_multiply_w8_2(stage1_real[7], stage1_imag[7])


    # Stage 2
    stage2_real=[0]*8
    stage2_imag=[0]*8

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

    #multiplication by twiddle factors
    stage2_real[5], stage2_imag[5] = twiddle_multiply_w8_1(stage2_real[5], stage2_imag[5])
    stage2_real[6], stage2_imag[6] = twiddle_multiply_w8_2(stage2_real[6], stage2_imag[6])
    stage2_real[7], stage2_imag[7] = twiddle_multiply_w8_3(stage2_real[7], stage2_imag[7])

    # Stage 3
    stage3_real=[0]*8
    stage3_imag=[0]*8

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

    # Output in natural order
    return stage3_real, stage3_imag


def fft_compare_demo(seed=0, scale=4096, tolerance_pct=0, bit_width=14):
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
    i_vals = rng.integers(-scale, scale + 1, size=8)
    q_vals = rng.integers(-scale, scale + 1, size=8)

    i_bin = [int_to_bin(int(v), bit_width) for v in i_vals]
    q_bin = [int_to_bin(int(v), bit_width) for v in q_vals]

    hw_real, hw_imag = fft_8point_hardware(i_bin, q_bin)
    ref = np.fft.fft(i_vals + 1j * q_vals)

    header = f"{'Bin':>3} {'HW Real':>9} {'HW Imag':>9} {'NP Real':>11} {'NP Imag':>11} {'|err|':>9} {'rel %':>7} {'Status':>7}"
    print("\n=== FFT Comparison ===")
    print(f"Seed: {seed} | Scale: {scale} | Tolerance: {tolerance_pct}%")
    print(f"I samples: {i_vals.tolist()}")
    print(f"Q samples: {q_vals.tolist()}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for idx in range(8):
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
        print("FFT hardware simulation test PASSED.")
    else:
        print("FFT hardware simulation test FAILED.")




