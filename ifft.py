"""
8-Point IFFT Hardware Simulation

This module implements an 8-point Radix-2 Decimation-in-Time (DIT) IFFT
that simulates the hardware architecture, mirroring the FFT implementation.

Key features:
- Bit-reversed input ordering
- 3-stage pipeline (14->15->16->17 bits)
- Hardware-style twiddle factor approximation using shift-and-add
- Conjugate twiddle factors compared to FFT (positive exponents)
- Scaling by 1/8 at the output

Mathematical Relationship:
IFFT(X[k]) = (1/N) * conjugate(FFT(conjugate(X[k])))
Where N is the number of points (8 in this case)
"""

import os

import numpy as np
from input_gen import int_to_bin
from fft import bin_to_int, half_round_up, approx_0_7071


def twiddle_multiply_w8_minus1(data_real, data_imag):
    """
    Multiply by W_8^(-1) = e^(j*2*pi/8) = cos(pi/4) + j*sin(pi/4) = 0.7071 + j*0.7071
    
    This is the conjugate of W_8^1 used in FFT.
    (a + jb) * (0.7071 + j*0.7071) = 0.7071*(a - b) + j*0.7071*(a + b)
    
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
    diff_real_imag = data_real - data_imag
    sum_real_imag = data_real + data_imag
    
    real_out = approx_0_7071(diff_real_imag)
    imag_out = approx_0_7071(sum_real_imag)
    
    return real_out, imag_out


def twiddle_multiply_w8_minus2(real, imag):
    """
    Multiply complex number by +j.
    
    This is the conjugate of W_8^2 = -j used in FFT.
    (a + jb) * (j) = -b + ja
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    return (-imag, real)


def twiddle_multiply_w8_minus3(real, imag):
    """
    Multiply by W_8^(-3) = e^(j*3*pi/4) = cos(3*pi/4) + j*sin(3*pi/4) = -0.7071 + j*0.7071
    
    This is the conjugate of W_8^3 used in FFT.
    (a + jb) * (-0.7071 + j*0.7071) = 0.7071*(-a - b) + j*0.7071*(a - b)
    
    Parameters:
    -----------
    real, imag : int
        Real and imaginary parts
        
    Returns:
    --------
    tuple : (new_real, new_imag)
    """
    neg_sum = -(real + imag)
    diff_a_b = real - imag
    
    new_real = approx_0_7071(neg_sum)
    new_imag = approx_0_7071(diff_a_b)
    
    return new_real, new_imag


def ifft_8point_hardware(input_i_binary_list, input_q_binary_list):
    """
    8-Point IFFT Hardware Simulation.
    
    Implements the 3-stage Radix-2 DIT IFFT with:
    - Bit-reversed input ordering
    - Hardware-style conjugate twiddle factor approximation
    - Scaling by 1/8 at the output (right shift by 3 bits)
    - (Not implemented) Proper bit-width expansion at each stage
    
    The IFFT is implemented using the relationship:
    IFFT = (1/N) * conjugate(FFT(conjugate(input)))
    But structured to match the FFT's 3-stage architecture with conjugate twiddles.
    
    Parameters:
    -----------
    input_i_binary_list : list of str
        List of 8 binary strings, each 14-bit wide (two's complement).
        These represent the real part of the frequency domain data.
        
    input_q_binary_list : list of str
        List of 8 binary strings, each 14-bit wide (two's complement).
        These represent the imaginary part of the frequency domain data.
        
    Returns:
    --------
    tuple of lists : (real_out_list, imag_out_list)
        Each list contains 8 integers representing the real and imaginary parts
        of the time-domain signal after IFFT
    """
    if len(input_i_binary_list) != 8 or len(input_q_binary_list) != 8:
        raise ValueError("Input must contain exactly 8 samples")
    
    # Convert binary strings to integers
    x_real = [bin_to_int(b) for b in input_i_binary_list]
    x_imag = [bin_to_int(b) for b in input_q_binary_list]
    
    # Stage 1: First butterfly operations with bit-reversed input order
    # Similar to FFT but with conjugate twiddle factors
    stage1_real = [0] * 8
    stage1_imag = [0] * 8

    # Butterfly operations for stage 1
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

    # Multiply by conjugate twiddle factors (W_8^(-2) = +j instead of -j)
    stage1_real[3], stage1_imag[3] = twiddle_multiply_w8_minus2(stage1_real[3], stage1_imag[3])
    stage1_real[7], stage1_imag[7] = twiddle_multiply_w8_minus2(stage1_real[7], stage1_imag[7])


    # Stage 2: Second butterfly stage
    stage2_real = [0] * 8
    stage2_imag = [0] * 8

    # Butterfly operations for stage 2
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

    # Multiply by conjugate twiddle factors
    stage2_real[5], stage2_imag[5] = twiddle_multiply_w8_minus1(stage2_real[5], stage2_imag[5])
    stage2_real[6], stage2_imag[6] = twiddle_multiply_w8_minus2(stage2_real[6], stage2_imag[6])
    stage2_real[7], stage2_imag[7] = twiddle_multiply_w8_minus3(stage2_real[7], stage2_imag[7])

    # Stage 3: Final butterfly stage
    stage3_real = [0] * 8
    stage3_imag = [0] * 8

    # Butterfly operations for stage 3
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

    # Scale by 1/8 (right shift by 3 bits) for IFFT normalization
    # This is the key difference from FFT which doesn't scale
    output_real = [val >> 3 for val in stage3_real]
    output_imag = [val >> 3 for val in stage3_imag]

    # Output in natural order
    return output_real, output_imag


def ifft_compare_demo(seed=0, scale=4096, tolerance_pct=0, bit_width=14):
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
    from fft import fft_8point_hardware
    
    # Generate random frequency-domain data for IFFT testing
    rng = np.random.default_rng(seed)
    freq_real = rng.integers(-scale, scale + 1, size=8)
    freq_imag = rng.integers(-scale, scale + 1, size=8)

    # Convert to binary representation
    freq_real_bin = [int_to_bin(int(v), bit_width) for v in freq_real]
    freq_imag_bin = [int_to_bin(int(v), bit_width) for v in freq_imag]

    # Run hardware IFFT
    hw_real, hw_imag = ifft_8point_hardware(freq_real_bin, freq_imag_bin)
    
    # Run NumPy IFFT for reference
    freq_complex = freq_real + 1j * freq_imag
    ref = np.fft.ifft(freq_complex)

    # Display results
    header = f"{'Bin':>3} {'HW Real':>9} {'HW Imag':>9} {'NP Real':>11} {'NP Imag':>11} {'|err|':>9} {'rel %':>7} {'Status':>7}"
    print("\n=== IFFT Comparison ===")
    print(f"Seed: {seed} | Scale: {scale} | Tolerance: {tolerance_pct}%")
    print(f"Frequency Real: {freq_real.tolist()}")
    print(f"Frequency Imag: {freq_imag.tolist()}")
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
    print(f"IFFT Result: {'PASSED' if all_pass else 'FAILED'}")
    
    # Test round-trip: IFFT(FFT(x)) should equal x
    print("\n=== Round-Trip Test: IFFT(FFT(x)) ===")
    
    # Generate time-domain signal
    time_real = rng.integers(-scale, scale + 1, size=8)
    time_imag = rng.integers(-scale, scale + 1, size=8)
    time_real_bin = [int_to_bin(int(v), bit_width) for v in time_real]
    time_imag_bin = [int_to_bin(int(v), bit_width) for v in time_imag]
    
    # FFT then IFFT
    fft_real, fft_imag = fft_8point_hardware(time_real_bin, time_imag_bin)
    fft_real_bin = [int_to_bin(int(v), 17) for v in fft_real]
    fft_imag_bin = [int_to_bin(int(v), 17) for v in fft_imag]
    
    ifft_real, ifft_imag = ifft_8point_hardware(fft_real_bin, fft_imag_bin)
    
    print(f"Original Time: Real={time_real.tolist()}")
    print(f"               Imag={time_imag.tolist()}")
    print(f"After IFFT(FFT): Real={ifft_real}")
    print(f"                 Imag={ifft_imag}")
    
    # Calculate round-trip error
    round_trip_pass = True
    max_error = 0
    for idx in range(8):
        err_real = abs(time_real[idx] - ifft_real[idx])
        err_imag = abs(time_imag[idx] - ifft_imag[idx])
        max_error = max(max_error, err_real, err_imag)
        if err_real > 5 or err_imag > 5:  # Allow small rounding errors
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
        print("\nIFFT hardware simulation test PASSED.")
    else:
        print("\nIFFT hardware simulation test FAILED.")
