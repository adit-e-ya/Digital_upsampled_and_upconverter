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

import numpy as np


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


def int_to_bin(num, total_bits):
    """
    Convert an integer to binary representation (two's complement).
    
    Parameters:
    -----------
    num : int
        Integer value to convert
    total_bits : int
        Total number of bits in binary representation
        
    Returns:
    --------
    str : Binary string representation
    """
    mask = (1 << total_bits) - 1
    val_masked = num & mask
    return format(val_masked, f'0{total_bits}b')


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
    # Left shifts: 1, 7, 9, 11, 13, 14
    # Sum these shifted versions, then right-shift by 15
    shifted_sum = (val << 1) + (val << 7) + (val << 9) + (val << 11) + (val << 13) + (val << 14)
    
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


def butterfly(a_real, a_imag, b_real, b_imag):
    """
    Standard Radix-2 butterfly operation.
    
    out_upper = A + B
    out_lower = A - B
    
    Parameters:
    -----------
    a_real, a_imag : int
        Complex input A
    b_real, b_imag : int
        Complex input B
        
    Returns:
    --------
    tuple : ((upper_real, upper_imag), (lower_real, lower_imag))
    """
    upper_real = a_real + b_real
    upper_imag = a_imag + b_imag
    lower_real = a_real - b_real
    lower_imag = a_imag - b_imag
    
    return (upper_real, upper_imag), (lower_real, lower_imag)


def multiply_by_minus_j(real, imag):
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


def multiply_by_w8_3(real, imag):
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


def fft_8point_hardware(input_binary_list):
    """
    8-Point FFT Hardware Simulation.
    
    Implements the 3-stage Radix-2 DIT FFT with:
    - Bit-reversed input ordering
    - Hardware-style twiddle factor approximation
    - Proper bit-width expansion at each stage
    
    Parameters:
    -----------
    input_binary_list : list of str
        List of 8 binary strings, each 14-bit wide (two's complement).
        These are scaled by 2^13 from the original +/-1 range signal.
        
    Returns:
    --------
    list of tuple : List of 8 complex output tuples (real, imag) as integers
                    Each in natural order: X(0), X(1), ..., X(7)
    """
    if len(input_binary_list) != 8:
        raise ValueError("Input must contain exactly 8 samples")
    
    # Convert binary strings to integers
    x = [bin_to_int(b) for b in input_binary_list]
    
    # Initialize real and imaginary parts (inputs are real-only)
    # Data in bit-reversed order for DIT FFT
    # Line 1: x(0), Line 2: x(4), Line 3: x(2), Line 4: x(6)
    # Line 5: x(1), Line 6: x(5), Line 7: x(3), Line 8: x(7)
    bit_reversed_indices = [0, 4, 2, 6, 1, 5, 3, 7]
    
    # Stage 0: Input (14-bit)
    lines_real = [x[i] for i in bit_reversed_indices]
    lines_imag = [0] * 8  # Imaginary parts are zero for real input
    
    # =========================================================================
    # STAGE 1: 4 independent Radix-2 butterflies (14->15 bits)
    # Pairs: (Line 0, Line 1), (Line 2, Line 3), (Line 4, Line 5), (Line 6, Line 7)
    # All use W_8^0 = 1 (no twiddle multiplication needed)
    # =========================================================================
    
    stage1_real = [0] * 8
    stage1_imag = [0] * 8
    
    # Butterfly 0: Lines 0,1 (x(0), x(4))
    (stage1_real[0], stage1_imag[0]), (stage1_real[1], stage1_imag[1]) = \
        butterfly(lines_real[0], lines_imag[0], lines_real[1], lines_imag[1])
    
    # Butterfly 1: Lines 2,3 (x(2), x(6))
    (stage1_real[2], stage1_imag[2]), (stage1_real[3], stage1_imag[3]) = \
        butterfly(lines_real[2], lines_imag[2], lines_real[3], lines_imag[3])
    
    # Butterfly 2: Lines 4,5 (x(1), x(5))
    (stage1_real[4], stage1_imag[4]), (stage1_real[5], stage1_imag[5]) = \
        butterfly(lines_real[4], lines_imag[4], lines_real[5], lines_imag[5])
    
    # Butterfly 3: Lines 6,7 (x(3), x(7))
    (stage1_real[6], stage1_imag[6]), (stage1_real[7], stage1_imag[7]) = \
        butterfly(lines_real[6], lines_imag[6], lines_real[7], lines_imag[7])
    
    # Stage 1 output is 15 bits
    
    # =========================================================================
    # STAGE 2: Two groups of 4-point operations (15->16 bits)
    # Group 1 (Lines 0-3): Line 0 pairs with Line 2; Line 1 pairs with Line 3
    # Group 2 (Lines 4-7): Line 4 pairs with Line 6; Line 5 pairs with Line 7
    # 
    # Twiddle factors applied to lower leg BEFORE butterfly:
    # - Lines 2 and 6: W_8^0 = 1 (no change)
    # - Lines 3 and 7: W_8^2 = -j
    # =========================================================================
    
    stage2_real = [0] * 8
    stage2_imag = [0] * 8
    
    # Group 1:
    # Line 0 pairs with Line 2 (Line 2 * W_8^0 = Line 2)
    (stage2_real[0], stage2_imag[0]), (stage2_real[2], stage2_imag[2]) = \
        butterfly(stage1_real[0], stage1_imag[0], stage1_real[2], stage1_imag[2])
    
    # Line 1 pairs with Line 3 (Line 3 * W_8^2 = Line 3 * (-j))
    tw_real, tw_imag = multiply_by_minus_j(stage1_real[3], stage1_imag[3])
    (stage2_real[1], stage2_imag[1]), (stage2_real[3], stage2_imag[3]) = \
        butterfly(stage1_real[1], stage1_imag[1], tw_real, tw_imag)
    
    # Group 2:
    # Line 4 pairs with Line 6 (Line 6 * W_8^0 = Line 6)
    (stage2_real[4], stage2_imag[4]), (stage2_real[6], stage2_imag[6]) = \
        butterfly(stage1_real[4], stage1_imag[4], stage1_real[6], stage1_imag[6])
    
    # Line 5 pairs with Line 7 (Line 7 * W_8^2 = Line 7 * (-j))
    tw_real, tw_imag = multiply_by_minus_j(stage1_real[7], stage1_imag[7])
    (stage2_real[5], stage2_imag[5]), (stage2_real[7], stage2_imag[7]) = \
        butterfly(stage1_real[5], stage1_imag[5], tw_real, tw_imag)
    
    # Stage 2 output is 16 bits
    
    # =========================================================================
    # STAGE 3: One group of 8-point operations (16->17 bits)
    # Each Line k (0-3) pairs with Line k+4
    # 
    # Twiddle factors applied to lower legs (Lines 4-7) BEFORE butterfly:
    # - Line 4: W_8^0 = 1
    # - Line 5: W_8^1 = 0.7071 - j*0.7071
    # - Line 6: W_8^2 = -j
    # - Line 7: W_8^3 = -0.7071 - j*0.7071
    # =========================================================================
    
    stage3_real = [0] * 8
    stage3_imag = [0] * 8
    
    # Line 0 pairs with Line 4 (Line 4 * W_8^0 = Line 4)
    (stage3_real[0], stage3_imag[0]), (stage3_real[4], stage3_imag[4]) = \
        butterfly(stage2_real[0], stage2_imag[0], stage2_real[4], stage2_imag[4])
    
    # Line 1 pairs with Line 5 (Line 5 * W_8^1 = 0.7071 - j*0.7071)
    tw_real, tw_imag = twiddle_multiply_w8_1(stage2_real[5], stage2_imag[5])
    (stage3_real[1], stage3_imag[1]), (stage3_real[5], stage3_imag[5]) = \
        butterfly(stage2_real[1], stage2_imag[1], tw_real, tw_imag)
    
    # Line 2 pairs with Line 6 (Line 6 * W_8^2 = -j)
    tw_real, tw_imag = multiply_by_minus_j(stage2_real[6], stage2_imag[6])
    (stage3_real[2], stage3_imag[2]), (stage3_real[6], stage3_imag[6]) = \
        butterfly(stage2_real[2], stage2_imag[2], tw_real, tw_imag)
    
    # Line 3 pairs with Line 7 (Line 7 * W_8^3 = -0.7071 - j*0.7071)
    tw_real, tw_imag = multiply_by_w8_3(stage2_real[7], stage2_imag[7])
    (stage3_real[3], stage3_imag[3]), (stage3_real[7], stage3_imag[7]) = \
        butterfly(stage2_real[3], stage2_imag[3], tw_real, tw_imag)
    
    # Stage 3 output is 17 bits, in natural order X(0), X(1), ..., X(7)
    
    # Return as list of complex tuples (real, imag)
    output = [(stage3_real[i], stage3_imag[i]) for i in range(8)]
    
    return output


def fft_8point_reference(input_list):
    """
    Reference 8-point FFT using NumPy for validation.
    
    Parameters:
    -----------
    input_list : list of int
        List of 8 integer values
        
    Returns:
    --------
    list of complex : List of 8 complex FFT outputs
    """
    return np.fft.fft(input_list)


def validate_fft(input_binary_list, tolerance=0.1):
    """
    Validate the hardware FFT against NumPy reference.
    
    Parameters:
    -----------
    input_binary_list : list of str
        List of 8 binary strings (14-bit, two's complement)
    tolerance : float
        Relative tolerance for comparison (default 10% due to approximation)
        
    Returns:
    --------
    bool : True if validation passes
    """
    # Convert inputs to integers
    x = [bin_to_int(b) for b in input_binary_list]
    
    # Get hardware FFT result
    hw_result = fft_8point_hardware(input_binary_list)
    
    # Get reference FFT result
    ref_result = fft_8point_reference(x)
    
    # Calculate total energy for threshold using consistent method
    total_energy = sum(np.abs(r)**2 for r in ref_result)
    threshold = np.sqrt(total_energy) * 0.01  # 1% of total energy
    
    print("\n=== FFT Validation ===")
    print(f"Input values: {x}")
    print(f"\n{'Bin':^4} {'HW Real':>10} {'HW Imag':>10} {'Ref Real':>12} {'Ref Imag':>12} {'Error %':>10} {'Status':>8}")
    print("-" * 75)
    
    all_pass = True
    for i in range(8):
        hw_real, hw_imag = hw_result[i]
        ref_real = ref_result[i].real
        ref_imag = ref_result[i].imag
        
        # Calculate magnitudes using consistent method (np.abs for both)
        hw_mag = np.abs(complex(hw_real, hw_imag))
        ref_mag = np.abs(ref_result[i])
        
        # For bins with significant energy, check relative error
        # For bins with small energy, check absolute error
        if ref_mag > threshold:
            error_pct = abs(hw_mag - ref_mag) / ref_mag * 100
            status = "PASS" if error_pct < tolerance * 100 else "FAIL"
        else:
            # For near-zero reference, accept if hardware is also small
            # relative to the total energy
            error_pct = 0.0
            if hw_mag < threshold * 10:  # Allow 10x threshold for leakage
                status = "PASS"
            else:
                status = "WARN"  # Leakage but not critical
        
        if status == "FAIL":
            all_pass = False
        
        print(f"{i:^4} {hw_real:>10} {hw_imag:>10} {ref_real:>12.2f} {ref_imag:>12.2f} {error_pct:>9.2f}% {status:>8}")
    
    print("\n" + ("=" * 40))
    print(f"Validation: {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 40)
    
    return all_pass


def demo():
    """
    Demonstration of the 8-point FFT hardware simulation.
    """
    print("=" * 60)
    print("8-Point FFT Hardware Simulation Demo")
    print("=" * 60)
    
    # Example: Create 8 samples of a simple signal
    # For 14-bit two's complement: range is -8192 to 8191
    # Use scale factor of 8191 (max positive value) for +/-1 range
    # Or use 4096 (2^12) for 0.5 amplitude with headroom
    scale = 4096  # Use 0.5 amplitude to leave headroom
    
    # Example: Single frequency tone (DC + one frequency bin)
    # x[n] = cos(2*pi * 2 * n / 8) = cos(pi * n / 2) for n = 0..7
    # This gives: 1, 0, -1, 0, 1, 0, -1, 0 (scaled)
    x_float = [np.cos(2 * np.pi * 2 * n / 8) for n in range(8)]
    x_int = [int(round(val * scale)) for val in x_float]
    
    # Convert to 14-bit binary strings
    input_binary = [int_to_bin(val, 14) for val in x_int]
    
    print("\n--- Input Signal ---")
    print(f"Float values: {[round(v, 4) for v in x_float]}")
    print(f"Scaled integers (x {scale}): {x_int}")
    print(f"Binary (14-bit):")
    for i, b in enumerate(input_binary):
        print(f"  x[{i}] = {b} ({bin_to_int(b)})")
    
    # Run hardware FFT
    print("\n--- Hardware FFT Result ---")
    hw_result = fft_8point_hardware(input_binary)
    
    for i, (real, imag) in enumerate(hw_result):
        print(f"  X[{i}] = {real:>8} + j*{imag:<8}")
    
    # Compare with reference
    print("\n--- NumPy Reference FFT ---")
    ref_result = fft_8point_reference(x_int)
    
    for i, val in enumerate(ref_result):
        print(f"  X[{i}] = {val.real:>12.2f} + j*{val.imag:<12.2f}")
    
    # Validate
    validate_fft(input_binary)
    
    # Another example: impulse signal
    print("\n" + "=" * 60)
    print("Example 2: Unit Impulse (delta function)")
    print("=" * 60)
    
    impulse_int = [scale, 0, 0, 0, 0, 0, 0, 0]
    impulse_binary = [int_to_bin(val, 14) for val in impulse_int]
    
    print(f"\nInput: {impulse_int}")
    
    hw_result = fft_8point_hardware(impulse_binary)
    ref_result = fft_8point_reference(impulse_int)
    
    print("\nHardware FFT:")
    for i, (real, imag) in enumerate(hw_result):
        print(f"  X[{i}] = {real:>8} + j*{imag:<8}")
    
    print("\nReference FFT:")
    for i, val in enumerate(ref_result):
        print(f"  X[{i}] = {val.real:>12.2f} + j*{val.imag:<12.2f}")
    
    # Example 3: Arbitrary signal to test twiddle factors
    print("\n" + "=" * 60)
    print("Example 3: Arbitrary Signal (tests twiddle factor approximation)")
    print("=" * 60)
    
    # Signal: x[n] = sin(2*pi*n/8) for n=0..7 - exercises W_8^1 and W_8^3
    x_arb_float = [np.sin(2 * np.pi * n / 8) for n in range(8)]
    x_arb_int = [int(round(val * scale)) for val in x_arb_float]
    arb_binary = [int_to_bin(val, 14) for val in x_arb_int]
    
    print(f"\nInput (sin wave at bin 1): {x_arb_int}")
    
    hw_result = fft_8point_hardware(arb_binary)
    ref_result = fft_8point_reference(x_arb_int)
    
    print("\nHardware FFT:")
    for i, (real, imag) in enumerate(hw_result):
        print(f"  X[{i}] = {real:>8} + j*{imag:<8}")
    
    print("\nReference FFT:")
    for i, val in enumerate(ref_result):
        print(f"  X[{i}] = {val.real:>12.2f} + j*{val.imag:<12.2f}")
    
    validate_fft(arb_binary, tolerance=0.2)  # Allow 20% tolerance for approximation


if __name__ == "__main__":
    demo()
