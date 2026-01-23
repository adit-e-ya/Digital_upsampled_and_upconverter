"""
16-Point FFT-based Upconverter

This module implements a 16-point FFT-based upconverter using the same
logic and style as the 8-point upconverter.
"""

from fft_16 import fft_16point_hardware
from ifft_16 import ifft_16point_hardware
from input_gen import int_to_bin


def upconverter_16(input_I_binary_list, input_Q_binary_list, shift):
    """
    Upconvert I/Q samples using 16-point FFT-IFFT with spectral shifting.
    
    Processes input I/Q samples in blocks of 16, applies a 16-point FFT,
    performs spectral rotation (circular shift), and then applies a 16-point IFFT.
    
    Parameters
    ----------
    input_I_binary_list : list of str
        List of binary strings (14-bit, two's complement) representing
        in-phase (I) samples. Length must be a multiple of 16.
    input_Q_binary_list : list of str
        List of binary strings (14-bit, two's complement) representing
        quadrature (Q) samples. Length must be a multiple of 16.
    shift : int
        Spectral shift amount (in bins). Will be taken modulo 16 to determine
        the rotation within each 16-point block.
    
    Returns
    -------
    tuple of list
        (ifft_real, ifft_imag) - Two lists of integers representing the
        real and imaginary parts of the upconverted output after IFFT.
    
    Raises
    ------
    ValueError
        If input lengths are not multiples of 16.
    """
    if len(input_I_binary_list) % 16 != 0 or len(input_Q_binary_list) % 16 != 0:
        raise ValueError("Input length must be multiple of 16 samples")

    ifft_real = []
    ifft_imag = []

    # Perform FFT on blocks of 16 samples of I and Q inputs
    for i in range(len(input_I_binary_list) // 16):

        block_I = input_I_binary_list[i*16:(i+1)*16]
        block_Q = input_Q_binary_list[i*16:(i+1)*16]
        
        fft_real_16block, fft_imag_16block = fft_16point_hardware(block_I, block_Q)
        shift_mod = shift % 16
        fft_real_16block_shifted = fft_real_16block[shift_mod:16] + fft_real_16block[0:shift_mod]
        fft_imag_16block_shifted = fft_imag_16block[shift_mod:16] + fft_imag_16block[0:shift_mod]

        # Converting to binary just for simulating hardware conditions of the ifft block
        # 16-point FFT output is 18-bit (expanded from 14-bit input through 4 stages)
        fft_real_binary = [int_to_bin(val, total_bits=18) for val in fft_real_16block_shifted]
        fft_imag_binary = [int_to_bin(val, total_bits=18) for val in fft_imag_16block_shifted]
        
        ifft_real_16block, ifft_imag_16block = ifft_16point_hardware(fft_real_binary, fft_imag_binary)

        ifft_real.extend(ifft_real_16block)
        ifft_imag.extend(ifft_imag_16block)

    return ifft_real, ifft_imag
