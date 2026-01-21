import numpy as np

def Sin_gen(bin_freq, nsam, nfft=1024):
    """
    generates array of sine wave with bin_freq cycles per nfft points(or samples)
    #nsam is the number of samples to generate
    #freq of the sine wave is sampling_rate * (bin_freq/ nfft)
    #multiple of these can be added to create more complex signals
    """
    sin_lut = np.sin(2 * np.pi * bin_freq * np.arange(1,nsam+1) / nfft)
    return sin_lut

def Cos_gen(bin_freq, nsam, nfft=1024):
    """similar to Sin_gen but generates cosine wave"""
    cos_lut = np.cos(2 * np.pi * bin_freq * np.arange(1,nsam+1) / nfft)
    return cos_lut


def int_to_bin(num, total_bits=14):
    """
    converts integer to binary representation (Two's Complement)
    total_bits: total number of bits in binary representation
    """
    # Create a mask of all 1s (e.g., for 14 bits: 0x3FFF)
    mask = (1 << total_bits) - 1
    
        
        # Apply mask. 
        # For -1: (-1 & 0x3FFF) becomes 0x3FFF (11...11)
        # For  1: ( 1 & 0x3FFF) remains 1
    val_masked = num & mask
    
    # Format as binary string
    binary = format(val_masked, '0{}b'.format(total_bits))
        
    return binary

def float_to_fixed_binary(input_array, total_bits=14, frac_bits=13):
    """
    converts floating point array to fixed point representation
    total_bits: total number of bits in fixed point representation
    frac_bits: number of bits for fractional part
    the remaining bit is for sign
    """
    scale_factor = 2 ** frac_bits
    max_val = 2 ** (total_bits - 1) - 1
    min_val = -2 ** (total_bits - 1)
    fixed_bin_arr = []

    integer_arr = np.round(input_array * scale_factor)
    clipped_array = np.clip(integer_arr, min_val, max_val)
    for num in clipped_array:
        fixed_bin_arr.append(int_to_bin(int(num), total_bits))

    return fixed_bin_arr

def input_gen(bin_freq, nsam, nfft=1024, total_bits=14, frac_bits=13):
    """
    generates fixed point binary representation of sine wave
    by multiplying each number with 2^frac_bits and converting to integer
    """

    #---add code here to generate other waveforms---
    
    sin_wave = Sin_gen(bin_freq, nsam, nfft)
    fixed_bin_sin = float_to_fixed_binary(sin_wave, total_bits, frac_bits)
    return fixed_bin_sin

print(input_gen(bin_freq=511, nsam=2000, nfft=1024, total_bits=14, frac_bits=13)[:100])