from fft import fft_8point_hardware
from ifft import ifft_8point_hardware
from input_gen import int_to_bin

def upconverter(input_I_binary_list, input_Q_binary_list, shift):
    """
    Upconvert I/Q samples using FFT-IFFT with spectral shifting.
    
    Processes input I/Q samples in blocks of 8, applies an 8-point FFT,
    performs spectral rotation (circular shift), and then applies an 8-point IFFT.
    
    Parameters
    ----------
    input_I_binary_list : list of str
        List of binary strings (14-bit, two's complement) representing
        in-phase (I) samples. Length must be a multiple of 8.
    input_Q_binary_list : list of str
        List of binary strings (14-bit, two's complement) representing
        quadrature (Q) samples. Length must be a multiple of 8.
    shift : int
        Spectral shift amount (in bins). Will be taken modulo 8 to determine
        the rotation within each 8-point block.
    
    Returns
    -------
    tuple of list
        (ifft_real, ifft_imag) - Two lists of integers representing the
        real and imaginary parts of the upconverted output after IFFT.
    
    Raises
    ------
    ValueError
        If input lengths are not multiples of 8.
    """
    if len(input_I_binary_list)%8 !=0 or len(input_Q_binary_list)%8 !=0:
        raise ValueError("Input length must be multiple of 8 samples")
    # fft_real=[]
    # fft_imag=[]

    ifft_real=[]
    ifft_imag=[]

    # Perform FFT on blocks of 8 samples of I and Q inputs
    for i in range(len(input_I_binary_list)//8):

        block_I = input_I_binary_list[i*8:(i+1)*8]
        block_Q = input_Q_binary_list[i*8:(i+1)*8]
        
        fft_real_8block, fft_imag_8block = fft_8point_hardware(block_I, block_Q)
        shift=shift%8
        fft_real_8block_shifted=fft_real_8block[shift:8]+fft_real_8block[0:shift]
        fft_imag_8block_shifted=fft_imag_8block[shift:8]+fft_imag_8block[0:shift]

        # fft_real.extend(fft_real_8block_shifted)
        # fft_imag.extend(fft_imag_8block_shifted)


        #converting to binary just for simulating hardware conditions of the ifft block
        fft_real_binary = [int_to_bin(val, total_bits=17) for val in fft_real_8block_shifted]
        fft_imag_binary = [int_to_bin(val, total_bits=17) for val in fft_imag_8block_shifted]
        
        ifft_real_8block, ifft_imag_8block = ifft_8point_hardware(fft_real_binary, fft_imag_binary)

        ifft_real.extend(ifft_real_8block)
        ifft_imag.extend(ifft_imag_8block)

    return ifft_real, ifft_imag






    
