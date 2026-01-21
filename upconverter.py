from fft import fft_8point_hardware
from ifft import ifft_8point_hardware

def upconverter(input_I_binary_list, input_Q_binary_list,shift):
    if len(input_I_binary_list)%8 !=0 or len(input_Q_binary_list)%8 !=0:
        raise ValueError("Input length must be multiple of 8 samples")
    fft_real=[]
    fft_imag=[]
    # Perform FFT on blocks of 8 samples of I and Q inputs
    for i in range(len(input_I_binary_list)//8):
        block_I = input_I_binary_list[i*8:(i+1)*8]
        block_Q = input_Q_binary_list[i*8:(i+1)*8]
        
        fft_real_8block, fft_imag_8block = fft_8point_hardware(block_I, block_Q)
        shift=shift%8
        fft_real_8block_shifted=fft_real_8block[shift:8]+fft_real_8block[0:shift]
        fft_imag_8block_shifted=fft_imag_8block[shift:8]+fft_imag_8block[0:shift]

        fft_real.extend(fft_real_8block_shifted)
        fft_imag.extend(fft_imag_8block_shifted)
    





    
