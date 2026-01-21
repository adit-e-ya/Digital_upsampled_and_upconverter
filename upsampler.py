def zoh_interpolation(input_real, input_imag, factor=2):
    """
    Performs Zero-Order Hold (ZOH) interpolation on input real and imaginary samples.
    
    Parameters
    ----------
    input_real : list of int
        List of integers representing the real samples.
    input_imag : list of int
        List of integers representing the imaginary samples.
    factor : int
        Interpolation factor (default is 2).
    
    Returns
    -------
    tuple of list
        (output_real, output_imag) - Two lists of integers representing the
        upsampled real and imaginary samples.
    """
    output_real = []
    output_imag = []

    for r, im in zip(input_real, input_imag):
        output_real.extend([r] * factor)
        output_imag.extend([im] * factor)

    return output_real, output_imag

def linear_interpolation(input_real, input_imag):
    """
    Performs Linear interpolation on input real and imaginary samples.
    
    Parameters
    ----------
    input_real : list of int
        List of integers representing the real samples.
    input_imag : list of int
        List of integers representing the imaginary samples.
    
    
    Returns
    -------
    tuple of list
        (output_real, output_imag) - Two lists of integers representing the
        upsampled by 2, real and imaginary samples.
    """
    output_real = []
    output_imag = []

    for i in range(len(input_real) - 1):
        r_start = input_real[i]
        r_end = input_real[i + 1]
        im_start = input_imag[i]
        im_end = input_imag[i + 1]
        output_real.append(r_start)
        output_imag.append(im_start)
        r_interp = (r_start + r_end) // 2
        im_interp = (im_start + im_end) // 2
        output_real.append(r_interp)
        output_imag.append(im_interp)

    return output_real, output_imag
