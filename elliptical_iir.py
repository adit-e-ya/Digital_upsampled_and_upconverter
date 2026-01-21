import numpy as np
from scipy import signal


class FixedElliptical:
    """
    Fixed-point Elliptical IIR filter implementation using cascaded second-order sections (SOS).
    
    This filter is designed to work with binary signals from the upsampler output.
    It uses fixed-point arithmetic for hardware compatibility.
    
    Parameters
    ----------
    bit_width : int
        Bit width for input/output signals (default: 14)
    filter_bits : int
        Bit width for filter coefficients (default: 32)
    order : int
        Filter order (default: 6)
    rp : float
        Passband ripple in dB (default: 0.1)
    rs : float
        Stopband attenuation in dB (default: 100.0)
    wp : float
        Cutoff frequency (default: 0.5, normalized 0 < wp < fs/2)
    fs : float
        Normalized sampling frequency (default: 2.0)
    """
    
    def __init__(self, bit_width=14, filter_bits=32, order=6, rp=0.1, rs=100.0, 
                 wp=0.5, fs=2.0):
        self.bit_width = bit_width
        self.filter_bits = filter_bits
        self.order = order
        self.rp = rp
        self.rs = rs
        self.wp = wp
        self.fs = fs
        
        # Design the elliptical filter in SOS format
        self.sos_float = signal.ellip(
            N=order,
            rp=rp,
            rs=rs,
            Wn=wp,
            btype='low',
            analog=False,
            output='sos',
            fs=fs
        )
        
        # Convert to fixed-point coefficients
        self.sos_fixed = self._convert_to_fixed_point(self.sos_float)
        
        # Initialize state variables for each section
        # Each section has 2 state variables (for the 2 delays in biquad)
        self.num_sections = self.sos_fixed.shape[0]
        self._reset_states()
    
    def _reset_states(self):
        """Reset state variables for all sections."""
        # States for real part processing
        self.states_real = np.zeros((self.num_sections, 2), dtype=np.int64)
        # States for imaginary part processing
        self.states_imag = np.zeros((self.num_sections, 2), dtype=np.int64)
    
    def _convert_to_fixed_point(self, sos_float):
        """
        Convert floating-point SOS coefficients to fixed-point.
        
        Parameters
        ----------
        sos_float : ndarray
            Floating-point SOS coefficients from scipy
        
        Returns
        -------
        ndarray
            Fixed-point SOS coefficients (integer representation)
        """
        # Scale factor for filter coefficients
        scale_factor = 2 ** (self.filter_bits - 1) - 1
        
        # Convert to fixed-point
        sos_fixed = np.round(sos_float * scale_factor).astype(np.int64)
        
        return sos_fixed
    
    def _process_section(self, x, section_idx, states):
        """
        Process one sample through a single SOS section.
        
        Parameters
        ----------
        x : int
            Input sample (integer)
        section_idx : int
            Index of the SOS section
        states : ndarray
            State variables for this section [z1, z2]
        
        Returns
        -------
        int
            Output sample (integer)
        """
        # Get coefficients for this section: [b0, b1, b2, a0, a1, a2]
        b0, b1, b2, a0, a1, a2 = self.sos_fixed[section_idx]
        
        # Get states
        z1, z2 = states
        
        # Biquad filter implementation using Direct Form II
        # Scale factor for coefficients
        coeff_scale = 2 ** (self.filter_bits - 1) - 1
        
        # Compute output
        # y[n] = b0*x[n] + z1
        y = (b0 * x + z1 * coeff_scale) // coeff_scale
        
        # Update states
        # z1[n+1] = b1*x[n] - a1*y[n] + z2
        new_z1 = (b1 * x - a1 * y + z2 * coeff_scale) // coeff_scale
        
        # z2[n+1] = b2*x[n] - a2*y[n]
        new_z2 = (b2 * x - a2 * y) // coeff_scale
        
        # Update states
        states[0] = new_z1
        states[1] = new_z2
        
        return int(y)
    
    def process_sample(self, real_sample, imag_sample):
        """
        Process a single complex sample through all SOS sections.
        
        Parameters
        ----------
        real_sample : int or str
            Real part of the sample (integer or binary string)
        imag_sample : int or str
            Imaginary part of the sample (integer or binary string)
        
        Returns
        -------
        tuple
            (filtered_real, filtered_imag) - filtered output as integers
        """
        # Convert binary strings to integers if necessary
        if isinstance(real_sample, str):
            real_sample = self._binary_to_int(real_sample)
        if isinstance(imag_sample, str):
            imag_sample = self._binary_to_int(imag_sample)
        
        # Process real part through all sections
        real_out = real_sample
        for section_idx in range(self.num_sections):
            real_out = self._process_section(real_out, section_idx, 
                                            self.states_real[section_idx])
        
        # Process imaginary part through all sections
        imag_out = imag_sample
        for section_idx in range(self.num_sections):
            imag_out = self._process_section(imag_out, section_idx, 
                                            self.states_imag[section_idx])
        
        return real_out, imag_out
    
    def process(self, input_real, input_imag):
        """
        Process arrays of real and imaginary samples.
        
        Parameters
        ----------
        input_real : list of int or list of str
            List of real samples (integers or binary strings)
        input_imag : list of int or list of str
            List of imaginary samples (integers or binary strings)
        
        Returns
        -------
        tuple
            (output_real, output_imag) - Two lists of filtered integer samples
        """
        output_real = []
        output_imag = []
        
        for r, im in zip(input_real, input_imag):
            r_out, im_out = self.process_sample(r, im)
            output_real.append(r_out)
            output_imag.append(im_out)
        
        return output_real, output_imag
    
    def _binary_to_int(self, binary_str):
        """
        Convert binary string (two's complement) to integer.
        
        Parameters
        ----------
        binary_str : str
            Binary string representation
        
        Returns
        -------
        int
            Integer value
        """
        # Convert binary string to integer
        val = int(binary_str, 2)
        
        # Check if sign bit is set (MSB)
        if val >= 2 ** (len(binary_str) - 1):
            # Negative number in two's complement
            val -= 2 ** len(binary_str)
        
        return val
    
    def reset(self):
        """Reset all state variables to zero."""
        self._reset_states()
    
    def get_frequency_response(self, worN=512):
        """
        Get the frequency response of the filter.
        
        Parameters
        ----------
        worN : int
            Number of frequency points
        
        Returns
        -------
        tuple
            (frequencies, magnitude, phase) - Frequency response data
        """
        w, h = signal.sosfreqz(self.sos_float, worN=worN, fs=self.fs)
        magnitude = 20 * np.log10(np.abs(h) + 1e-10)
        phase = np.angle(h)
        
        return w, magnitude, phase


def elliptical_filter(input_real, input_imag, bit_width=14, filter_bits=32, 
                      order=6, rp=0.1, rs=100.0, wp=0.5, fs=2.0):
    """
    Convenience function to filter a signal using an elliptical IIR filter.
    
    Parameters
    ----------
    input_real : list
        Real part of input samples (integers or binary strings)
    input_imag : list
        Imaginary part of input samples (integers or binary strings)
    bit_width : int
        Bit width for input/output signals (default: 14)
    filter_bits : int
        Bit width for filter coefficients (default: 32)
    order : int
        Filter order (default: 6)
    rp : float
        Passband ripple in dB (default: 0.1)
    rs : float
        Stopband attenuation in dB (default: 100.0)
    wp : float
        Cutoff frequency (default: 0.5, normalized)
    fs : float
        Normalized sampling frequency (default: 2.0)
    
    Returns
    -------
    tuple
        (output_real, output_imag) - Filtered output as lists of integers
    """
    filter_obj = FixedElliptical(bit_width, filter_bits, order, rp, rs, wp, fs)
    return filter_obj.process(input_real, input_imag)
