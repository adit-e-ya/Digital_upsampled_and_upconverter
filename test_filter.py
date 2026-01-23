"""
Test script to debug the elliptical IIR filter
"""
import numpy as np
from scipy import signal
from elliptical_iir import FixedElliptical

# Create test signal
n = 256
t = np.arange(n)
# Signal with low freq (should pass) and high freq (should be filtered)
sig = 1000 * np.sin(2*np.pi*0.05*t) + 500 * np.sin(2*np.pi*0.4*t)
sig_int = sig.astype(int).tolist()
imag = [0] * n

# Test fixed-point filter
print("Creating filter with wp=0.5, fs=4.0")
filt = FixedElliptical(bit_width=14, filter_bits=32, order=6, rp=0.1, rs=100.0, wp=0.5, fs=4.0)

print("Filter SOS (float):")
print(filt.sos_float)
print()
print("Filter SOS (fixed):")
print(filt.sos_fixed)
print()

# Process with fixed-point
out_real, out_imag = filt.process(sig_int, imag)

# Compare with scipy sosfilt (floating point reference)
out_scipy = signal.sosfilt(filt.sos_float, sig)

print("Input range:", min(sig_int), "to", max(sig_int))
print("Fixed output range:", min(out_real), "to", max(out_real))
print("Scipy output range:", min(out_scipy), "to", max(out_scipy))
print()
print("First 10 fixed outputs:", out_real[:10])
print("First 10 scipy outputs:", [int(x) for x in out_scipy[:10]])

# Check if scipy filter works correctly
print("\n--- Testing scipy filter directly ---")
sos_scipy = signal.ellip(N=6, rp=0.1, rs=100.0, Wn=0.5, btype='low', fs=4.0, output='sos')
out_scipy_direct = signal.sosfilt(sos_scipy, sig)
print("Scipy direct filter output range:", min(out_scipy_direct), "to", max(out_scipy_direct))
