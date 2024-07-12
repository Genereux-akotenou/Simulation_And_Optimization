
%matplotlib inline
from matplotlib import pyplot as plt
from numba import vectorize, cuda
import numpy as np
import math

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def make_pulses(i, period, amplitude):
    return max(math.sin(i / period) - 0.3, 0.0) * amplitude

@cuda.jit
def add_ufunc(pulses, noise, out):
    idx = cuda.grid(1)
    if idx < len(pulses):
        out[idx] = pulses[idx] + noise[idx]

n = 100000
noise = (np.random.normal(size=n) * 3).astype(np.float32)
t = np.arange(n, dtype=np.float32)
period = n / 23

# Allocate device memory
d_pulses = cuda.to_device(np.zeros(n, dtype=np.float32))
d_waveform = cuda.to_device(np.zeros(n, dtype=np.float32))

# Call make_pulses with device memory
make_pulses[n, 1](t, period, 100.0, d_pulses)

# Call add_ufunc with device memory
threadsperblock = 256
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
add_ufunc[blockspergrid, threadsperblock](d_pulses, noise, d_waveform)

# Copy result back to host
waveform = d_waveform.copy_to_host()


# pulses = make_pulses(t, period, 100.0)
# waveform = add_ufunc(pulses, noise)
plt.plot(waveform)
