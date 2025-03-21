import scipy
import numpy as np


# for i in range(12000):
#     m = scipy.io.loadmat(f'lpca_out/ZINC/k8/idx_{i}.mat')
#     scipy.io.savemat(f'lpca_out/ZINC/k8_onlyU/idx_{i}.mat', { 'U': np.hstack((m['U'], np.transpose(m['V'])))})

for i in range(12000):
    m = scipy.io.loadmat(f'lpca_out/ZINC_fixed/k24/idx_{i}.mat')
    scipy.io.savemat(f'lpca_out/ZINC_fixed/k24_noV/idx_{i}.mat', { 'U': m['U']})