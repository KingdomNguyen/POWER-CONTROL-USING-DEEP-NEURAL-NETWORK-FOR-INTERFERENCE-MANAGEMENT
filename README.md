# Deep Learning-Based Power Control for Multi-Cell Interference Management in Wireless Networks

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)



*Fast DNN model for power control in a multi-cell wireless network*

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
## Problem Statement

This project addresses the distributed power control problem in a multi-cell wireless network:
- K users transmit simultaneously across N cells, creating interference.
  
- Each user's transmission power must be optimized to maximize the weighted sum-rate (network throughput) while:
  + Respecting maximum power constraints (0 ≤ p_k ≤ P_max).
  + Managing inter-cell interference (signals from users in other cells degrade performance).

A multi-cell interference network models a wireless system where: Multiple base stations (cells) serve users in overlapping geographic areas. Users in one cell cause interference to users in neighboring cells due to shared frequency bands. For example: 5G networks, where dense cell deployments are common. Inter-cell interference reduces signal quality and system capacity and so it's critical for managing it in order to ensure High data rates (e.g., streaming, VR) or Network reliability (e.g., dropped calls).

Traditional approaches like Weighted Minimum Mean-Square Error (WMMSE) face computational complexity limitations in large networks. The method used in this project is based on approach from the paper "*Towards Optimal Power Control via Ensembling Deep Neural Networks*" (This is not their orginal code). This method use an unsupervised deep learning. The neural network directly optimizes power allocations to maximize sum-rate, using the negative weighted sum-rate as its loss function. This will be faster and have higher sum-rate than WMMSE.

## Methodology
### Problem formulation
In a wireless network with multiple cells (base stations) and users, we need to assign transmit powers to all users to:
- Maximize total network speed (sum-rate).
- Minimize interference between cells.
- Respect each user's maximum power limit (P_max).
### Weighted minimum mean-square error (WMMSE)
WMMSE is an iterative algorithm that:
1. Calculates interference for each user.
2. Updates powers to balance signal strength and interference.

At its core, WMMSE alternates between three key steps:
1. SINR Calculation:

   WMMSE then calculates adaptive weights and precoding coefficients:
   ```python
   signal = H ** 2 * p
   interference = (H ** 2 * p).sum(axis=(0,1)) - signal
   sinr = signal / (interference + noise_power)
   ```
   This measures how much each user's signal is affected by interference from all other users in neighboring cells.
2. Weight and Precoder Updates
   WMMSE then calculates adaptive weights and precoding coefficients:
   ```python
   w = 1 / (1 + 1/(sinr + eps))  # User weights
   v = w * H / (H**2 * p + interference + noise_power)  # Precoder coefficients
   ```
   These values automatically prioritize users with poor channel conditions.

3. Power Allocation with Bisection
   For each cell, the code performs bisection search to find optimal power levels that satisfy maximum power constraints:
   ```python
   for n in range(N_cells):
    low, high = 0, 1e6
    for _ in range(20):  # Bisection iterations
        lambda_n = (low + high)/2
        p_tmp = w[n,:] * v[n,:] / (H[n,:]**2 * (1 + lambda_n))
        if p_tmp.sum() > P_max: low = lambda_n
        else: high = lambda_n
    p_new[n,:] = w[n,:] * v[n,:] / (H[n,:]**2 * (1 + lambda_n))
   ```
### Deep Neural Network (DNN)
The DNN approach replaces traditional iterative optimization with a neural network that learns optimal power control directly from channel conditions. Key aspects of the implementation:
1. Network architecture
```python
input_layer = Input(shape=(N*K,))  # Flattened channel state
x = Dense(512, activation='swish')(input_layer)  # Hidden layers
x = Dense(256, activation='swish')(x)
output_layer = Dense(N*K, activation='sigmoid')(x)  # Normalized power allocations
```
2. Unsupervised Training
```python
def wsr_loss(y_true, y_pred):
    P = y_pred * P_max  # Scaled power allocations
    signal = H**2 * P
    interference = tf.reduce_sum(H**2 * P) - signal
    sinr = signal / (interference + 1e-3)
    return -tf.reduce_sum(weights * tf.math.log(1 + sinr))  # Negative WSR
```
- Directly maximizes weighted sum-rate (WSR)
- No labeled data required (uses CSI as both input and "label")
- Automatically learns interference patterns
3. Multi-Cell Handling
- Input layer accepts flattened multi-cell channel matrix (N×K → N*K vector)
- Batch normalization layers help stabilize training across varying cell conditions
- Implicitly models interference through network weights
### Multi-cell channel modeling
Simulate realistic channel with:
- Path Loss: Signal weakens with distance (1 / (1 + d^3.7)).
- Rayleigh Fading: Random signal variations due to multipath.
```python
def generate_multi_cell_channels(self, num_samples):
    # Cell locations arranged in a circle
    cell_locations = np.array([[np.cos(2*np.pi*n/self.N), np.sin(2*np.pi*n/self.N)] 
                              for n in range(self.N)])
    
    # Random user locations near each cell center
    user_locations = cell_locations[n] + 0.5 * np.random.rand(self.K, 2)
    
    # Path loss + fading
    distances = np.linalg.norm(cell_locations - user_locations, axis=2)
    path_loss = 1 / (1 + distances**3.7)  # Stronger signal for closer users
    fading = np.random.randn() + 1j * np.random.randn()  # Random fading
    H = path_loss * fading  # Final channel gain
    return H.reshape(num_samples, -1)  # Flatten for DNN input
```
## Installation 
### Prerequisites
- Python 3.8+ (Tested with 3.10)
- CUDA 11.7+ (if using GPU)
- TensorFlow 2.10+
## Usage
You can run the code directly
## Results
Due to resource limitations, I will not consider extremely large multi-cell systems. If you want to experiment, vary the network size, parameters, and amount of data required for training.And note that I am running the cases with the same set of parameters, it will be faster to reduce the network size for small system cases.

Some results in different cases:

N_cells = 3 , K_users = 12:
```
=== Performance Comparison ===
WMMSE Average Time: 17.0807 ms/sample
DNN Average Time: 0.1432 ms/sample
Speedup Factor: 119.3x

=== Sum Rate Comparison ===
WMMSE Average WSR: 21.4463 bps/Hz
DNN Average WSR: 30.0804 bps/Hz
WSR Ratio (DNN/WMMSE): 1.4026
```
N_cells = 3 , K_users = 24:
```
=== Performance Comparison ===
WMMSE Average Time: 17.2732 ms/sample
DNN Average Time: 0.1405 ms/sample
Speedup Factor: 122.9x

=== Sum Rate Comparison ===
WMMSE Average WSR: 22.7033 bps/Hz
DNN Average WSR: 30.7606 bps/Hz
WSR Ratio (DNN/WMMSE): 1.3549
```
N_cells = 7 , K_users = 28:
```
=== Performance Comparison ===
WMMSE Average Time: 37.6159 ms/sample
DNN Average Time: 0.1366 ms/sample
Speedup Factor: 275.4x

=== Sum Rate Comparison ===
WMMSE Average WSR: 22.5882 bps/Hz
DNN Average WSR: 30.8001 bps/Hz
WSR Ratio (DNN/WMMSE): 1.3635
```

## Reference
[1] H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu, and N. D. Sidiropoulos,
“Learning to optimize: Training deep neural networks for interference
management,” IEEE Trans. Signal Process., vol. 66, no. 20, pp. 5438–
5453, Oct. 2018
[2]  F. Liang, C. Shen, W. Yu, and F. Wu, “Towards optimal power control
via ensembling deep neural networks,” IEEE Trans. Commun., vol. 68,
no. 3, pp. 1760–1776, Mar. 2020.
