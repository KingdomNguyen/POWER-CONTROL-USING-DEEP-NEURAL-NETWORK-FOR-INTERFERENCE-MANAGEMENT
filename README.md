# POWER-CONTROL-USING-DEEP-NEURAL-NETWORK-FOR-INTERFERENCE-MANAGEMENT

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

Weighted minimum mean-square error (WMMSE) is a popular and traditional algorithm in wireless communication but it is computational omplexity because of large cells and users. The method used in this project is based on approach from the paper "*Learning to Optimize: Training Deep Neural Networks for Interference Management*" (This is not orginal code, you may find their source in: [https://github.com/Haoran-S/TSP-DNN](https://github.com/Haoran-S/TSP-DNN))

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
   ```python
   def wmmse_multi_cell(self, H_flat, max_iter=50):
    H = H_flat.reshape(self.N, self.K)
    p = np.ones(self.total_users) * (self.P_max / self.total_users)  # Start with equal power
    for _ in range(max_iter):
        # Calculate interference (how much other users are disrupting this one)
        interference = np.sum(H[:, k]**2 * p_matrix[:, k]) - H[n, k]**2 * p_matrix[n, k]
        
        # Update power to maximize SINR (signal-to-interference ratio)
        sinr = (H**2 * p_matrix) / (1 + interference)
        p_new = (w * v**2) / np.sum(w * v**2) * self.P_max  # New power values
        p = np.minimum(self.P_max, p_new)  # Clip to P_max
    return p
   ```
This algorithm is slow because it requires many iterations per sample and omputes interference for every user pair
### Deep Neural Network (DNN)
The DNN learns to predict optimal power allocations directly from channel data, skipping iterative calculations.
```python
def build_dnn_model(self):
    input_layer = Input(shape=(self.N * self.K,))  # Input: Flattened channel gains
    x = Dense(256, activation='relu')(input_layer)  # Learn interference patterns
    x = BatchNormalization()(x)  # Stabilize training
    x = Dense(128, activation='relu')(x)  # Refine features
    x = Dense(64, activation='relu')(x)  
    output_layer = Dense(self.total_users, activation='sigmoid')(x)  # Output: Normalized powers
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')  # Train to minimize power errors
    return model
```
### Training
We use WMMSE for training process of DNN, and by this way, DNN will try to mimic WMMSE and it predicts power very fastly after training because it doesn't need iterations.
```python
 def prepare_data(self, num_samples):
        """Generate multi-cell dataset"""
        H = self.generate_multi_cell_channels(num_samples)
        P = np.array([self.wmmse_multi_cell(h) for h in H])
        # Scale data
        self.scaler.fit(H)
        H_scaled = self.scaler.transform(H)
        P_scaled = P / self.P_max  # Normalize power to [0,1]
        return H_scaled, P_scaled
```
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
- PyTorch
## Usage
