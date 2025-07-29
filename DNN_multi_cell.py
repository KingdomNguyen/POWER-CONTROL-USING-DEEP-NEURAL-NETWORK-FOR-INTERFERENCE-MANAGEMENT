import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
class UnsupervisedWSROptimizer:
    def __init__(self, N_cells, K_users_per_cell, P_max, user_weights=None):
        """
        Initialize the WSR optimizer

        Parameters:
        - N_cells: Number of cells
        - K_users_per_cell: Users per cell
        - P_max: Maximum power per user
        - user_weights: Optional weights for each user (w_k in the paper)
        """
        self.N = N_cells
        self.K = K_users_per_cell
        self.P_max = P_max
        self.total_users = N_cells * K_users_per_cell
        self.user_weights = user_weights if user_weights is not None else np.ones(self.total_users)
        self.model = self._build_model()
        self.scaler = MinMaxScaler(feature_range=(0.1, 1))  # Avoid zeros
        self.wmmse_times = []
        self.dnn_times = []
    def _build_model(self):
        """Build the DNN model with custom WSR loss function"""
        input_layer = Input(shape=(self.total_users,))
        # Network architecture
        x = Dense(512, activation='relu', kernel_initializer='he_normal')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        output_layer = Dense(self.total_users, activation='sigmoid')(x)  # Power in [0,1]
        model = Model(inputs=input_layer, outputs=output_layer)
        # Custom WSR loss function with proper type casting
        def wsr_loss(y_true, y_pred):
            # y_true contains the channel information (H)
            # y_pred contains the power allocations
            H = tf.cast(tf.reshape(y_true, [-1, self.N, self.K]), tf.float32)
            P = tf.cast(tf.reshape(y_pred * self.P_max, [-1, self.N, self.K]), tf.float32)
            weights = tf.cast(tf.reshape(self.user_weights, [1, self.N, self.K]), tf.float32)
            # Compute SINR
            signal = tf.square(H) * P
            interference = tf.reduce_sum(tf.square(H) * P, axis=[1, 2], keepdims=True) - signal
            sinr = signal / (interference + 1e-3)  # Small constant for numerical stability
            # Compute weighted sum rate 
            wsr = tf.reduce_sum(
                weights * tf.math.log(1.0 + sinr) / tf.math.log(2.0),  # log2 conversion
                axis=[1, 2]
            )
            return -wsr  # Minimize negative WSR to maximize WSR
        model.compile(optimizer=Adam(learning_rate=3e-4), loss=wsr_loss)
        return model
    def compute_wsr(self, H, P, noise_power=1e-4):
        """
        Compute weighted sum rate for given channel and power allocation
        """
        H = np.array(H).reshape(-1, self.N, self.K)
        P = np.array(P).reshape(-1, self.N, self.K)
        batch_size = H.shape[0]
        sinr = np.zeros((batch_size, self.N, self.K))
        for n in range(self.N):
            for k in range(self.K):
                # Calculate interference from all other users
                interference = (H ** 2 * P).sum(axis=(1, 2)) - (H[:, n, k] ** 2 * P[:, n, k])
                sinr[:, n, k] = (H[:, n, k] ** 2 * P[:, n, k]) / (interference + noise_power)
        # Apply user weights and compute sum rate
        weighted_rates = np.reshape(self.user_weights, (1, self.N, self.K)) * np.log2(1 + sinr)
        return np.sum(weighted_rates, axis=(1, 2))
    def generate_channels(self, num_samples):
        """
        Generate realistic multi-cell channel conditions with:
        - Path loss based on distance
        - Rayleigh fading
        - Cell layout with wrap-around
        """
        # Cell layout (circular arrangement)
        cell_locations = np.array([
            [np.cos(2 * np.pi * n / self.N), np.sin(2 * np.pi * n / self.N)]
            for n in range(self.N)
        ])
        H = np.zeros((num_samples, self.N, self.K))
        for sample in range(num_samples):
            for n in range(self.N):
                # Random user locations within 500m of their cell center
                user_locations = cell_locations[n] + 0.3 * np.random.rand(self.K, 2)
                # Calculate distances from all cells to all users
                distances = np.linalg.norm(
                    cell_locations.reshape(self.N, 1, 2) - user_locations.reshape(1, self.K, 2),
                    axis=2
                )
                # Path loss model (3.7 exponent)
                path_loss = 1 / (1 + distances ** 3.7)
                # Rayleigh fading component
                fading = np.sqrt(0.5) * (np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K))
                # Combined channel gain
                H[sample, n, :] = np.abs(path_loss * fading)[n, :]
        return H.reshape(num_samples, -1)  # Flatten to (samples, N*K)
    def wmmse_optimize(self, H_flat, max_iter=50):
        """
        WMMSE algorithm for power allocation (for benchmarking)
        Implements the algorithm described in the literature
        """
        H = H_flat.reshape(self.N, self.K)
        p = np.ones((self.N, self.K)) * (self.P_max / (self.N * self.K))  # Initial power
        eps = 1e-10  # Small constant
        noise_power = 1e-4  # Noise power
        sum_rate_prev = 0
        for _ in range(max_iter):
            # Compute SINR
            signal = H ** 2 * p
            interference = (H ** 2 * p).sum(axis=(0, 1)) - signal
            sinr = signal / (interference + noise_power + eps)
            # Compute weights and coefficients
            w = 1 / (1 + 1 / (sinr + eps))
            v = w * H / (H ** 2 * p + interference + noise_power + eps)
            # Update power with bisection search for Lagrange multiplier
            p_new = np.zeros((self.N, self.K))
            for n in range(self.N):
                # Bisection search to find optimal lambda
                low, high = 0, 1e6
                for _ in range(20):
                    lambda_n = (low + high) / 2
                    p_tmp = w[n, :] * v[n, :] / (H[n, :] ** 2 * (1 + lambda_n))
                    total_power = p_tmp.sum()
                    if total_power > self.P_max:
                        low = lambda_n
                    else:
                        high = lambda_n
                p_new[n, :] = w[n, :] * v[n, :] / (H[n, :] ** 2 * (1 + lambda_n))
            # Check convergence
            sum_rate_current = np.sum(np.reshape(self.user_weights, (self.N, self.K)) * np.log2(1 + sinr))
            if np.abs(sum_rate_current - sum_rate_prev) < 1e-5:
                break
            sum_rate_prev = sum_rate_current
            p = p_new
        return p.flatten()
    def train(self, num_samples=10000, epochs=50, batch_size=1024):
        """Train the model using unsupervised learning"""
        # Generate training data
        H_train = self.generate_channels(num_samples)
        H_train_scaled = self.scaler.fit_transform(H_train)
        # Convert to float32 for TensorFlow compatibility
        H_train_scaled = H_train_scaled.astype(np.float32)
        # For unsupervised learning, we pass the channel information as both input and "labels"
        # since we don't have true labels
        train_labels = H_train_scaled  # Using channel info as labels for loss calculation
        # Train with early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        print("Starting training...")
        history = self.model.fit(
            H_train_scaled, train_labels,  # Pass channel info as both input and labels
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training History (Negative WSR)')
        plt.xlabel('Epoch')
        plt.ylabel('Negative WSR')
        plt.legend()
        plt.grid()
        plt.show()
        return history
    def benchmark(self, num_test_samples=1000):
        """Compare DNN performance against WMMSE"""
        # Generate test data
        H_test = self.generate_channels(num_test_samples)
        H_test_scaled = self.scaler.transform(H_test).astype(np.float32)
        # DNN predictions
        start_time = time.time()
        P_dnn = self.model.predict(H_test_scaled, batch_size=1024) * self.P_max
        dnn_time = (time.time() - start_time) * 1000 / num_test_samples  # ms per sample
        self.dnn_times.append(dnn_time)
        # WMMSE calculations
        wmmse_times = []
        P_wmmse = []
        for h in H_test:
            start = time.time()
            p = self.wmmse_optimize(h)
            wmmse_times.append(time.time() - start)
            P_wmmse.append(p)
        avg_wmmse_time = np.mean(wmmse_times) * 1000  # ms per sample
        self.wmmse_times.append(avg_wmmse_time)
        P_wmmse = np.array(P_wmmse)
        # Compute sum rates
        wsr_wmmse = self.compute_wsr(H_test, P_wmmse)
        wsr_dnn = self.compute_wsr(H_test, P_dnn)
        # Print results
        print("\n=== Performance Comparison ===")
        print(f"WMMSE Average Time: {avg_wmmse_time:.4f} ms/sample")
        print(f"DNN Average Time: {dnn_time:.4f} ms/sample")
        print(f"Speedup Factor: {avg_wmmse_time / dnn_time:.1f}x")
        print("\n=== Sum Rate Comparison ===")
        print(f"WMMSE Average WSR: {np.mean(wsr_wmmse):.4f} bps/Hz")
        print(f"DNN Average WSR: {np.mean(wsr_dnn):.4f} bps/Hz")
        print(f"WSR Ratio (DNN/WMMSE): {np.mean(wsr_dnn) / np.mean(wsr_wmmse):.4f}")
        return {
            'wsr_wmmse': wsr_wmmse,
            'wsr_dnn': wsr_dnn,
            'time_wmmse': avg_wmmse_time,
            'time_dnn': dnn_time
        }
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    # Configuration
    N_cells = 7  # Number of transmitters
    K_users = 28  # Users per cell
    P_max = 10  # Maximum power per user
    num_samples = 10000  # Training samples
    # Create random user weights
    user_weights = np.random.uniform(0.5, 2.0, N_cells * K_users)
    # Initialize and train
    optimizer = UnsupervisedWSROptimizer(N_cells, K_users, P_max, user_weights)
    optimizer.train(num_samples=num_samples, epochs=150)
    # Benchmark against WMMSE
    results = optimizer.benchmark(num_test_samples=1000)