# Script 3: Gaussian-Tuned Soft-Decision BER Simulation
# Simulates error correction for BPSK using soft-decision decoding, as
# described in Section 5, with results reported in Section 6.
import numpy as np

# Parameters (Section 6.1)
num_trials = 10000
k = 3
sigma = np.sqrt(0.25)
outlier_prob = 0.1
outlier_amplitude = 5

# Initialize error counters for error correction
errors_correction_awgn = 0
errors_correction_impulsive = 0

np.random.seed(42)
for _ in range(num_trials):
    # Generate true symbol: vec{x} = +1 or -1
    x_true = np.random.choice([1, -1])

    # Generate complex noise for k=3 transmissions
    noise_awgn = np.random.normal(0, sigma, k) + 1j * np.random.normal(0, sigma, k)
    noise_impulsive = np.random.normal(0, sigma, k) + 1j * np.random.normal(0, sigma, k)
    for i in range(k):
        if np.random.random() < outlier_prob:
            noise_impulsive[i] = np.random.choice([outlier_amplitude, -outlier_amplitude])

    # Received signals: vec{y}_i = vec{x} + n_i
    y_awgn = x_true + noise_awgn
    y_impulsive = x_true + noise_impulsive

    # Extract real components: y_{i,R} = Re(vec{y}_i)
    y_R_awgn = np.real(y_awgn)
    y_R_impulsive = np.real(y_impulsive)

    # Compute weights: w_i = y_{i,R}^2 / sigma^2 (Section 6)
    weights_awgn = y_R_awgn**2 / (sigma**2)
    weights_impulsive = y_R_impulsive**2 / (sigma**2)

    # Weighted voting
    vote_plus1_awgn = np.sum(weights_awgn[y_R_awgn > 0])
    vote_minus1_awgn = np.sum(weights_awgn[y_R_awgn < 0])
    vote_plus1_impulsive = np.sum(weights_impulsive[y_R_impulsive > 0])
    vote_minus1_impulsive = np.sum(weights_impulsive[y_R_impulsive < 0])

    x_decoded_awgn = 1 if vote_plus1_awgn >= vote_minus1_awgn else -1
    x_decoded_impulsive = 1 if vote_plus1_impulsive >= vote_minus1_impulsive else -1

    # Count errors
    if x_decoded_awgn != x_true:
        errors_correction_awgn += 1
    if x_decoded_impulsive != x_true:
        errors_correction_impulsive += 1

# Compute BER (Section 6)
ber_correction_awgn = errors_correction_awgn / num_trials * 100  # Approx. 2.49%
ber_correction_impulsive = errors_correction_impulsive / num_trials * 100  # Approx. 12.52%
print(f"Script 3 - Soft-Decision: Correction AWGN BER = {ber_correction_awgn:.2f}%, "
      f"Correction Impulsive BER = {ber_correction_impulsive:.2f}%")
