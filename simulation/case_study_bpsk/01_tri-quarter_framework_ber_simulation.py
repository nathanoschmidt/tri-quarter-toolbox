# Script 1: Tri-Quarter Framework BER Simulation
# Simulates noise filtering and error correction for BPSK using the
# Tri-Quarter framework as described in Sections 4 and 5, with results
# reported in Section 6.
# Note: The Tri-Quarter noise filtering uses the same decision rule as
# standard thresholding (decide +1 if y >= 0, else -1), and both methods
# produce identical BER results.
import numpy as np

# Parameters (Section 6.1)
num_trials = 100000  # Number of BPSK symbols
k = 3  # Transmissions per symbol for error correction
sigma = np.sqrt(0.25)  # Noise standard deviation (SNR = 6 dB, sigma^2 = 0.25)
outlier_prob = 0.1  # Probability of impulsive outliers
outlier_amplitude = 5  # Outlier amplitude for impulsive noise

# Initialize error counters for noise filtering (Tri-Quarter and Standard
# Thresholding) and error correction
errors_filtering_awgn_tri = 0  # Tri-Quarter filtering, AWGN
errors_filtering_impulsive_tri = 0  # Tri-Quarter filtering, Impulsive
errors_filtering_awgn_std = 0  # Standard thresholding, AWGN
errors_filtering_impulsive_std = 0  # Standard thresholding, Impulsive
errors_correction_awgn = 0
errors_correction_impulsive = 0

np.random.seed(42)  # For reproducibility
for _ in range(num_trials):
    # Generate true symbol: vec{x} = +1 or -1 (Section 2)
    x_true = np.random.choice([1, -1])

    # --- Noise Filtering Simulation (Section 4, k=1) ---
    # Generate real noise
    noise_awgn = np.random.normal(0, sigma, 1)
    noise_impulsive = np.random.normal(0, sigma, 1)
    if np.random.random() < outlier_prob:
        noise_impulsive[0] = np.random.choice([outlier_amplitude, -outlier_amplitude])

    # Received signal: y = x + n
    y_awgn = x_true + noise_awgn[0]
    y_impulsive = x_true + noise_impulsive[0]

    # Tri-Quarter decoding (Section 4, Eq. 4.1)
    # Decision rule: decide +1 if y >= 0, else -1 (same as standard thresholding)
    x_decoded_awgn_tri = 1 if y_awgn >= 0 else -1
    x_decoded_impulsive_tri = 1 if y_impulsive >= 0 else -1

    # Standard Thresholding decoding
    # Explicitly computed for clarity, uses the same decision rule: decide +1 if y >= 0, else -1
    x_decoded_awgn_std = 1 if y_awgn >= 0 else -1
    x_decoded_impulsive_std = 1 if y_impulsive >= 0 else -1

    # Count errors for Tri-Quarter filtering
    if x_decoded_awgn_tri != x_true:
        errors_filtering_awgn_tri += 1
    if x_decoded_impulsive_tri != x_true:
        errors_filtering_impulsive_tri += 1

    # Count errors for Standard Thresholding (identical to Tri-Quarter)
    if x_decoded_awgn_std != x_true:
        errors_filtering_awgn_std += 1
    if x_decoded_impulsive_std != x_true:
        errors_filtering_impulsive_std += 1

    # --- Error Correction Simulation (Section 5, k=3) ---
    # Generate real noise for k=3 transmissions
    noise_awgn = np.random.normal(0, sigma, k)
    noise_impulsive = np.random.normal(0, sigma, k)
    for i in range(k):
        if np.random.random() < outlier_prob:
            noise_impulsive[i] = np.random.choice([outlier_amplitude, -outlier_amplitude])

    # Received signals: y_i = x + n_i
    y_awgn = x_true + noise_awgn
    y_impulsive = x_true + noise_impulsive

    # Compute distances: d_i = |y_i - sign(y_i) * 1| (Section 5)
    distances_awgn = np.abs(y_awgn - np.sign(y_awgn) * 1)
    distances_impulsive = np.abs(y_impulsive - np.sign(y_impulsive) * 1)

    # Compute weights: w_i = 1 / (1 + d_i^2)
    weights_awgn = 1 / (1 + distances_awgn**2)
    weights_impulsive = 1 / (1 + distances_impulsive**2)

    # Weighted voting (Section 5, Eq. 5.1-5.3)
    vote_plus1_awgn = np.sum(weights_awgn[y_awgn > 0])
    vote_minus1_awgn = np.sum(weights_awgn[y_awgn < 0])
    vote_plus1_impulsive = np.sum(weights_impulsive[y_impulsive > 0])
    vote_minus1_impulsive = np.sum(weights_impulsive[y_impulsive < 0])

    x_decoded_awgn = 1 if vote_plus1_awgn >= vote_minus1_awgn else -1
    x_decoded_impulsive = 1 if vote_plus1_impulsive >= vote_minus1_impulsive else -1

    # Count errors for Tri-Quarter error correction
    if x_decoded_awgn != x_true:
        errors_correction_awgn += 1
    if x_decoded_impulsive != x_true:
        errors_correction_impulsive += 1

# Compute BER (Section 6)
ber_filtering_awgn_tri = errors_filtering_awgn_tri / num_trials * 100
ber_filtering_impulsive_tri = errors_filtering_impulsive_tri / num_trials * 100
ber_filtering_awgn_std = errors_filtering_awgn_std / num_trials * 100
ber_filtering_impulsive_std = errors_filtering_impulsive_std / num_trials * 100
ber_correction_awgn = errors_correction_awgn / num_trials * 100
ber_correction_impulsive = errors_correction_impulsive / num_trials * 100

# Report results explicitly for both Tri-Quarter and Standard Thresholding
print(f"Script 1 - Noise Filtering Results:")
print(f"Tri-Quarter Filtering AWGN BER = {ber_filtering_awgn_tri:.3f}%")
print(f"Standard Thresholding AWGN BER = {ber_filtering_awgn_std:.3f}%")
print(f"Tri-Quarter Filtering Impulsive BER = {ber_filtering_impulsive_tri:.3f}%")
print(f"Standard Thresholding Impulsive BER = {ber_filtering_impulsive_std:.3f}%")
print(f"Tri-Quarter Correction AWGN BER = {ber_correction_awgn:.3f}%")
print(f"Tri-Quarter Correction Impulsive BER = {ber_correction_impulsive:.3f}%")
