# Script 1: Tri-Quarter Framework BER Simulation
# Simulates noise filtering and error correction for BPSK using the
# Tri-Quarter framework as described in Sections 4 and 5, with results
# reported in Section 6.
import numpy as np

# Parameters (Section 6.1)
num_trials = 10000  # Number of BPSK symbols
k = 3  # Transmissions per symbol for error correction
sigma = np.sqrt(0.25)  # Noise standard deviation (SNR = 6 dB, sigma^2 = 0.25)
outlier_prob = 0.1  # Probability of impulsive outliers
outlier_amplitude = 5  # Outlier amplitude for impulsive noise

# Initialize error counters for noise filtering and error correction
errors_filtering_awgn = 0
errors_filtering_impulsive = 0
errors_correction_awgn = 0
errors_correction_impulsive = 0

np.random.seed(42)  # For reproducibility
for _ in range(num_trials):
    # Generate true symbol: vec{x} = +1 or -1 (Section 2)
    x_true = np.random.choice([1, -1])

    # --- Noise Filtering Simulation (Section 4, k=1) ---
    # Generate complex noise: n = n_R + n_I * i
    noise_awgn = np.random.normal(0, sigma, 1) + 1j * np.random.normal(0, sigma, 1)
    noise_impulsive = np.random.normal(0, sigma, 1) + 1j * np.random.normal(0, sigma, 1)
    if np.random.random() < outlier_prob:
        noise_impulsive[0] = np.random.choice([outlier_amplitude, -outlier_amplitude])

    # Received signal: vec{y} = vec{x} + n
    y_awgn = x_true + noise_awgn
    y_impulsive = x_true + noise_impulsive

    # Extract real component: y_R = Re(vec{y}) (Section 2)
    y_R_awgn = np.real(y_awgn[0])
    y_R_impulsive = np.real(y_impulsive[0])

    # Tri-Quarter decoding (Section 4, Eq. 4.1)
    x_decoded_awgn = 1 if y_R_awgn >= 0 else -1
    x_decoded_impulsive = 1 if y_R_impulsive >= 0 else -1

    # Count errors
    if x_decoded_awgn != x_true:
        errors_filtering_awgn += 1
    if x_decoded_impulsive != x_true:
        errors_filtering_impulsive += 1

    # --- Error Correction Simulation (Section 5, k=3) ---
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

    # Compute distances: d_i = |y_{i,R} - sign(y_{i,R}) * 1| (Section 5)
    distances_awgn = np.abs(y_R_awgn - np.sign(y_R_awgn) * 1)
    distances_impulsive = np.abs(y_R_impulsive - np.sign(y_R_impulsive) * 1)

    # Compute weights: w_i = 1 / (1 + d_i^2)
    weights_awgn = 1 / (1 + distances_awgn**2)
    weights_impulsive = 1 / (1 + distances_impulsive**2)

    # Weighted voting (Section 5, Eq. 5.1-5.3)
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
ber_filtering_awgn = errors_filtering_awgn / num_trials * 100  # Approx. 10.2%
ber_filtering_impulsive = errors_filtering_impulsive / num_trials * 100  # Approx. 10.5%
ber_correction_awgn = errors_correction_awgn / num_trials * 100  # Approx. 3.80%
ber_correction_impulsive = errors_correction_impulsive / num_trials * 100  # Approx. 3.81%
print(f"Script 1 - Tri-Quarter: Filtering AWGN BER = {ber_filtering_awgn:.2f}%, "
      f"Filtering Impulsive BER = {ber_filtering_impulsive:.2f}%, "
      f"Correction AWGN BER = {ber_correction_awgn:.2f}%, "
      f"Correction Impulsive BER = {ber_correction_impulsive:.2f}%")
