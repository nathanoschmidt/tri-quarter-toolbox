# Script 2: Majority Voting BER Simulation
# Simulates error correction for BPSK using majority voting, as
# described in Section 5, with results reported in Section 6.
import numpy as np

# Parameters (Section 6.1)
num_trials = 100000
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

    # Generate real noise for k=3 transmissions
    noise_awgn = np.random.normal(0, sigma, k)
    noise_impulsive = np.random.normal(0, sigma, k)
    for i in range(k):
        if np.random.random() < outlier_prob:
            noise_impulsive[i] = np.random.choice([outlier_amplitude, -outlier_amplitude])

    # Received signals: y_i = x + n_i
    y_awgn = x_true + noise_awgn
    y_impulsive = x_true + noise_impulsive

    # Compute signs: sign(y_i) (Section 3, Eq. 3.3)
    votes_awgn = np.sign(y_awgn)
    votes_impulsive = np.sign(y_impulsive)

    # Majority voting
    vote_count_awgn = np.sum(votes_awgn > 0)
    vote_count_impulsive = np.sum(votes_impulsive > 0)
    x_decoded_awgn = 1 if vote_count_awgn >= (k - vote_count_awgn) else -1
    x_decoded_impulsive = 1 if vote_count_impulsive >= (k - vote_count_impulsive) else -1

    # Count errors
    if x_decoded_awgn != x_true:
        errors_correction_awgn += 1
    if x_decoded_impulsive != x_true:
        errors_correction_impulsive += 1

# Compute BER (Section 6)
ber_correction_awgn = errors_correction_awgn / num_trials * 100  # Approx. 0.002%
ber_correction_impulsive = errors_correction_impulsive / num_trials * 100  # Approx. 1.5%
print(f"Script 2 - Majority Voting: Correction AWGN BER = {ber_correction_awgn:.3f}%, "
      f"Correction Impulsive BER = {ber_correction_impulsive:.3f}%")
