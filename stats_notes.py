acc_a = 0.94
n = 90

import numpy as np
from scipy import stats

# Calculate p-values for n_correct_b from 80 to 90
# Using t-test on proportions directly (without rounding)
print("n_correct_b\tacc_b\t\tp-value (one-tailed: B > A)")
print("-" * 55)

# For method A: mean = acc_a, std = sqrt(acc_a * (1 - acc_a))
std_a = np.sqrt(acc_a * (1 - acc_a))

for n_correct_b in range(62, 91):
    # Calculate accuracy from number of correct answers
    acc_b = n_correct_b / n
    
    # For method B: mean = acc_b, std = sqrt(acc_b * (1 - acc_b))
    std_b = np.sqrt(acc_b * (1 - acc_b))
    
    # Perform one-tailed t-test using means and standard deviations
    # H0: mean_b <= mean_a, H1: mean_b > mean_a
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=acc_b, std1=std_b, nobs1=n,
        mean2=acc_a, std2=std_a, nobs2=n,
        alternative='greater'
    )
    
    print(f"{n_correct_b}\t\t{acc_b:.4f}\t\t{p_value:.6f}")

