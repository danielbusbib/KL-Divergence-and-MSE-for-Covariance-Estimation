# KL-Divergence-and-MSE-for-Covariance-Estimation
code of the paper: Comparing KL Divergence and MSE for Covariance Estimation in Target Detection, SSP 25

## Abstract
Covariance estimation is a core part of adaptive
target detection. Most of the works focus on the Mean Squared
Error (MSE) metric because it is easy to work with. However,
MSE does not always capture the statistical information needed
for detection. We advocate for switching to the Kullback-Leibler
(KL) divergence. To support this, we analyze the Normalized
Signal to Noise Ratio (NSNR) associated with the worst-case
target. We show that the KL metric has a structure similar to
NSNR and bounds it. To further clarify our point, we derive a
simple variant of a classic MSE-based estimator by incorporating
KL in a leave-one-out cross-validation (LOOCV) framework.
Numerical experiments with various estimators on both synthetic
and real data also demonstrate that KL and NSNR behave
similarly and are different than MSE. Simply changing the metric
in the LOOCV estimator improves KL and NSNR performance
while reducing MSE performance.

## KL BASED LOOCV SHRINKAGE ESTIMATOR
```python
def loocv_loglike(X, alpha):
    D, N = X.shape
    S = X @ X.T / N
    T = np.trace(S)/D * np.eye(D)
    C = (1 - alpha) * S + alpha * T
    invC = np.linalg.inv(C)
    z = np.einsum('ji,jk,ki->i', X, invC, X)
    trace_term = np.mean(z / (1 - (1 - alpha) / N * z))
    log_det = np.log(np.linalg.det(C))
    dist = trace_term + log_det
    return dist


def loocv(X):
  alphas = np.logspace(-3,-.01,20)
  B, M, D = X.shape
  Chat = np.zeros((B, D, D))
  for b in range(B):
    Xi = X[b,:,:].T
    distances = np.array([loocv_loglike(Xi, alpha) for alpha in alphas])
    optimal_alpha = alphas[np.argmin(distances)]
    S = Xi @ Xi.T / M
    Chat[b] = (1-optimal_alpha) * S + optimal_alpha * np.trace(S)/D * np.eye(D)
  return Chat

```

## Citation

Please cite if you are using this code for your research:

```
@INPROCEEDINGS{11073417,
  author={Busbib, Daniel and Diskin, Tzvi and Wiesel, Ami},
  booktitle={2025 IEEE Statistical Signal Processing Workshop (SSP)}, 
  title={Comparing KL Divergence and MSE for Covariance Estimation in Target Detection}, 
  year={2025},
  pages={101-105},
  keywords={Measurement;Conferences;Estimation;Signal processing algorithms;Object detection;Switches;Signal processing;Signal to noise ratio},
  doi={10.1109/SSP64130.2025.11073417}}

```


