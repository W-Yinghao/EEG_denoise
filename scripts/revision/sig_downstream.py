import numpy as np
from scipy import stats

ICA = np.array([
[89.29,39.29,25.93,50.00,40.74,27.27,62.96,29.63,36.00],
[82.61,92.86,33.33,41.67,33.33,31.82,33.33,37.04,36.00],
[53.57,85.71,81.48,58.33,33.33,54.55,44.44,33.33,48.00],
[53.57,46.43,81.48,91.67,33.33,54.55,37.04,37.04,44.00],
[42.86,35.71,37.04,91.67,88.89,54.55,40.74,33.33,40.00],
[50.00,39.29,37.04,50.00,77.78,90.91,44.44,33.33,44.00],
[57.14,50.00,37.04,41.67,55.56,90.91,74.07,33.33,40.00],
[39.29,39.29,44.44,50.00,59.26,45.45,66.67,74.07,44.00],
[32.14,32.14,37.04,45.83,44.44,45.45,33.33,74.07,84.00]])
SAD = np.array([
[85.59,46.43,46.43,52.00,37.04,31.82,68.22,37.04,38.46],
[85.71,90.01,34.28,44.00,44.44,37.27,32.14,44.18,35.77],
[60.71,92.86,82.11,58.00,58.15,55.00,49.29,51.85,44.62],
[52.86,47.21,78.57,88.45,48.15,50.00,38.57,40.74,46.15],
[45.86,35.71,37.04,91.67,85.47,52.00,42.81,36.12,46.15],
[52.14,35.00,46.43,44.44,79.12,88.00,49.15,33.33,50.00],
[55.71,50.00,42.16,52.00,55.56,90.91,76.12,43.10,47.28],
[39.29,39.29,44.44,47.25,59.26,45.45,70.37,78.88,44.00],
[25.15,35.71,32.14,38.20,44.44,45.45,34.07,73.08,85.15]])

diag_mask = np.eye(9,dtype=bool)
def summ(name,a,b):
    d=b-a
    t,pt=stats.ttest_rel(b,a)
    try: w,pw=stats.wilcoxon(b,a)
    except Exception as e: w,pw=np.nan,np.nan
    # bootstrap CI of mean diff
    rng=np.random.default_rng(42); n=len(d)
    bs=[np.mean(d[rng.integers(0,n,n)]) for _ in range(20000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    dz=np.mean(d)/np.std(d,ddof=1)
    print(f"{name:22s} n={n:2d}  meanΔ={np.mean(d):+6.2f}  95%CI[{lo:+.2f},{hi:+.2f}]  "
          f"paired-t p={pt:.3f}  Wilcoxon p={pw:.3f}  Cohen dz={dz:+.2f}")

print("=== Paired comparisons (SADDPM - ICA), using ONLY the published Table 2/3 numbers ===")
summ("All 81 cells", ICA.ravel(), SAD.ravel())
summ("Off-diagonal (72)", ICA[~diag_mask], SAD[~diag_mask])
summ("Diagonal (9)", ICA[diag_mask], SAD[diag_mask])
# per-target means (column means)
summ("Per-target means (9)", ICA.mean(0), SAD.mean(0))

print("\n=== Spread / stability claim (per-target means) ===")
ica_m, sad_m = ICA.mean(0), SAD.mean(0)
print(f"std(per-target mean): ICA={ica_m.std(ddof=1):.2f}  SADDPM={sad_m.std(ddof=1):.2f}  ratio={sad_m.std(ddof=1)/ica_m.std(ddof=1):.3f}")
lev = stats.levene(ica_m, sad_m, center='median')
print(f"Levene (equal variance of per-target means) p={lev.pvalue:.3f}")
# bootstrap CI on SD ratio by resampling the 9 target subjects (paired columns)
rng=np.random.default_rng(7); ratios=[]
for _ in range(20000):
    idx=rng.integers(0,9,9)
    ri=ICA[:,idx].mean(0).std(ddof=1); rs=SAD[:,idx].mean(0).std(ddof=1)
    if ri>0: ratios.append(rs/ri)
lo,hi=np.percentile(ratios,[2.5,97.5])
print(f"bootstrap 95% CI of SD ratio (SADDPM/ICA): [{lo:.2f}, {hi:.2f}]  (point {np.median(ratios):.2f})")

print("\n=== consistency check vs printed Mean rows ===")
print("ICA col means:", np.round(ica_m,2))
print("SAD col means:", np.round(sad_m,2))
print("printed ICA Mean: [55.61,51.19,46.09,57.87,51.85,55.05,48.56,42.80,46.22]")
print("printed SAD Mean: [55.89,52.47,49.29,57.33,56.85,55.10,51.19,48.70,48.62]")
print("Aggregate: ICA all=%.2f off=%.2f diag=%.2f | SAD all=%.2f off=%.2f diag=%.2f"%(
    ICA.mean(),ICA[~diag_mask].mean(),ICA[diag_mask].mean(),
    SAD.mean(),SAD[~diag_mask].mean(),SAD[diag_mask].mean()))
