import pandas as pd
import numpy as np


data = {
    'Bicycle':       [32,  0,  2,  2,  0,  6,  2, 26,  0,  7,  1,  0],
    'Bridge':        [0,  40,  3,  1,  1,  0,  0,  2,  1,  2,  1,  5],
    'Bus':           [4,  12, 75, 16,  0,  1,  1,  2,  2,  4,  3,  2],
    'Car':           [16, 20, 65, 134,  2, 35,  1, 20, 18,  6, 17, 23],
    'Chimney':       [0,  1,  1,  0,  8,  0,  0,  0,  0,  2,  1,  1],
    'Crosswalk':     [5,  2,  2, 16,  0, 85,  1, 10,  0,  0,  1,  2],
    'Hydrant':       [1,  0,  0,  0,  0,  0, 93,  0,  0,  1,  0,  1],
    'Motorcycle':    [2,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0],
    'Palm':          [3,  2,  3,  5,  1,  1,  1,  0, 54,  1, 15,  6],
    'Stair':         [3,  1,  1,  0,  1,  1,  1,  0, 0,  12,  1,  1],
    'Traffic_Light': [0,  4,  2,  7,  0,  0,  3,  1, 18,  5, 38,  2],
    'Other':         [7, 14,  3, 14,  7, 11,  4,  1, 28,  3, 12, 31]
}
conf_mat = pd.DataFrame(data, index=list(data.keys()))


# Per-class TPR and TNR
def compute_tpr_tnr(confusion: pd.DataFrame) -> tuple[dict, dict]:
    """
    Computes true positive rate and true negative rate per class.


    :param confusion: The confusion matrix


    :return [tpr, tnr]: tuple of dictionaries containing true positive rate
    and true negative rates for each class.
    """
    total = confusion.values.sum()
    tpr, tnr = {}, {}
    for cls in confusion.index:
        TP = confusion.at[cls, cls]
        FN = confusion.loc[cls].sum() - TP
        FP = confusion[cls].sum() - TP
        TN = total - TP - FN - FP
        tpr[cls] = TP / (TP + FN) if TP + FN > 0 else 0
        tnr[cls] = TN / (TN + FP) if TN + FP > 0 else 0
    return tpr, tnr


tpr, tnr = compute_tpr_tnr(conf_mat)


# Probability to solve a single grid:
def solve_probability(n_images: int,
                      n_targets: int,
                      tpr_rate: float,
                      tnr_rate: float) -> float:
    """
    Computes the probability of correctly labeling all images in one challenge.
    - tpr_rate**n_targets for true positives
    - tnr_rate**(n_images - n_targets) for true negatives


    :param n_images: number of images in the reCAPTCHA grid (usually 9)
    :param n_targets: how many images are correct from the grid
    :param tpr_rate: true positive rate for all classes combined
    :param tnr_rate: true negative rate for all classes combined


    :returns probability:
    """
    return (tpr_rate ** n_targets) * (tnr_rate ** (n_images - n_targets))


def expected_solve_rate(n_images: int,
                        p_m: np.ndarray,
                        tpr: float,
                        tnr: float) -> float:
    """
    Compute the expected grid solve probability, averaging over the
    probability mass function p_m for M = {0,1,...,n_images} targets.


    p_m[k] should be P(M = k), and sum(p_m) == 1.
    """
    ms = np.arange(len(p_m))
    probs = [solve_probability(n_images, m, tpr, tnr) for m in ms]
    return float(np.dot(p_m, probs))


if __name__ == "__main__":

    # 3×3 reCAPTCHA grid
    N = 9

    # True Positive rate and True Negative rate (derived from
    # the tpr_tnr.py script)
    tpr_est = 0.514
    tnr_est = 0.956

    # Suppose we have 0 targets: 10%   1 target: 20%   2 targets: 30%
    # 3 targets: 25%   4 targets: 10%   5+ targets: 5%
    p_m = np.array([0.10, 0.20, 0.30, 0.25, 0.10, 0.05] + [0]*(N-5))
    p_m = p_m[:N+1]
    p_m = p_m / p_m.sum()  # normalize to sum to 1

    # Compute per‐m solve probabilities:
    for m in range(0, 6):
        print(f"M = {m:>1} targets → P_solve = "
              f"{solve_probability(N, m, tpr_est, tnr_est):.4f}")

    # And the overall expected success rate:
    overall = expected_solve_rate(N, p_m, tpr_est, tnr_est)
    print(f"\nExpected solve rate over all M: {overall:.4%}")
