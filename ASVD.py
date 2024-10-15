import copy
import random
import numpy as np
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array
from fancyimpute.common import masked_mae, generate_random_column_samples
import warnings

F32PREC = np.finfo(np.float32).eps

class ClassificationAdaptiveSVDSolver:

    def __init__(
            self,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=100,
            max_rank=None,
            n_power_iterations=1,
            init_fill_method="std",
            missing_value_type = None,
            soft_threshold_ratio = 0.98,
            svd_upper_limit = 0.5,
            svd_lower_limit=0.25,
            mnar_upper_limit=0.4,
            mnar_lower_limit=0.2,
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters for the imputation algorithm
        """
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.init_fill_method = init_fill_method
        self.missing_value_type = missing_value_type
        self.soft_threshold_ratio = soft_threshold_ratio
        self.svd_upper_limit = svd_upper_limit
        self.svd_lower_limit = svd_lower_limit
        self.mnar_upper_limit = mnar_upper_limit
        self.mnar_lower_limit = mnar_lower_limit
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.verbose = verbose

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            warnings.simplefilter("always")
            warnings.warn("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        if max_rank:
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations,
                random_state=None)
        else:
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return X_reconstruction, rank

    def _max_singular_value(self, X_filled):
        _, s, _ = randomized_svd(X_filled, 1, n_iter=5, random_state=None)
        return s[0]

    def clip(self, X):
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def fill(self, X, missing_mask, X_init):
        if self.init_fill_method not in ("zero", "mean", "median", "min", "random", "std"):
            raise ValueError("Invalid fill method: '%s'" % (self.init_fill_method))
        elif self.init_fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif self.init_fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif self.init_fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif self.init_fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif self.init_fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        elif self.init_fill_method == "std":
            X = copy.deepcopy(X_init)
        return X

    def solve(self, X, missing_mask, missing_init, X_classify, X_min):
        X = check_array(X, force_all_finite=False)
        X_init = X.copy()
        X_filled = X
        observed_mask = ~missing_mask
        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            print(f"[SoftImpute] Max Singular Value of X_init = {max_singular_value:.6f}")

        shrinkage_value = self.shrinkage_value or self._determine_shrinkage_value(X)
        mnar_positions = X_classify == "MNAR"
        svd_random_weights = np.random.uniform(self.svd_lower_limit, self.svd_upper_limit, size=np.sum(mnar_positions))
        X_filled[mnar_positions] *= svd_random_weights
        for i in range(self.max_iters):
            X_reconstruction, rank = self._svd_step(X_filled, shrinkage_value, max_rank=self.max_rank)
            if self.verbose:
                mae = masked_mae(X_true=X_init, X_pred=X_reconstruction, mask=observed_mask)
                print(f"[SoftImpute] Iter {i + 1}: observed MAE={mae:.6f} rank={rank}")

            converged = self._converged(X_filled, X_reconstruction, missing_mask)
            X_reconstruction = self.clip(X_reconstruction)
            if i != self.max_iters - 1 and not converged:
                min_mask = mnar_positions & (X_reconstruction > X_min)
                mnar_random_weights = np.random.uniform(self.mnar_lower_limit, self.mnar_upper_limit, size=np.sum(min_mask))
                X_reconstruction[min_mask] = (missing_init[min_mask] * mnar_random_weights +
                                              X_reconstruction[min_mask] * (1 - mnar_random_weights))
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break
        if self.verbose:
            print(f"[SoftImpute] Stopped after iteration {i + 1} for lambda={shrinkage_value}")

        return X_filled

    def _determine_shrinkage_value(self, X):
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        shrinkage_sum = np.sum(s) * self.soft_threshold_ratio
        current_sum = np.sum(s)
        k = len(s)
        while current_sum > shrinkage_sum and k > 0:
            k -= 1
            current_sum -= s[k]
        return s[k]

    def fit_transform(self, X, X_init, X_classify, X_min):
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask

        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)

        X_filled = self.fill(X, missing_mask, X_init)

        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                f"Expected fill method to return a NumPy array but got {type(X_filled)}"
            )

        X_result = self.solve(X_filled, missing_mask, X_init, X_classify, X_min)

        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                f"Expected solve method to return a NumPy array but got {type(X_result)}"
            )

        X_result = self.project_result(X_result)

        X_result[observed_mask] = X_original[observed_mask]

        return X_result

    def prepare_input_data(self, X):
        X = check_array(X, force_all_finite=False)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def project_result(self, X):
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)
