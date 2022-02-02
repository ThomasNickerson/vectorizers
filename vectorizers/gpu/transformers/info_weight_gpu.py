import math
from numba import cuda
import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MOCK_TARGET = cp.ones(1, dtype=np.int32)

# Utility functions needed for inside the CUDA kernel, maybe move to different file?
@cuda.jit(device=True, fastmath=True)
def gpu_sum(a):
    res = 0
    for x in a:
        res += x
    return res


@cuda.jit(device=True, fastmath=True)
def gpu_mean(a):
    return gpu_sum(a) / a.size


# TODO: Either remove this or figure out a GPU suitable algorithm
# def column_kl_divergence_exact_prior(
#    count_indices,
#    count_data,
#    baseline_probabilities,
#    prior_strength=0.1,
#    target=MOCK_TARGET,
# ):
#    observed_norm = count_data.sum() + prior_strength
#    observed_zero_constant = (prior_strength / observed_norm) * cp.log(
#        prior_strength / observed_norm
#    )
#    result = 0.0
#    count_indices_set = set(count_indices)
#    for i in range(baseline_probabilities.shape[0]):
#        if i in count_indices_set:
#            idx = cp.searchsorted(count_indices, i)
#            observed_probability = (
#               count_data[idx] + prior_strength * baseline_probabilities[i]
#            ) / observed_norm
#            if observed_probability > 0.0:
#                result += observed_probability * cp.log(
#                    observed_probability / baseline_probabilities[i]
#                )
#        else:
#            result += baseline_probabilities[i] * observed_zero_constant
#
#    return result


@cuda.jit(fastmath=True)
def column_kl_divergence_approx_prior_kernel(
    indptr, count_indices, count_data, baseline_probabilities, prior_strength, result
):
    col = cuda.grid(1)

    # guard against out of bounds access
    if col >= result.size:
        return

    observed_norm = gpu_sum(count_data[indptr[col] : indptr[col + 1]]) + prior_strength
    observed_zero_constant = (prior_strength / observed_norm) * math.log(
        prior_strength / observed_norm
    )

    zero_count_component_estimate = (
        gpu_mean(baseline_probabilities)
        * observed_zero_constant
        * (
            baseline_probabilities.shape[0]
            - count_indices[indptr[col] : indptr[col + 1]].shape[0]
        )
    )

    result[col] = zero_count_component_estimate

    for i in range(count_indices[indptr[col] : indptr[col + 1]].shape[0]):
        idx = count_indices[indptr[col] : indptr[col + 1]][i]
        observed_probability = (
            count_data[indptr[col] : indptr[col + 1]][i]
            + prior_strength * baseline_probabilities[idx]
        ) / observed_norm
        if observed_probability > 0.0 and baseline_probabilities[idx] > 0:
            result[col] += observed_probability * math.log(
                observed_probability / baseline_probabilities[idx]
            )


# TODO: GPU version of supervised column kl
def gpu_supervised_column_kl(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength=0.1,
    target=MOCK_TARGET,
):
    observed = cp.zeros_like(baseline_probabilities)
    for i in range(count_indices.shape[0]):
        idx = count_indices[i]
        label = target[idx]
        observed[label] += count_data[i]

    observed += prior_strength * baseline_probabilities
    observed /= observed.sum()

    return cp.sum(observed * cp.log(observed / baseline_probabilities))


def gpu_column_weights(
    indptr,
    indices,
    data,
    baseline_probabilities,
    column_kl_divergence_func,
    prior_strength=0.1,
    target=MOCK_TARGET,
):
    n_cols = indptr.shape[0] - 1
    weights = cp.ones(n_cols)

    threadsperblock = 32
    blockspergrid = (weights.size + (threadsperblock - 1)) // threadsperblock

    # Supervised info weight not implemented yet
    column_kl_divergence_func[blockspergrid, threadsperblock](
        indptr,
        indices,
        data,
        baseline_probabilities,
        prior_strength,
        #       target,
        weights,
    )
    return weights


def gpu_information_weight(
    data, prior_strength=0.1, approximate_prior=False, target=None
):
    """Compute information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution. The Bayesian prior can either be computed
    exactly (the default) at some computational expense, or estimated for a much
    fast computation, often suitable for large or very sparse datasets.

    Parameters
    ----------
    data: scipy sparse matrix (n_samples, n_features)
        A matrix of count data where rows represent observations and
        columns represent features. Column weightings will be learned
        from this data.

    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    approximate_prior: bool (optional, default=False)
        Whether to approximate weights based on the Bayesian prior or perform
        exact computations. Approximations are much faster especialyl for very
        large or very sparse datasets.

    target: ndarray or None (optional, default=None)
        If supervised target labels are available, these can be used to define distributions
        over the target classes rather than over rows, allowing weights to be
        supervised and target based. If None then unsupervised weighting is used.

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    if approximate_prior:
        column_kl_divergence_func = column_kl_divergence_approx_prior_kernel
    else:
        raise NotImplementedError(
            "GPUInformationWeightTransformer doesn't support exact prior computation yet."
        )

    baseline_counts = cp.squeeze(cp.array(data.sum(axis=1)))
    if target is None:
        baseline_probabilities = baseline_counts / baseline_counts.sum()
    else:
        raise NotImplementedError(
            "GPUInformationWeightTransformer doesn't support supervision yet."
        )

    csc_data = data.tocsc()
    csc_data.sort_indices()

    weights = gpu_column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        baseline_probabilities,
        column_kl_divergence_func,
        prior_strength=prior_strength,
        target=target,
    )

    return weights


class GPUInformationWeightTransformer(BaseEstimator, TransformerMixin):
    """A data transformer that re-weights columns of count data. Column weights
    are computed as information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution. The Bayesian prior can either be computed
    exactly (the default) at some computational expense, or estimated for a much
    fast computation, often suitable for large or very sparse datasets.

    Parameters
    ----------
    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    approximate_prior: bool (optional, default=False)
        Whether to approximate weights based on the Bayesian prior or perform
        exact computations. Approximations are much faster especialyl for very
        large or very sparse datasets.

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """

    def __init__(
        self,
        prior_strength=1e-4,
        approx_prior=True,
        weight_power=2.0,
        supervision_weight=0.95,
    ):
        self.prior_strength = prior_strength
        self.approx_prior = approx_prior
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight

    def fit(self, X, y=None, **fit_kwds):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        self:
            The trained model.
        """
        if not sp.isspmatrix(X):
            X = sp.csc_matrix(X)

        self.information_weights_ = gpu_information_weight(
            X, self.prior_strength, self.approx_prior
        )

        if y is not None:
            unsupervised_power = (1.0 - self.supervision_weight) * self.weight_power
            supervised_power = self.supervision_weight * self.weight_power

            self.information_weights_ /= cp.mean(self.information_weights_)
            self.information_weights_ = cp.power(
                self.information_weights_, unsupervised_power
            )

            target_classes = cp.unique(y)
            target_dict = dict(
                cp.vstack((target_classes, cp.arange(target_classes.shape[0]))).T
            )
            target = cp.array(
                [np.int32(target_dict[label]) for label in y], dtype=np.int32
            )
            self.supervised_weights_ = gpu_information_weight(
                X, self.prior_strength, self.approx_prior, target=target
            )
            self.supervised_weights_ /= cp.mean(self.supervised_weights_)
            self.supervised_weights_ = cp.power(
                self.supervised_weights_, supervised_power
            )

            self.information_weights_ = (
                self.information_weights_ * self.supervised_weights_
            )
        else:
            self.information_weights_ /= cp.mean(self.information_weights_)
            self.information_weights_ = cp.power(
                self.information_weights_, self.weight_power
            )

        return self

    def transform(self, X):
        """Reweight data ``X`` based on learned information weights of columns.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be transformed. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        result: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The reweighted data.
        """
        result = X @ sp.diags(self.information_weights_)
        return result
