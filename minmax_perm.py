import numpy as np
import pandas as pd


def minmax_perm(
    df: pd.DataFrame,
    control_groups: list,
    group_col: str,
    value_col: str,
    type: str,
    iterations: int,
):
    obs_means = df.groupby(group_col)[value_col].mean().sort_index()
    keys, values = obs_means.index, obs_means.values
    diff_matrix = values[:, None] - values

    np.fill_diagonal(diff_matrix, np.nan)

    if type == "greater":
        results = np.nanmin(diff_matrix, axis=1)
    elif type == "lesser":
        results = np.nanmax(diff_matrix, axis=1)

    control_groups.sort()
    result_map = dict(zip(keys, results))
    control_maps = {k: result_map[k] for k in control_groups}

    def perm_diff():
        shuffled_indices = np.random.permutation(len(df))
        start = 0
        mean_dict = {}
        for i, n in enumerate(df[group_col].value_counts().sort_index()):
            idx = shuffled_indices[start : start + n]
            mean = df[value_col].loc[idx].mean()
            start += n
            mean_dict[keys[i]] = mean

        perm_results = {}

        if type == "greater":
            for control in control_maps:
                c_mean = mean_dict[control]
                min_diff = +np.inf
                for group, g_mean in mean_dict.items():
                    if group != control:
                        diff = c_mean - g_mean
                        if diff < min_diff:
                            min_diff = diff
                if min_diff < +np.inf:
                    perm_results[control] = min_diff
            return perm_results

        elif type == "lesser":
            for control in control_maps:
                c_mean = mean_dict[control]
                max_diff = -np.inf
                for group, g_mean in mean_dict.items():
                    if group != control:
                        diff = c_mean - g_mean
                        if diff > max_diff:
                            max_diff = diff
                if max_diff > -np.inf:
                    perm_results[control] = max_diff
            return perm_results

    results = pd.DataFrame([perm_diff() for _ in range(iterations)])
    p_values = (
        (results > control_maps).mean()
        if type == "greater"
        else (results < control_maps).mean()
    )
    return results, p_values, control_maps
