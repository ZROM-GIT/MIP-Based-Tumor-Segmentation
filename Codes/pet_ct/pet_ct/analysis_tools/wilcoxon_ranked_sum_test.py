from scipy.stats import wilcoxon

def wilcoxon_ranked_sum_test(x, y, alternative='greater'):
    """
    Perform Wilcoxon signed-rank test for two paired samples.
    :param x: First list of values
    :param y: Second list of values
    :param alternative: Alternative hypothesis. Can be 'greater', 'less', or 'two-sided'.
    :return: statistic, p-value
    """

    (stat, p_value) = wilcoxon(x=x, y=y, alternative=alternative)

    return stat, p_value

if __name__ == "__main__":
    Tr_AS_Ts_Orig = [0.59, 0.592, 0.606, 0.597, 0.567]
    Tr_Orig_Ts_Orig = [0.583, 0.585, 0.563, 0.583, 0.574]
    Tr_3D_Ts_Orig = [0.617, 0.6, 0.572, 0.539, 0.657]
    Tr_3D_Ts_AS = [0.609, 0.589, 0.566, 0.534, 0.655]
    Tr_AS_Ts_AS = [0.618, 0.622, 0.636, 0.626, 0.59]
    Tr_Orig_Ts_AS = [0.606, 0.607, 0.58, 0.604, 0.591]
    stat, p_value = wilcoxon_ranked_sum_test(Tr_3D_Ts_Orig, Tr_Orig_Ts_Orig, alternative='greater')
    print(f"Statistic: {stat}, p-value: {p_value}")