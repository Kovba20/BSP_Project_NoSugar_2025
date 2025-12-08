import numpy as np
from scipy import stats

def perform_paired_ttest(df):
    """Perform paired t-tests for hypothesis testing."""
    results = {}

    # Test 1: Postural baseline vs post-fatigue (8-12 Hz band power) - PRIMARY HYPOTHESIS
    post_baseline = df[df['condition'] == 'post']['band_power_8_12'].values
    post_fatigue = df[df['condition'] == 'fat_post']['band_power_8_12'].values

    subjects_post = df[df['condition'] == 'post']['subject'].values
    subjects_fat_post = df[df['condition'] == 'fat_post']['subject'].values

    if len(post_baseline) > 0 and len(post_fatigue) > 0:
        common_subjects = set(subjects_post) & set(subjects_fat_post)
        if len(common_subjects) >= 2:
            paired_baseline = []
            paired_fatigue = []
            for subj in sorted(common_subjects):
                paired_baseline.append \
                    (df[(df['condition'] == 'post') & (df['subject'] == subj)]['band_power_8_12'].values[0])
                paired_fatigue.append \
                    (df[(df['condition'] == 'fat_post') & (df['subject'] == subj)]['band_power_8_12'].values[0])

            t_stat, p_value = stats.ttest_rel(paired_fatigue, paired_baseline)
            effect_size = (np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.std(paired_baseline) if np.std \
                (paired_baseline) > 0 else 0

            results['postural_8_12hz'] = {
                'comparison': 'Postural: Post-Fatigue vs Baseline',
                'feature': 'band_power_8_12',
                'n_subjects': len(common_subjects),
                'baseline_mean': np.mean(paired_baseline),
                'baseline_std': np.std(paired_baseline),
                'fatigue_mean': np.mean(paired_fatigue),
                'fatigue_std': np.std(paired_fatigue),
                'mean_diff': np.mean(paired_fatigue) - np.mean(paired_baseline),
                'percent_change': ((np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.mean(paired_baseline)) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant': p_value < 0.05
            }

    # Test 2: Rest baseline vs post-fatigue (8-12 Hz band power)
    rest_baseline = df[df['condition'] == 'rest']['band_power_8_12'].values
    rest_fatigue = df[df['condition'] == 'fat_rest']['band_power_8_12'].values

    subjects_rest = df[df['condition'] == 'rest']['subject'].values
    subjects_fat_rest = df[df['condition'] == 'fat_rest']['subject'].values

    if len(rest_baseline) > 0 and len(rest_fatigue) > 0:
        common_subjects = set(subjects_rest) & set(subjects_fat_rest)
        if len(common_subjects) >= 2:
            paired_baseline = []
            paired_fatigue = []
            for subj in sorted(common_subjects):
                paired_baseline.append \
                    (df[(df['condition'] == 'rest') & (df['subject'] == subj)]['band_power_8_12'].values[0])
                paired_fatigue.append \
                    (df[(df['condition'] == 'fat_rest') & (df['subject'] == subj)]['band_power_8_12'].values[0])

            t_stat, p_value = stats.ttest_rel(paired_fatigue, paired_baseline)
            effect_size = (np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.std(paired_baseline) if np.std \
                (paired_baseline) > 0 else 0

            results['rest_8_12hz'] = {
                'comparison': 'Rest: Post-Fatigue vs Baseline',
                'feature': 'band_power_8_12',
                'n_subjects': len(common_subjects),
                'baseline_mean': np.mean(paired_baseline),
                'baseline_std': np.std(paired_baseline),
                'fatigue_mean': np.mean(paired_fatigue),
                'fatigue_std': np.std(paired_fatigue),
                'mean_diff': np.mean(paired_fatigue) - np.mean(paired_baseline),
                'percent_change': ((np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.mean(paired_baseline)) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant': p_value < 0.05
            }

    # Test 3: Postural RMS comparison
    if len(post_baseline) > 0 and len(post_fatigue) > 0:
        common_subjects = set(subjects_post) & set(subjects_fat_post)
        if len(common_subjects) >= 2:
            paired_baseline = []
            paired_fatigue = []
            for subj in sorted(common_subjects):
                paired_baseline.append(df[(df['condition'] == 'post') & (df['subject'] == subj)]['rms'].values[0])
                paired_fatigue.append(df[(df['condition'] == 'fat_post') & (df['subject'] == subj)]['rms'].values[0])

            t_stat, p_value = stats.ttest_rel(paired_fatigue, paired_baseline)
            effect_size = (np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.std(paired_baseline) if np.std \
                (paired_baseline) > 0 else 0

            results['postural_rms'] = {
                'comparison': 'Postural: Post-Fatigue vs Baseline',
                'feature': 'rms',
                'n_subjects': len(common_subjects),
                'baseline_mean': np.mean(paired_baseline),
                'baseline_std': np.std(paired_baseline),
                'fatigue_mean': np.mean(paired_fatigue),
                'fatigue_std': np.std(paired_fatigue),
                'mean_diff': np.mean(paired_fatigue) - np.mean(paired_baseline),
                'percent_change': ((np.mean(paired_fatigue) - np.mean(paired_baseline)) / np.mean(paired_baseline)) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant': p_value < 0.05
            }

    return results