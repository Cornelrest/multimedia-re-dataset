"""
Statistical Validation Module for Multimedia Requirements Engineering

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University, Czech Republic
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.power import ttest_power
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Some advanced statistical tests may be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available. Plotting functions will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Data class for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: bool = True
    sample_size: int = 0

@dataclass
class ExperimentalData:
    """Data class for experimental results."""
    control_group: np.ndarray
    treatment_group: np.ndarray
    metric_name: str
    group_labels: Tuple[str, str] = ("Control", "Treatment")

class StatisticalValidator:
    """Main class for statistical validation of experimental results."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical validator.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.results = {}
        self.assumptions_log = []
        
    def validate_experiment(self, experimental_data: Dict[str, ExperimentalData]) -> Dict[str, Any]:
        """
        Comprehensive validation of experimental results.
        
        Args:
            experimental_data: Dictionary of ExperimentalData objects for each metric
            
        Returns:
            Dictionary containing all validation results
        """
        logger.info("Starting comprehensive statistical validation...")
        
        validation_results = {
            'summary': {},
            'individual_tests': {},
            'assumption_checks': {},
            'effect_sizes': {},
            'power_analysis': {},
            'multiple_comparisons': {},
            'overall_conclusion': ''
        }
        
        # Individual metric validation
        for metric_name, data in experimental_data.items():
            logger.info(f"Validating metric: {metric_name}")
            
            # Check assumptions
            assumptions = self.check_statistical_assumptions(data)
            validation_results['assumption_checks'][metric_name] = assumptions
            
            # Perform appropriate tests
            test_results = self.perform_statistical_tests(data, assumptions)
            validation_results['individual_tests'][metric_name] = test_results
            
            # Calculate effect sizes
            effect_size = self.calculate_effect_sizes(data)
            validation_results['effect_sizes'][metric_name] = effect_size
            
            # Power analysis
            power_results = self.perform_power_analysis(data)
            validation_results['power_analysis'][metric_name] = power_results
        
        # Multiple comparisons correction
        if len(experimental_data) > 1:
            mc_results = self.correct_multiple_comparisons(experimental_data)
            validation_results['multiple_comparisons'] = mc_results
        
        # Generate summary
        validation_results['summary'] = self.generate_validation_summary(validation_results)
        
        # Overall conclusion
        validation_results['overall_conclusion'] = self.generate_overall_conclusion(validation_results)
        
        logger.info("Statistical validation completed.")
        return validation_results

    def check_statistical_assumptions(self, data: ExperimentalData) -> Dict[str, Any]:
        """
        Check statistical assumptions for parametric tests.
        
        Args:
            data: Experimental data to check
            
        Returns:
            Dictionary containing assumption check results
        """
        assumptions = {
            'normality': {},
            'homogeneity_of_variance': {},
            'independence': {},
            'sample_size': {},
            'recommendations': []
        }
        
        # Normality tests
        control_shapiro = shapiro(data.control_group)
        treatment_shapiro = shapiro(data.treatment_group)
        
        assumptions['normality'] = {
            'control_shapiro_stat': control_shapiro.statistic,
            'control_shapiro_p': control_shapiro.pvalue,
            'control_normal': control_shapiro.pvalue > self.alpha,
            'treatment_shapiro_stat': treatment_shapiro.statistic,
            'treatment_shapiro_p': treatment_shapiro.pvalue,
            'treatment_normal': treatment_shapiro.pvalue > self.alpha,
            'both_normal': (control_shapiro.pvalue > self.alpha and 
                          treatment_shapiro.pvalue > self.alpha)
        }
        
        # Homogeneity of variance (Levene's test)
        levene_stat, levene_p = levene(data.control_group, data.treatment_group)
        assumptions['homogeneity_of_variance'] = {
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'equal_variances': levene_p > self.alpha
        }
        
        # Independence check (based on experimental design)
        assumptions['independence'] = {
            'independent_groups': True,  # Assumes between-subjects design
            'random_assignment': True    # Assumes proper randomization
        }
        
        # Sample size adequacy
        n_control = len(data.control_group)
        n_treatment = len(data.treatment_group)
        
        assumptions['sample_size'] = {
            'control_n': n_control,
            'treatment_n': n_treatment,
            'total_n': n_control + n_treatment,
            'adequate_size': n_control >= 20 and n_treatment >= 20,
            'balanced_groups': abs(n_control - n_treatment) <= 5
        }
        
        # Generate recommendations
        if not assumptions['normality']['both_normal']:
            assumptions['recommendations'].append("Consider non-parametric tests due to non-normality")
        
        if not assumptions['homogeneity_of_variance']['equal_variances']:
            assumptions['recommendations'].append("Use Welch's t-test due to unequal variances")
        
        if not assumptions['sample_size']['adequate_size']:
            assumptions['recommendations'].append("Sample size may be inadequate for reliable inference")
        
        return assumptions

    def perform_statistical_tests(self, data: ExperimentalData, 
                                assumptions: Dict[str, Any]) -> Dict[str, StatisticalResult]:
        """
        Perform appropriate statistical tests based on assumption checks.
        
        Args:
            data: Experimental data
            assumptions: Results from assumption checking
            
        Returns:
            Dictionary of statistical test results
        """
        test_results = {}
        
        # Determine appropriate test
        both_normal = assumptions['normality']['both_normal']
        equal_variances = assumptions['homogeneity_of_variance']['equal_variances']
        
        if both_normal:
            if equal_variances:
                # Standard independent t-test
                result = self.independent_t_test(data, equal_var=True)
                test_results['parametric_primary'] = result
            else:
                # Welch's t-test
                result = self.independent_t_test(data, equal_var=False)
                test_results['parametric_primary'] = result
        
        # Always perform non-parametric alternative
        result = self.mann_whitney_u_test(data)
        test_results['non_parametric'] = result
        
        # Robust tests for additional validation
        if both_normal:
            # Bootstrap confidence interval
            bootstrap_result = self.bootstrap_mean_difference(data)
            test_results['bootstrap'] = bootstrap_result
        
        return test_results

    def independent_t_test(self, data: ExperimentalData, equal_var: bool = True) -> StatisticalResult:
        """
        Perform independent samples t-test.
        
        Args:
            data: Experimental data
            equal_var: Whether to assume equal variances
            
        Returns:
            Statistical test result
        """
        statistic, p_value = ttest_ind(
            data.treatment_group, 
            data.control_group, 
            equal_var=equal_var
        )
        
        # Calculate effect size (Cohen's d)
        effect_size = self.cohens_d(data.treatment_group, data.control_group)
        
        # Calculate confidence interval for mean difference
        ci = self.mean_difference_ci(data.treatment_group, data.control_group)
        
        test_name = "Independent t-test" if equal_var else "Welch's t-test"
        interpretation = self.interpret_t_test(statistic, p_value, effect_size)
        
        return StatisticalResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            sample_size=len(data.control_group) + len(data.treatment_group)
        )

    def mann_whitney_u_test(self, data: ExperimentalData) -> StatisticalResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            data: Experimental data
            
        Returns:
            Statistical test result
        """
        statistic, p_value = mannwhitneyu(
            data.treatment_group,
            data.control_group,
            alternative='two-sided'
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data.treatment_group), len(data.control_group)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        interpretation = self.interpret_mann_whitney(statistic, p_value, effect_size)
        
        return StatisticalResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            sample_size=n1 + n2
        )

    def bootstrap_mean_difference(self, data: ExperimentalData, 
                                n_bootstrap: int = 10000) -> StatisticalResult:
        """
        Bootstrap confidence interval for mean difference.
        
        Args:
            data: Experimental data
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap result
        """
        np.random.seed(42)  # For reproducibility
        
        n_treatment = len(data.treatment_group)
        n_control = len(data.control_group)
        
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            boot_treatment = np.random.choice(data.treatment_group, n_treatment, replace=True)
            boot_control = np.random.choice(data.control_group, n_control, replace=True)
            
            # Calculate mean difference
            diff = np.mean(boot_treatment) - np.mean(boot_control)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Calculate p-value (proportion of bootstrap samples with diff <= 0)
        p_value = np.mean(bootstrap_diffs <= 0) * 2  # Two-tailed
        
        observed_diff = np.mean(data.treatment_group) - np.mean(data.control_group)
        
        interpretation = f"Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]. "
        if 0 not in (ci_lower, ci_upper):
            interpretation += "Significant difference (CI excludes 0)."
        else:
            interpretation += "No significant difference (CI includes 0)."
        
        return StatisticalResult(
            test_name="Bootstrap mean difference",
            statistic=observed_diff,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            sample_size=n_treatment + n_control
        )

    def calculate_effect_sizes(self, data: ExperimentalData) -> Dict[str, float]:
        """
        Calculate various effect size measures.
        
        Args:
            data: Experimental data
            
        Returns:
            Dictionary of effect sizes
        """
        effect_sizes = {}
        
        # Cohen's d
        effect_sizes['cohens_d'] = self.cohens_d(data.treatment_group, data.control_group)
        
        # Hedges' g (bias-corrected)
        effect_sizes['hedges_g'] = self.hedges_g(data.treatment_group, data.control_group)
        
        # Glass's delta
        effect_sizes['glass_delta'] = self.glass_delta(data.treatment_group, data.control_group)
        
        # Common language effect size
        effect_sizes['cles'] = self.common_language_effect_size(data.treatment_group, data.control_group)
        
        # Probability of superiority
        effect_sizes['prob_superiority'] = self.probability_of_superiority(data.treatment_group, data.control_group)
        
        return effect_sizes

    def perform_power_analysis(self, data: ExperimentalData) -> Dict[str, Any]:
        """
        Perform power analysis for the statistical tests.
        
        Args:
            data: Experimental data
            
        Returns:
            Dictionary containing power analysis results
        """
        power_results = {}
        
        n_control = len(data.control_group)
        n_treatment = len(data.treatment_group)
        effect_size = self.cohens_d(data.treatment_group, data.control_group)
        
        if STATSMODELS_AVAILABLE:
            # Post-hoc power analysis
            observed_power = ttest_power(
                effect_size=abs(effect_size),
                nobs=min(n_control, n_treatment),
                alpha=self.alpha,
                alternative='two-sided'
            )
            
            power_results['observed_power'] = observed_power
            
            # Required sample size for 80% power
            from statsmodels.stats.power import solve_power
            required_n = solve_power(
                effect_size=abs(effect_size),
                power=0.8,
                alpha=self.alpha,
                alternative='two-sided'
            )
            
            power_results['required_n_80_power'] = required_n
            
            # Minimum detectable effect for current sample size
            min_detectable_effect = solve_power(
                nobs=min(n_control, n_treatment),
                power=0.8,
                alpha=self.alpha,
                alternative='two-sided'
            )
            
            power_results['min_detectable_effect'] = min_detectable_effect
        
        else:
            # Simple power estimation
            power_results['observed_power'] = self.estimate_power_simple(
                effect_size, min(n_control, n_treatment)
            )
        
        power_results['sample_sizes'] = {
            'control': n_control,
            'treatment': n_treatment,
            'total': n_control + n_treatment
        }
        
        power_results['interpretation'] = self.interpret_power_analysis(power_results)
        
        return power_results

    def correct_multiple_comparisons(self, experimental_data: Dict[str, ExperimentalData]) -> Dict[str, Any]:
        """
        Apply multiple comparisons correction.
        
        Args:
            experimental_data: Dictionary of experimental data for each metric
            
        Returns:
            Multiple comparisons correction results
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Multiple comparisons correction limited.")
            return {}
        
        # Extract p-values from parametric tests
        p_values = []
        metric_names = []
        
        for metric_name, data in experimental_data.items():
            # Use the primary parametric test result
            assumptions = self.check_statistical_assumptions(data)
            test_results = self.perform_statistical_tests(data, assumptions)
            
            if 'parametric_primary' in test_results:
                p_values.append(test_results['parametric_primary'].p_value)
                metric_names.append(metric_name)
        
        if not p_values:
            return {}
        
        # Apply different correction methods
        corrections = {}
        
        # Bonferroni correction
        bonf_rejected, bonf_corrected, _, bonf_alpha = multipletests(
            p_values, alpha=self.alpha, method='bonferroni'
        )
        
        corrections['bonferroni'] = {
            'method': 'Bonferroni',
            'corrected_alpha': bonf_alpha,
            'original_p_values': dict(zip(metric_names, p_values)),
            'corrected_p_values': dict(zip(metric_names, bonf_corrected)),
            'rejected': dict(zip(metric_names, bonf_rejected)),
            'significant_metrics': [name for name, rejected in zip(metric_names, bonf_rejected) if rejected]
        }
        
        # Benjamini-Hochberg (FDR) correction
        fdr_rejected, fdr_corrected, _, _ = multipletests(
            p_values, alpha=self.alpha, method='fdr_bh'
        )
        
        corrections['fdr_bh'] = {
            'method': 'Benjamini-Hochberg (FDR)',
            'corrected_p_values': dict(zip(metric_names, fdr_corrected)),
            'rejected': dict(zip(metric_names, fdr_rejected)),
            'significant_metrics': [name for name, rejected in zip(metric_names, fdr_rejected) if rejected]
        }
        
        # Holm correction
        holm_rejected, holm_corrected, _, _ = multipletests(
            p_values, alpha=self.alpha, method='holm'
        )
        
        corrections['holm'] = {
            'method': 'Holm',
            'corrected_p_values': dict(zip(metric_names, holm_corrected)),
            'rejected': dict(zip(metric_names, holm_rejected)),
            'significant_metrics': [name for name, rejected in zip(metric_names, holm_rejected) if rejected]
        }
        
        corrections['summary'] = self.summarize_multiple_comparisons(corrections)
        
        return corrections

    def generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of validation results.
        
        Args:
            validation_results: Complete validation results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_metrics': len(validation_results.get('individual_tests', {})),
            'significant_results': {},
            'effect_size_summary': {},
            'power_summary': {},
            'assumption_violations': []
        }
        
        # Count significant results
        for metric, tests in validation_results.get('individual_tests', {}).items():
            for test_name, result in tests.items():
                if result.p_value < self.alpha:
                    if metric not in summary['significant_results']:
                        summary['significant_results'][metric] = []
                    summary['significant_results'][metric].append(test_name)
        
        # Effect size summary
        for metric, effect_sizes in validation_results.get('effect_sizes', {}).items():
            cohens_d = effect_sizes.get('cohens_d', 0)
            summary['effect_size_summary'][metric] = {
                'cohens_d': cohens_d,
                'magnitude': self.interpret_cohens_d_magnitude(cohens_d)
            }
        
        # Power summary
        for metric, power_data in validation_results.get('power_analysis', {}).items():
            summary['power_summary'][metric] = {
                'observed_power': power_data.get('observed_power', 0),
                'adequate_power': power_data.get('observed_power', 0) > 0.8
            }
        
        # Assumption violations
        for metric, assumptions in validation_results.get('assumption_checks', {}).items():
            violations = []
            if not assumptions['normality']['both_normal']:
                violations.append('normality')
            if not assumptions['homogeneity_of_variance']['equal_variances']:
                violations.append('equal_variances')
            if not assumptions['sample_size']['adequate_size']:
                violations.append('sample_size')
            
            if violations:
                summary['assumption_violations'].append({
                    'metric': metric,
                    'violations': violations
                })
        
        return summary

    def generate_overall_conclusion(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate overall conclusion from validation results.
        
        Args:
            validation_results: Complete validation results
            
        Returns:
            Overall conclusion string
        """
        summary = validation_results.get('summary', {})
        significant_count = len(summary.get('significant_results', {}))
        total_metrics = summary.get('total_metrics', 0)
        
        conclusion = f"Statistical validation of {total_metrics} metrics revealed "
        conclusion += f"{significant_count} metrics with statistically significant differences "
        conclusion += f"between treatment and control groups (α = {self.alpha}).\n\n"
        
        # Effect sizes
        large_effects = 0
        for metric, effect_data in summary.get('effect_size_summary', {}).items():
            if abs(effect_data.get('cohens_d', 0)) > 0.8:
                large_effects += 1
        
        conclusion += f"{large_effects} metrics showed large effect sizes (|d| > 0.8), "
        conclusion += "indicating substantial practical significance.\n\n"
        
        # Power analysis
        adequate_power_count = sum(
            1 for power_data in summary.get('power_summary', {}).values()
            if power_data.get('adequate_power', False)
        )
        
        conclusion += f"{adequate_power_count}/{total_metrics} metrics had adequate "
        conclusion += "statistical power (>80%) for reliable inference.\n\n"
        
        # Multiple comparisons
        mc_results = validation_results.get('multiple_comparisons', {})
        if mc_results:
            bonf_significant = len(mc_results.get('bonferroni', {}).get('significant_metrics', []))
            conclusion += f"After Bonferroni correction for multiple comparisons, "
            conclusion += f"{bonf_significant} metrics remained statistically significant.\n\n"
        
        # Final recommendation
        if significant_count > 0 and large_effects > 0:
            conclusion += "CONCLUSION: Strong evidence supports the effectiveness of the "
            conclusion += "multimedia-enhanced approach with both statistical significance "
            conclusion += "and practical importance."
        elif significant_count > 0:
            conclusion += "CONCLUSION: Moderate evidence supports the multimedia-enhanced "
            conclusion += "approach with statistical significance but limited effect sizes."
        else:
            conclusion += "CONCLUSION: Insufficient evidence to support the multimedia-enhanced "
            conclusion += "approach. Consider methodological improvements or larger sample sizes."
        
        return conclusion

    # Helper methods for statistical calculations

    def cohens_d(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(treatment), len(control)
        pooled_std = np.sqrt(((n1 - 1) * np.var(treatment, ddof=1) + 
                             (n2 - 1) * np.var(control, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(treatment) - np.mean(control)) / pooled_std

    def hedges_g(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate Hedges' g (bias-corrected Cohen's d)."""
        d = self.cohens_d(treatment, control)
        n = len(treatment) + len(control)
        correction_factor = 1 - (3 / (4 * n - 9))
        return d * correction_factor

    def glass_delta(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate Glass's delta effect size."""
        return (np.mean(treatment) - np.mean(control)) / np.std(control, ddof=1)

    def common_language_effect_size(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate common language effect size."""
        comparisons = 0
        favorable = 0
        
        for t_val in treatment:
            for c_val in control:
                comparisons += 1
                if t_val > c_val:
                    favorable += 1
        
        return favorable / comparisons if comparisons > 0 else 0.5

    def probability_of_superiority(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate probability of superiority."""
        # Same as common language effect size
        return self.common_language_effect_size(treatment, control)

    def mean_difference_ci(self, treatment: np.ndarray, control: np.ndarray, 
                          confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference."""
        diff = np.mean(treatment) - np.mean(control)
        n1, n2 = len(treatment), len(control)
        
        # Pooled standard error
        pooled_var = ((n1 - 1) * np.var(treatment, ddof=1) + 
                     (n2 - 1) * np.var(control, ddof=1)) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # t-critical value
        df = n1 + n2 - 2
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_crit * se_diff
        
        return (diff - margin_error, diff + margin_error)

    def estimate_power_simple(self, effect_size: float, n_per_group: int) -> float:
        """Simple power estimation when statsmodels unavailable."""
        # Simplified power calculation
        delta = abs(effect_size) * np.sqrt(n_per_group / 2)
        power = 1 - stats.norm.cdf(1.96 - delta) + stats.norm.cdf(-1.96 - delta)
        return max(0, min(1, power))

    def interpret_t_test(self, statistic: float, p_value: float, effect_size: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        direction = "higher" if statistic > 0 else "lower"
        magnitude = self.interpret_cohens_d_magnitude(effect_size)
        
        return (f"Result is {significance} (p = {p_value:.4f}). "
               f"Treatment group scored {direction} than control group "
               f"with a {magnitude} effect size (d = {effect_size:.3f}).")

    def interpret_mann_whitney(self, statistic: float, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        return (f"Non-parametric test is {significance} (p = {p_value:.4f}). "
               f"Rank-biserial correlation = {effect_size:.3f}.")

    def interpret_cohens_d_magnitude(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def interpret_power_analysis(self, power_results: Dict[str, Any]) -> str:
        """Interpret power analysis results."""
        observed_power = power_results.get('observed_power', 0)
        
        interpretation = f"Observed power: {observed_power:.3f}. "
        
        if observed_power >= 0.8:
            interpretation += "Adequate power for reliable inference."
        elif observed_power >= 0.6:
            interpretation += "Moderate power - results should be interpreted with caution."
        else:
            interpretation += "Low power - high risk of Type II error."
        
        if 'required_n_80_power' in power_results:
            required_n = power_results['required_n_80_power']
            interpretation += f" Required sample size for 80% power: {required_n:.0f} per group."
        
        return interpretation

    def summarize_multiple_comparisons(self, corrections: Dict[str, Any]) -> str:
        """Summarize multiple comparisons correction results."""
        bonf_sig = len(corrections.get('bonferroni', {}).get('significant_metrics', []))
        fdr_sig = len(corrections.get('fdr_bh', {}).get('significant_metrics', []))
        holm_sig = len(corrections.get('holm', {}).get('significant_metrics', []))
        
        summary = f"Multiple comparisons correction results:\n"
        summary += f"- Bonferroni: {bonf_sig} significant metrics\n"
        summary += f"- FDR (Benjamini-Hochberg): {fdr_sig} significant metrics\n"
        summary += f"- Holm: {holm_sig} significant metrics\n"
        
        if bonf_sig > 0:
            summary += "Bonferroni correction (most conservative) still shows significant results."
        else:
            summary += "No metrics survive the conservative Bonferroni correction."
        
        return summary

def calculate_cohens_kappa(expert1_reqs: List[str], expert2_reqs: List[str], 
                          all_possible_reqs: List[str]) -> Dict[str, float]:
    """
    Calculate Cohen's kappa for inter-rater reliability.
    
    Args:
        expert1_reqs: Requirements identified by expert 1
        expert2_reqs: Requirements identified by expert 2
        all_possible_reqs: Complete list of possible requirements
        
    Returns:
        Dictionary containing kappa statistics
    """
    # Create binary matrices (requirement present/absent)
    expert1_binary = np.array([1 if req in expert1_reqs else 0 for req in all_possible_reqs])
    expert2_binary = np.array([1 if req in expert2_reqs else 0 for req in all_possible_reqs])
    
    # Calculate observed agreement
    observed_agreement = np.mean(expert1_binary == expert2_binary)
    
    # Calculate expected agreement by chance
    p1_positive = np.mean(expert1_binary)
    p2_positive = np.mean(expert2_binary)
    expected_agreement = (p1_positive * p2_positive + 
                         (1-p1_positive) * (1-p2_positive))
    
    # Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    # Calculate standard error and confidence interval
    n = len(all_possible_reqs)
    se_kappa = np.sqrt(expected_agreement / (n * (1 - expected_agreement)**2))
    
    # 95% confidence interval
    z_critical = 1.96
    ci_lower = kappa - z_critical * se_kappa
    ci_upper = kappa + z_critical * se_kappa
    
    # Interpretation
    if kappa < 0:
        interpretation = "Poor agreement (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Near perfect agreement"
    
    return {
        'kappa': kappa,
        'observed_agreement': observed_agreement,
        'expected_agreement': expected_agreement,
        'standard_error': se_kappa,
        'confidence_interval': (ci_lower, ci_upper),
        'interpretation': interpretation,
        'sample_size': n
    }

def validate_multimedia_re_results(control_data: Dict[str, np.ndarray],
                                  treatment_data: Dict[str, np.ndarray],
                                  alpha: float = 0.05) -> Dict[str, Any]:
    """
    Comprehensive validation function for multimedia RE experimental results.
    
    Args:
        control_data: Dictionary of control group measurements
        treatment_data: Dictionary of treatment group measurements  
        alpha: Significance level
        
    Returns:
        Complete validation results
    """
    validator = StatisticalValidator(alpha)
    
    # Prepare experimental data
    experimental_data = {}
    for metric in control_data.keys():
        if metric in treatment_data:
            experimental_data[metric] = ExperimentalData(
                control_group=control_data[metric],
                treatment_group=treatment_data[metric],
                metric_name=metric
            )
    
    # Perform comprehensive validation
    results = validator.validate_experiment(experimental_data)
    
    return results

# Example usage and testing functions
def generate_sample_data() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate sample data for testing purposes."""
    np.random.seed(42)
    
    # Control group data (traditional methods)
    control_data = {
        'completeness': np.random.normal(78.4, 12.3, 30),
        'precision': np.random.normal(0.76, 0.08, 30),
        'recall': np.random.normal(0.62, 0.10, 30),
        'satisfaction': np.random.normal(4.8, 1.2, 30)
    }
    
    # Treatment group data (multimedia methods) - with improvements
    treatment_data = {
        'completeness': np.random.normal(96.8, 14.2, 30),
        'precision': np.random.normal(0.82, 0.07, 30),
        'recall': np.random.normal(0.76, 0.11, 30),
        'satisfaction': np.random.normal(6.1, 0.9, 30)
    }
    
    # Ensure positive values where appropriate
    for metric in ['completeness', 'satisfaction']:
        control_data[metric] = np.maximum(control_data[metric], 0)
        treatment_data[metric] = np.maximum(treatment_data[metric], 0)
    
    for metric in ['precision', 'recall']:
        control_data[metric] = np.clip(control_data[metric], 0, 1)
        treatment_data[metric] = np.clip(treatment_data[metric], 0, 1)
    
    return control_data, treatment_data

if __name__ == "__main__":
    # Example usage
    print("Testing Statistical Validation Module")
    print("=" * 50)
    
    # Generate sample data
    control_data, treatment_data = generate_sample_data()
    
    # Run validation
    results = validate_multimedia_re_results(control_data, treatment_data)
    
    # Print summary
    print("Validation Results Summary:")
    print("-" * 30)
    for metric, significance in results['summary']['significant_results'].items():
        print(f"{metric}: {significance}")
    
    print(f"\nOverall Conclusion:")
    print(results['overall_conclusion'])
