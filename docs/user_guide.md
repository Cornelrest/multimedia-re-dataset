# User Guide: Requirements Engineering Multimedia Dataset

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Dataset Overview](#dataset-overview)
4. [Data Generation](#data-generation)
5. [Data Analysis](#data-analysis)
6. [Statistical Analysis](#statistical-analysis)
7. [Visualization](#visualization)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Introduction

This guide provides comprehensive instructions for using the Requirements Engineering Multimedia Dataset, generated for the empirical study "Knowledge Extraction from Multimedia Data in Requirements Engineering: An Empirical Study."

### What You'll Learn

- How to generate and load the complete dataset
- Perform statistical analysis matching the paper's findings
- Create visualizations and conduct custom analyses
- Extend the framework for your own research

### Prerequisites

- Python 3.8 or higher
- Basic knowledge of pandas and data analysis
- Understanding of requirements engineering concepts

## Installation & Setup

### Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv re_dataset_env
source re_dataset_env/bin/activate  # On Windows: re_dataset_env\Scripts\activate

# Install required packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Step 2: Download Repository

```bash
# Clone the repository
git clone https://github.com/multimedia-re-study/dataset.git
cd dataset

# Or download ZIP and extract
```

### Step 3: Verify Installation

```python
import pandas as pd
import numpy as np
print("Setup complete! Ready to generate dataset.")
```

## Dataset Overview

### File Structure After Generation

```
generated_data/
├── participants.csv                    # 60 participants with demographics
├── ground_truth_requirements.csv      # 127 expert-validated requirements
├── participant_results.csv            # Performance metrics per participant
├── audio_analysis.csv                 # Audio processing results (treatment group only)
├── video_analysis.csv                 # Video analysis results (treatment group only)
├── image_analysis.csv                 # Image processing results (treatment group only)
├── cost_analysis.csv                  # Economic analysis data
├── summary_statistics.json            # Key statistical findings
└── dataset_metadata.json              # Complete metadata
```

### Key Metrics Included

| Category | Metrics |
|----------|---------|
| **Completeness** | Requirements identified, completeness percentage |
| **Accuracy** | Precision, recall, F1-score |
| **Satisfaction** | 7-point Likert scale scores |
| **Efficiency** | Processing times, cost analysis |
| **Types** | Functional vs. non-functional requirements |

## Data Generation

### Basic Generation

```python
from dataset_generator import RequirementsDatasetGenerator

# Initialize generator
generator = RequirementsDatasetGenerator()

# Generate complete dataset
generator.generate_complete_dataset()
```

### Custom Generation

```python
# Create generator with custom parameters
generator = RequirementsDatasetGenerator()

# Modify statistical parameters if needed
generator.control_stats['mean_requirements'] = 75.0  # Custom mean
generator.treatment_stats['mean_requirements'] = 95.0

# Generate with custom output directory
generator.generate_complete_dataset(output_dir="custom_data")
```

### Generated Data Validation

```python
import pandas as pd
import os

def validate_dataset(data_dir="generated_data"):
    """Validate generated dataset structure and content."""
    
    required_files = [
        'participants.csv',
        'ground_truth_requirements.csv',
        'participant_results.csv',
        'audio_analysis.csv',
        'video_analysis.csv',
        'image_analysis.csv'
    ]
    
    print("Dataset Validation Report")
    print("=" * 40)
    
    for file in required_files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ {file}: {len(df)} records")
        else:
            print(f"❌ {file}: Missing")
    
    # Validate participant distribution
    participants = pd.read_csv(os.path.join(data_dir, 'participants.csv'))
    group_counts = participants['group_assignment'].value_counts()
    print(f"\nGroup Distribution:")
    print(f"  Control: {group_counts.get('Control', 0)}")
    print(f"  Treatment: {group_counts.get('Treatment', 0)}")

# Run validation
validate_dataset()
```

## Data Analysis

### Loading Data

```python
import pandas as pd
import numpy as np

# Load main datasets
participants = pd.read_csv('generated_data/participants.csv')
results = pd.read_csv('generated_data/participant_results.csv')
ground_truth = pd.read_csv('generated_data/ground_truth_requirements.csv')

# Display basic info
print(f"Participants: {len(participants)}")
print(f"Results: {len(results)}")
print(f"Ground truth requirements: {len(ground_truth)}")
```

### Basic Analysis

```python
# Split by experimental groups
control_group = results[results['group_assignment'] == 'Control']
treatment_group = results[results['group_assignment'] == 'Treatment']

# Calculate means and standard deviations
def group_statistics(group_data, group_name):
    stats = {
        'requirements_mean': group_data['requirements_identified'].mean(),
        'requirements_std': group_data['requirements_identified'].std(),
        'precision_mean': group_data['precision'].mean(),
        'recall_mean': group_data['recall'].mean(),
        'satisfaction_mean': group_data['satisfaction_score'].mean(),
        'satisfaction_std': group_data['satisfaction_score'].std()
    }
    
    print(f"{group_name} Group Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    print()
    
    return stats

control_stats = group_statistics(control_group, "Control")
treatment_stats = group_statistics(treatment_group, "Treatment")
```

### Improvement Calculations

```python
# Calculate improvements (matching paper findings)
def calculate_improvements(control_stats, treatment_stats):
    improvements = {}
    
    # Requirements completeness improvement
    req_improvement = ((treatment_stats['requirements_mean'] - 
                       control_stats['requirements_mean']) / 
                      control_stats['requirements_mean']) * 100
    
    # Precision improvement
    precision_improvement = ((treatment_stats['precision_mean'] - 
                            control_stats['precision_mean']) / 
                           control_stats['precision_mean']) * 100
    
    # Satisfaction improvement
    satisfaction_improvement = ((treatment_stats['satisfaction_mean'] - 
                               control_stats['satisfaction_mean']) / 
                              control_stats['satisfaction_mean']) * 100
    
    improvements = {
        'Requirements': f"{req_improvement:.1f}%",
        'Precision': f"{precision_improvement:.1f}%",
        'Satisfaction': f"{satisfaction_improvement:.1f}%"
    }
    
    print("Improvements (Treatment vs Control):")
    for metric, improvement in improvements.items():
        print(f"  {metric}: {improvement}")
    
    return improvements

improvements = calculate_improvements(control_stats, treatment_stats)
```

### Requirement Type Analysis

```python
# Analyze functional vs non-functional requirements
def analyze_requirement_types(results_df):
    control = results_df[results_df['group_assignment'] == 'Control']
    treatment = results_df[results_df['group_assignment'] == 'Treatment']
    
    # Total requirements by type
    analysis = {
        'functional': {
            'control_total': control['functional_requirements'].sum(),
            'treatment_total': treatment['functional_requirements'].sum(),
            'control_mean': control['functional_requirements'].mean(),
            'treatment_mean': treatment['functional_requirements'].mean()
        },
        'non_functional': {
            'control_total': control['non_functional_requirements'].sum(),
            'treatment_total': treatment['non_functional_requirements'].sum(),
            'control_mean': control['non_functional_requirements'].mean(),
            'treatment_mean': treatment['non_functional_requirements'].mean()
        }
    }
    
    # Calculate improvements
    func_improvement = ((analysis['functional']['treatment_mean'] - 
                        analysis['functional']['control_mean']) / 
                       analysis['functional']['control_mean']) * 100
    
    nf_improvement = ((analysis['non_functional']['treatment_mean'] - 
                      analysis['non_functional']['control_mean']) / 
                     analysis['non_functional']['control_mean']) * 100
    
    print("Requirement Type Analysis:")
    print(f"  Functional improvement: {func_improvement:.1f}%")
    print(f"  Non-functional improvement: {nf_improvement:.1f}%")
    
    return analysis

req_type_analysis = analyze_requirement_types(results)
```

### Multimedia Data Analysis

```python
# Analyze multimedia processing results (treatment group only)
def analyze_multimedia_data():
    audio_data = pd.read_csv('generated_data/audio_analysis.csv')
    video_data = pd.read_csv('generated_data/video_analysis.csv')
    image_data = pd.read_csv('generated_data/image_analysis.csv')
    
    multimedia_stats = {
        'audio': {
            'total_segments': len(audio_data),
            'avg_confidence': audio_data['confidence_score'].mean(),
            'avg_requirements_per_segment': audio_data['requirements_extracted'].mean(),
            'avg_processing_time': audio_data['processing_time_seconds'].mean()
        },
        'video': {
            'total_segments': len(video_data),
            'avg_confidence': video_data['confidence_score'].mean(),
            'avg_requirements_per_segment': video_data['requirements_extracted'].mean(),
            'avg_interactions': video_data['interactions_detected'].mean()
        },
        'image': {
            'total_images': len(image_data),
            'avg_confidence': image_data['confidence_score'].mean(),
            'avg_requirements_per_image': image_data['requirements_extracted'].mean(),
            'avg_ocr_accuracy': image_data['ocr_accuracy'].mean()
        }
    }
    
    print("Multimedia Analysis Summary:")
    for data_type, stats in multimedia_stats.items():
        print(f"\n{data_type.upper()} Data:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.3f}")
    
    return multimedia_stats

multimedia_stats = analyze_multimedia_data()
```

## Statistical Analysis

### Hypothesis Testing

```python
from scipy import stats

def perform_statistical_tests(control_group, treatment_group):
    """Perform statistical tests matching the paper."""
    
    tests_results = {}
    
    # T-test for requirements identified
    t_stat, p_value = stats.ttest_ind(
        treatment_group['requirements_identified'],
        control_group['requirements_identified']
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(control_group) - 1) * control_group['requirements_identified'].var() +
                         (len(treatment_group) - 1) * treatment_group['requirements_identified'].var()) /
                        (len(control_group) + len(treatment_group) - 2))
    
    cohens_d = (treatment_group['requirements_identified'].mean() - 
                control_group['requirements_identified'].mean()) / pooled_std
    
    tests_results['requirements'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
    }
    
    # Similar tests for other metrics
    for metric in ['precision', 'recall', 'f1_score', 'satisfaction_score']:
        t_stat, p_value = stats.ttest_ind(
            treatment_group[metric], control_group[metric]
        )
        tests_results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    # Print results
    print("Statistical Test Results:")
    print("=" * 40)
    for metric, results in tests_results.items():
        print(f"\n{metric.upper()}:")
        print(f"  t-statistic: {results['t_statistic']:.3f}")
        print(f"  p-value: {results['p_value']:.6f}")
        if 'cohens_d' in results:
            print(f"  Cohen's d: {results['cohens_d']:.3f} ({results['effect_size']} effect)")
    
    return tests_results

statistical_results = perform_statistical_tests(control_group, treatment_group)
```

### Power Analysis

```python
from scipy.stats import norm

def calculate_power_analysis(control_group, treatment_group, alpha=0.05):
    """Calculate statistical power for the study design."""
    
    # Calculate effect size
    pooled_std = np.sqrt(((len(control_group) - 1) * control_group['requirements_identified'].var() +
                         (len(treatment_group) - 1) * treatment_group['requirements_identified'].var()) /
                        (len(control_group) + len(treatment_group) - 2))
    
    effect_size = (treatment_group['requirements_identified'].mean() - 
                   control_group['requirements_identified'].mean()) / pooled_std
    
    # Calculate power
    n_per_group = len(control_group)
    standard_error = pooled_std * np.sqrt(2 / n_per_group)
    critical_value = norm.ppf(1 - alpha/2) * standard_error
    
    # Power calculation
    beta = norm.cdf(critical_value - effect_size * pooled_std) + norm.cdf(-critical_value - effect_size * pooled_std)
    power = 1 - beta
    
    print("Power Analysis:")
    print(f"  Effect size (Cohen's d): {effect_size:.3f}")
    print(f"  Sample size per group: {n_per_group}")
    print(f"  Statistical power: {power:.3f}")
    print(f"  Alpha level: {alpha}")
    
    return power, effect_size

power, effect_size = calculate_power_analysis(control_group, treatment_group)
```

## Visualization

### Basic Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_comparison_plots(results_df):
    """Create comparison plots for key metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Control vs Treatment Group Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('requirements_identified', 'Requirements Identified'),
        ('precision', 'Precision'),
        ('satisfaction_score', 'Satisfaction Score'),
        ('total_time_min', 'Total Time (minutes)')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Box plot
        sns.boxplot(data=results_df, x='group_assignment', y=metric, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Group')
        
        # Add mean lines
        control_mean = results_df[results_df['group_assignment'] == 'Control'][metric].mean()
        treatment_mean = results_df[results_df['group_assignment'] == 'Treatment'][metric].mean()
        
        ax.axhline(y=control_mean, color='red', linestyle='--', alpha=0.7, label=f'Control Mean: {control_mean:.2f}')
        ax.axhline(y=treatment_mean, color='blue', linestyle='--', alpha=0.7, label=f'Treatment Mean: {treatment_mean:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('generated_data/comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

create_comparison_plots(results)
```

### Advanced Visualizations

```python
def create_satisfaction_distribution():
    """Create satisfaction score distribution plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    control_satisfaction = control_group['satisfaction_score']
    treatment_satisfaction = treatment_group['satisfaction_score']
    
    ax1.hist(control_satisfaction, alpha=0.7, label='Control', bins=10, color='red')
    ax1.hist(treatment_satisfaction, alpha=0.7, label='Treatment', bins=10, color='blue')
    ax1.set_xlabel('Satisfaction Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Satisfaction Score Distribution')
    ax1.legend()
    
    # Box plot with violin
    data_for_plot = pd.concat([
        pd.DataFrame({'Score': control_satisfaction, 'Group': 'Control'}),
        pd.DataFrame({'Score': treatment_satisfaction, 'Group': 'Treatment'})
    ])
    
    sns.violinplot(data=data_for_plot, x='Group', y='Score', ax=ax2)
    ax2.set_title('Satisfaction Score Distribution (Violin Plot)')
    
    plt.tight_layout()
    plt.savefig('generated_data/satisfaction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

create_satisfaction_distribution()
```

### Performance Metrics Visualization

```python
def create_performance_radar_chart():
    """Create radar chart comparing performance metrics."""
    
    from math import pi
    
    # Metrics for radar chart
    metrics = ['Precision', 'Recall', 'F1-Score', 'Completeness', 'Satisfaction']
    
    control_values = [
        control_group['precision'].mean(),
        control_group['recall'].mean(),
        control_group['f1_score'].mean(),
        control_group['completeness_score'].mean(),
        control_group['satisfaction_score'].mean() / 7  # Normalize to 0-1
    ]
    
    treatment_values = [
        treatment_group['precision'].mean(),
        treatment_group['recall'].mean(),
        treatment_group['f1_score'].mean(),
        treatment_group['completeness_score'].mean(),
        treatment_group['satisfaction_score'].mean() / 7  # Normalize to 0-1
    ]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angles
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot control group
    control_values += control_values[:1]  # Complete the circle
    ax.plot(angles, control_values, 'o-', linewidth=2, label='Control', color='red')
    ax.fill(angles, control_values, alpha=0.25, color='red')
    
    # Plot treatment group
    treatment_values += treatment_values[:1]  # Complete the circle
    ax.plot(angles, treatment_values, 'o-', linewidth=2, label='Treatment', color='blue')
    ax.fill(angles, treatment_values, alpha=0.25, color='blue')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Metrics Comparison\n(All metrics normalized to 0-1 scale)', 
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.savefig('generated_data/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

create_performance_radar_chart()
```

## Advanced Usage

### Custom Analysis Functions

```python
def analyze_stakeholder_differences(results_df):
    """Analyze performance differences by stakeholder type."""
    
    stakeholder_analysis = results_df.groupby(['stakeholder_type', 'group_assignment']).agg({
        'requirements_identified': ['mean', 'std'],
        'satisfaction_score': ['mean', 'std'],
        'precision': 'mean'
    }).round(3)
    
    print("Stakeholder Type Analysis:")
    print(stakeholder_analysis)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, stakeholder in enumerate(['Educator', 'Student', 'Administrator']):
        subset = results_df[results_df['stakeholder_type'] == stakeholder]
        sns.boxplot(data=subset, x='group_assignment', y='requirements_identified', ax=axes[idx])
        axes[idx].set_title(f'{stakeholder} - Requirements Identified')
    
    plt.tight_layout()
    plt.savefig('generated_data/stakeholder_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stakeholder_analysis

stakeholder_results = analyze_stakeholder_differences(results)
```

### Time Series Analysis

```python
def analyze_processing_efficiency(results_df):
    """Analyze processing time efficiency."""
    
    # Time analysis
    time_comparison = results_df.groupby('group_assignment').agg({
        'data_collection_time_min': ['mean', 'std'],
        'analysis_time_min': ['mean', 'std'],
        'validation_time_min': ['mean', 'std'],
        'total_time_min': ['mean', 'std']
    }).round(2)
    
    print("Processing Time Analysis:")
    print(time_comparison)
    
    # Calculate time savings
    control_total = results_df[results_df['group_assignment'] == 'Control']['total_time_min'].mean()
    treatment_total = results_df[results_df['group_assignment'] == 'Treatment']['total_time_min'].mean()
    time_savings = ((control_total - treatment_total) / control_total) * 100
    
    print(f"\nTime Savings: {time_savings:.1f}%")
    
    # Create stacked bar chart
    time_data = {
        'Control': [
            results_df[results_df['group_assignment'] == 'Control']['data_collection_time_min'].mean(),
            results_df[results_df['group_assignment'] == 'Control']['analysis_time_min'].mean(),
            0  # No validation time for control
        ],
        'Treatment': [
            results_df[results_df['group_assignment'] == 'Treatment']['data_collection_time_min'].mean(),
            results_df[results_df['group_assignment'] == 'Treatment']['analysis_time_min'].mean(),
            results_df[results_df['group_assignment'] == 'Treatment']['validation_time_min'].mean()
        ]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Data Collection', 'Analysis', 'Validation']
    x = np.arange(len(time_data))
    width = 0.35
    
    bottom_control = np.zeros(2)
    bottom_treatment = np.zeros(2)
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, category in enumerate(categories):
        control_vals = [time_data['Control'][i], 0] if i < len(time_data['Control']) else [0, 0]
        treatment_vals = [0, time_data['Treatment'][i]] if i < len(time_data['Treatment']) else [0, 0]
        
        ax.bar(['Control', 'Treatment'], [time_data['Control'][i], time_data['Treatment'][i]], 
               label=category, color=colors[i])
    
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Processing Time Breakdown by Group')
    ax.legend()
    
    plt.savefig('generated_data/time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return time_comparison

time_analysis = analyze_processing_efficiency(results)
```

### Cost-Benefit Analysis

```python
def perform_cost_benefit_analysis():
    """Perform detailed cost-benefit analysis."""
    
    cost_data = pd.read_csv('generated_data/cost_analysis.csv')
    
    # Calculate totals
    setup_costs = cost_data[cost_data['category'] == 'Setup Cost']['amount_usd'].sum()
    project_savings = cost_data[cost_data['category'] == 'Per-Project Savings']['amount_usd'].sum()
    
    # Break-even analysis
    break_even_projects = setup_costs / project_savings
    
    print("Cost-Benefit Analysis:")
    print(f"  Total setup costs: ${setup_costs:,}")
    print(f"  Savings per project: ${project_savings:,}")
    print(f"  Break-even point: {break_even_projects:.1f} projects")
    
    # ROI calculation for different project volumes
    project_volumes = range(1, 11)
    roi_values = []
    
    for projects in project_volumes:
        total_savings = projects * project_savings
        roi = ((total_savings - setup_costs) / setup_costs) * 100
        roi_values.append(roi)
    
    # Plot ROI
    plt.figure(figsize=(10, 6))
    plt.plot(project_volumes, roi_values, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    plt.xlabel('Number of Projects')
    plt.ylabel('ROI (%)')
    plt.title('Return on Investment by Project Volume')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Highlight break-even point
    break_even_idx = next(i for i, roi in enumerate(roi_values) if roi >= 0)
    plt.scatter(project_volumes[break_even_idx], roi_values[break_even_idx], 
                color='green', s=100, zorder=5, label=f'Break-even at {project_volumes[break_even_idx]} projects')
    
    plt.savefig('generated_data/roi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cost_data, break_even_projects

cost_analysis, break_even = perform_cost_benefit_analysis()
```

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors
```python
# Problem: ModuleNotFoundError
# Solution: Ensure all dependencies are installed
try:
    import pandas as pd
    import numpy as np
    print("✅ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install pandas numpy matplotlib seaborn scipy")
```

#### Issue 2: Data Generation Fails
```python
# Problem: Dataset generation throws errors
# Solution: Check disk space and permissions
import os

def check_system_requirements():
    # Check available disk space (simplified)
    current_dir = os.getcwd()
    try:
        # Try to create a test file
        test_file = os.path.join(current_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Write permissions OK")
    except Exception as e:
        print(f"❌ Write permission error: {e}")
    
    # Check if output directory exists
    if not os.path.exists('generated_data'):
        try:
            os.makedirs('generated_data')
            print("✅ Created output directory")
        except Exception as e:
            print(f"❌ Cannot create directory: {e}")

check_system_requirements()
```

#### Issue 3: Statistical Tests Fail
```python
# Problem: Statistical tests return unexpected results
# Solution: Validate data integrity
def validate_statistical_data(results_df):
    issues = []
    
    # Check for missing values
    if results_df.isnull().any().any():
        issues.append("Missing values detected")
    
    # Check for reasonable ranges
    if not (0 <= results_df['precision'].max() <= 1):
        issues.append("Precision values out of range")
    
    if not (1 <= results_df['satisfaction_score'].max() <= 7):
        issues.append("Satisfaction scores out of range")
    
    # Check group balance
    group_counts = results_df['group_assignment'].value_counts()
    if abs(group_counts['Control'] - group_counts['Treatment']) > 2:
        issues.append("Unbalanced groups")
    
    if issues:
        print("❌ Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data validation passed")
    
    return len(issues) == 0

validate_statistical_data(results)
```

### Performance Optimization

```python
def optimize_data_loading():
    """Optimize data loading for large datasets."""
    
    # Use efficient data types
    dtype_map = {
        'participant_id': 'category',
        'group_assignment': 'category',
        'stakeholder_type': 'category',
        'institution': 'category'
    }
    
    # Load with optimized types
    participants = pd.read_csv('generated_data/participants.csv', dtype=dtype_map)
    
    print(f"Memory usage optimized: {participants.memory_usage(deep=True).sum()} bytes")
    
    return participants

# Use this for large-scale analysis
optimized_participants = optimize_data_loading()
```

## FAQ

### Q1: How accurate is the synthetic data compared to real data?

**A:** The synthetic data is generated to match the statistical properties reported in the empirical study, including means, standard deviations, and effect sizes. While individual data points are synthetic, the aggregate patterns and statistical relationships are faithful to the original research findings.

### Q2: Can I use this dataset for my own research?

**A:** Yes! The dataset is provided under Creative Commons Attribution 4.0 license. You can use, modify, and distribute it for any purpose, including commercial use, as long as you provide appropriate attribution.

### Q3: How do I cite this dataset?

**A:** Use the BibTeX citation provided in the README.md file. The dataset has a DOI for permanent citation.

### Q4: Can I modify the statistical parameters?

**A:** Absolutely. The `RequirementsDatasetGenerator` class allows you to modify statistical parameters before generation:

```python
generator = RequirementsDatasetGenerator()
generator.control_stats['mean_requirements'] = 80.0  # Custom value
generator.generate_complete_dataset()
```

### Q5: What if I find issues with the data?

**A:** Please open an issue on the GitHub repository with details about the problem. We welcome contributions to improve data quality and analysis capabilities.

### Q6: Can I extend the framework for other domains?

**A:** Yes! The framework is designed to be extensible. You can modify the requirement categories, add new multimedia data types, or adapt the statistical models for different domains.

### Q7: How do I validate my analysis results?

**A:** Use the built-in validation functions and compare your results with the expected values from the paper. The `summary_statistics.json` file contains reference values for verification.

---

For additional support, please contact: okechukwu@utb.cz