#!/usr/bin/env python3
"""
Data Validation Script for Requirements Engineering Dataset
==========================================================

This script validates the generated dataset to ensure it meets the specifications
from the empirical study and maintains statistical integrity.

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University in Zlin
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class DatasetValidator:
    """
    Comprehensive validator for the Requirements Engineering dataset.
    """
    
    def __init__(self, data_dir: str = "generated_data"):
        self.data_dir = data_dir
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
        # Expected values from the paper
        self.expected_values = {
            'total_participants': 60,
            'control_group_size': 30,
            'treatment_group_size': 30,
            'total_requirements': 127,
            'functional_requirements': 73,
            'non_functional_requirements': 54,
            'institutions_count': 3,
            'stakeholder_types_count': 3,
            
            # Statistical targets (approximate)
            'control_mean_requirements': 78.4,
            'treatment_mean_requirements': 96.8,
            'improvement_percentage': 23.5,
            'control_satisfaction_mean': 4.8,
            'treatment_satisfaction_mean': 6.1
        }

    def validate_file_structure(self) -> bool:
        """Validate that all required files exist."""
        
        required_files = [
            'participants.csv',
            'ground_truth_requirements.csv',
            'participant_results.csv',
            'audio_analysis.csv',
            'video_analysis.csv',
            'image_analysis.csv',
            'cost_analysis.csv',
            'summary_statistics.json',
            'dataset_metadata.json'
        ]
        
        print("🔍 Validating file structure...")
        
        missing_files = []
        existing_files = []
        
        for file in required_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                existing_files.append(file)
                # Check if file is not empty
                if os.path.getsize(file_path) == 0:
                    self.errors.append(f"File {file} exists but is empty")
            else:
                missing_files.append(file)
        
        if missing_files:
            self.errors.append(f"Missing files: {missing_files}")
            print(f"❌ Missing files: {missing_files}")
        else:
            print(f"✅ All {len(required_files)} required files present")
        
        self.validation_results['file_structure'] = {
            'passed': len(missing_files) == 0,
            'existing_files': existing_files,
            'missing_files': missing_files
        }
        
        return len(missing_files) == 0

    def validate_participants_data(self) -> bool:
        """Validate participants.csv structure and content."""
        
        print("🔍 Validating participants data...")
        
        try:
            participants = pd.read_csv(os.path.join(self.data_dir, 'participants.csv'))
        except Exception as e:
            self.errors.append(f"Cannot load participants.csv: {e}")
            return False
        
        issues = []
        
        # Check row count
        if len(participants) != self.expected_values['total_participants']:
            issues.append(f"Expected {self.expected_values['total_participants']} participants, got {len(participants)}")
        
        # Check required columns
        required_columns = [
            'participant_id', 'institution', 'stakeholder_type', 
            'group_assignment', 'age', 'gender', 'experience_years',
            'consent_given', 'session_date', 'anonymized_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in participants.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check group distribution
        if 'group_assignment' in participants.columns:
            group_counts = participants['group_assignment'].value_counts()
            control_count = group_counts.get('Control', 0)
            treatment_count = group_counts.get('Treatment', 0)
            
            if control_count != self.expected_values['control_group_size']:
                issues.append(f"Control group: expected {self.expected_values['control_group_size']}, got {control_count}")
            
            if treatment_count != self.expected_values['treatment_group_size']:
                issues.append(f"Treatment group: expected {self.expected_values['treatment_group_size']}, got {treatment_count}")
        
        # Check institutions
        if 'institution' in participants.columns:
            unique_institutions = participants['institution'].nunique()
            if unique_institutions != self.expected_values['institutions_count']:
                issues.append(f"Expected {self.expected_values['institutions_count']} institutions, got {unique_institutions}")
        
        # Check stakeholder types
        if 'stakeholder_type' in participants.columns:
            unique_types = participants['stakeholder_type'].nunique()
            if unique_types != self.expected_values['stakeholder_types_count']:
                issues.append(f"Expected {self.expected_values['stakeholder_types_count']} stakeholder types, got {unique_types}")
        
        # Check data integrity
        if 'age' in participants.columns:
            age_range = participants['age'].min(), participants['age'].max()
            if age_range[0] < 18 or age_range[1] > 70:
                self.warnings.append(f"Age range seems unusual: {age_range}")
        
        # Check for duplicates
        if 'participant_id' in participants.columns:
            if participants['participant_id'].duplicated().any():
                issues.append("Duplicate participant IDs found")
        
        if issues:
            self.errors.extend(issues)
            print(f"❌ Participants validation failed: {len(issues)} issues")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Participants data validation passed")
        
        self.validation_results['participants'] = {
            'passed': len(issues) == 0,
            'row_count': len(participants),
            'issues': issues
        }
        
        return len(issues) == 0

    def validate_ground_truth_requirements(self) -> bool:
        """Validate ground_truth_requirements.csv structure and content."""
        
        print("🔍 Validating ground truth requirements...")
        
        try:
            requirements = pd.read_csv(os.path.join(self.data_dir, 'ground_truth_requirements.csv'))
        except Exception as e:
            self.errors.append(f"Cannot load ground_truth_requirements.csv: {e}")
            return False
        
        issues = []
        
        # Check total count
        if len(requirements) != self.expected_values['total_requirements']:
            issues.append(f"Expected {self.expected_values['total_requirements']} requirements, got {len(requirements)}")
        
        # Check requirement types
        if 'type' in requirements.columns:
            type_counts = requirements['type'].value_counts()
            functional_count = type_counts.get('Functional', 0)
            non_functional_count = type_counts.get('Non-Functional', 0)
            
            if functional_count != self.expected_values['functional_requirements']:
                issues.append(f"Functional requirements: expected {self.expected_values['functional_requirements']}, got {functional_count}")
            
            if non_functional_count != self.expected_values['non_functional_requirements']:
                issues.append(f"Non-functional requirements: expected {self.expected_values['non_functional_requirements']}, got {non_functional_count}")
        
        # Check required columns
        required_columns = ['requirement_id', 'type', 'category', 'priority', 'complexity', 'description', 'expert_consensus']
        missing_columns = [col for col in required_columns if col not in requirements.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check data ranges
        if 'expert_consensus' in requirements.columns:
            consensus_range = requirements['expert_consensus'].min(), requirements['expert_consensus'].max()
            if consensus_range[0] < 0 or consensus_range[1] > 1:
                issues.append(f"Expert consensus should be 0-1, got range: {consensus_range}")
        
        # Check for duplicates
        if 'requirement_id' in requirements.columns:
            if requirements['requirement_id'].duplicated().any():
                issues.append("Duplicate requirement IDs found")
        
        if issues:
            self.errors.extend(issues)
            print(f"❌ Ground truth validation failed: {len(issues)} issues")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Ground truth requirements validation passed")
        
        self.validation_results['ground_truth'] = {
            'passed': len(issues) == 0,
            'total_requirements': len(requirements),
            'functional_count': type_counts.get('Functional', 0) if 'type' in requirements.columns else 0,
            'non_functional_count': type_counts.get('Non-Functional', 0) if 'type' in requirements.columns else 0,
            'issues': issues
        }
        
        return len(issues) == 0

    def validate_participant_results(self) -> bool:
        """Validate participant_results.csv structure and statistical properties."""
        
        print("🔍 Validating participant results...")
        
        try:
            results = pd.read_csv(os.path.join(self.data_dir, 'participant_results.csv'))
        except Exception as e:
            self.errors.append(f"Cannot load participant_results.csv: {e}")
            return False
        
        issues = []
        
        # Check row count matches participants
        if len(results) != self.expected_values['total_participants']:
            issues.append(f"Results count doesn't match participants: expected {self.expected_values['total_participants']}, got {len(results)}")
        
        # Check required columns
        required_columns = [
            'participant_id', 'group_assignment', 'requirements_identified',
            'functional_requirements', 'non_functional_requirements',
            'completeness_score', 'precision', 'recall', 'f1_score',
            'satisfaction_score', 'total_time_min'
        ]
        
        missing_columns = [col for col in required_columns if col not in results.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Statistical validation
        if 'group_assignment' in results.columns and 'requirements_identified' in results.columns:
            control_group = results[results['group_assignment'] == 'Control']
            treatment_group = results[results['group_assignment'] == 'Treatment']
            
            if len(control_group) == 0 or len(treatment_group) == 0:
                issues.append("One or both groups are empty")
            else:
                # Check means are approximately correct
                control_mean = control_group['requirements_identified'].mean()
                treatment_mean = treatment_group['requirements_identified'].mean()
                
                # Allow 10% tolerance
                control_tolerance = abs(control_mean - self.expected_values['control_mean_requirements']) / self.expected_values['control_mean_requirements']
                treatment_tolerance = abs(treatment_mean - self.expected_values['treatment_mean_requirements']) / self.expected_values['treatment_mean_requirements']
                
                if control_tolerance > 0.1:
                    self.warnings.append(f"Control group mean requirements ({control_mean:.1f}) differs significantly from expected ({self.expected_values['control_mean_requirements']})")
                
                if treatment_tolerance > 0.1:
                    self.warnings.append(f"Treatment group mean requirements ({treatment_mean:.1f}) differs significantly from expected ({self.expected_values['treatment_mean_requirements']})")
                
                # Check improvement percentage
                actual_improvement = ((treatment_mean - control_mean) / control_mean) * 100
                expected_improvement = self.expected_values['improvement_percentage']
                improvement_tolerance = abs(actual_improvement - expected_improvement) / expected_improvement
                
                if improvement_tolerance > 0.1:
                    self.warnings.append(f"Improvement percentage ({actual_improvement:.1f}%) differs from expected ({expected_improvement}%)")
        
        # Check data ranges
        numeric_columns_ranges = {
            'completeness_score': (0, 1),
            'precision': (0, 1),
            'recall': (0, 1),
            'f1_score': (0, 1),
            'satisfaction_score': (1, 7)
        }
        
        for col, (min_val, max_val) in numeric_columns_ranges.items():
            if col in results.columns:
                col_min, col_max = results[col].min(), results[col].max()
                if col_min < min_val or col_max > max_val:
                    issues.append(f"{col} values out of range: expected [{min_val}, {max_val}], got [{col_min:.3f}, {col_max:.3f}]")
        
        if issues:
            self.errors.extend(issues)
            print(f"❌ Participant results validation failed: {len(issues)} issues")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Participant results validation passed")
        
        self.validation_results['participant_results'] = {
            'passed': len(issues) == 0,
            'issues': issues
        }
        
        return len(issues) == 0

    def validate_multimedia_data(self) -> bool:
        """Validate multimedia analysis files."""
        
        print("🔍 Validating multimedia data...")
        
        multimedia_files = ['audio_analysis.csv', 'video_analysis.csv', 'image_analysis.csv']
        all_passed = True
        
        for file in multimedia_files:
            try:
                df = pd.read_csv(os.path.join(self.data_dir, file))
                
                # Check if all participant_ids are from treatment group
                if 'participant_id' in df.columns:
                    unique_participants = df['participant_id'].nunique()
                    # Should have data for treatment group only (30 participants)
                    if unique_participants > 30:
                        self.warnings.append(f"{file}: More participants than expected treatment group size")
                
                # Check confidence scores are in valid range
                if 'confidence_score' in df.columns:
                    conf_min, conf_max = df['confidence_score'].min(), df['confidence_score'].max()
                    if conf_min < 0 or conf_max > 1:
                        self.errors.append(f"{file}: Confidence scores out of range [0,1]: [{conf_min:.3f}, {conf_max:.3f}]")
                        all_passed = False
                
                # Check for negative processing times
                if 'processing_time_seconds' in df.columns:
                    if (df['processing_time_seconds'] < 0).any():
                        self.errors.append(f"{file}: Negative processing times found")
                        all_passed = False
                
                print(f"✅ {file}: {len(df)} records validated")
                
            except Exception as e:
                self.errors.append(f"Cannot load {file}: {e}")
                all_passed = False
                print(f"❌ {file}: Validation failed")
        
        self.validation_results['multimedia_data'] = {
            'passed': all_passed
        }
        
        return all_passed

    def validate_statistical_integrity(self) -> bool:
        """Validate statistical properties match the paper."""
        
        print("🔍 Validating statistical integrity...")
        
        try:
            results = pd.read_csv(os.path.join(self.data_dir, 'participant_results.csv'))
        except Exception as e:
            self.errors.append(f"Cannot load participant results for statistical validation: {e}")
            return False
        
        issues = []
        
        if 'group_assignment' not in results.columns:
            issues.append("Cannot perform statistical validation: missing group_assignment column")
            return False
        
        control_group = results[results['group_assignment'] == 'Control']
        treatment_group = results[results['group_assignment'] == 'Treatment']
        
        # Perform t-tests for key metrics
        statistical_tests = {
            'requirements_identified': 'Requirements identification',
            'precision': 'Precision',
            'satisfaction_score': 'Satisfaction score'
        }
        
        for metric, description in statistical_tests.items():
            if metric in results.columns:
                try:
                    t_stat, p_value = stats.ttest_ind(
                        treatment_group[metric], 
                        control_group[metric]
                    )
                    
                    # Check if p-value indicates significance (should be < 0.05)
                    if p_value >= 0.05:
                        self.warnings.append(f"{description}: Not statistically significant (p={p_value:.4f})")
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(control_group) - 1) * control_group[metric].var() +
                                         (len(treatment_group) - 1) * treatment_group[metric].var()) /
                                        (len(control_group) + len(treatment_group) - 2))
                    
                    cohens_d = (treatment_group[metric].mean() - control_group[metric].mean()) / pooled_std
                    
                    # Effect size should be large (> 0.8) for main findings
                    if metric == 'requirements_identified' and abs(cohens_d) < 0.8:
                        self.warnings.append(f"{description}: Effect size smaller than expected (d={cohens_d:.3f})")
                    
                except Exception as e:
                    issues.append(f"Statistical test failed for {metric}: {e}")
        
        # Check normality assumptions
        for metric in ['requirements_identified', 'satisfaction_score']:
            if metric in results.columns:
                for group_name, group_data in [('Control', control_group), ('Treatment', treatment_group)]:
                    if len(group_data) > 3:  # Need minimum data for normality test
                        _, p_norm = stats.shapiro(group_data[metric])
                        if p_norm < 0.05:
                            self.warnings.append(f"{metric} in {group_name} group may not be normally distributed (p={p_norm:.4f})")
        
        if issues:
            self.errors.extend(issues)
            print(f"❌ Statistical validation failed: {len(issues)} issues")
        else:
            print("✅ Statistical integrity validation passed")
        
        self.validation_results['statistical_integrity'] = {
            'passed': len(issues) == 0,
            'issues': issues
        }
        
        return len(issues) == 0

    def validate_metadata(self) -> bool:
        """Validate metadata files."""
        
        print("🔍 Validating metadata...")
        
        metadata_files = ['summary_statistics.json', 'dataset_metadata.json']
        all_passed = True
        
        for file in metadata_files:
            try:
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    data = json.load(f)
                
                if not data:
                    self.errors.append(f"{file} is empty")
                    all_passed = False
                else:
                    print(f"✅ {file}: Valid JSON with {len(data)} top-level keys")
                
            except json.JSONDecodeError as e:
                self.errors.append(f"{file}: Invalid JSON - {e}")
                all_passed = False
                print(f"❌ {file}: Invalid JSON")
            except Exception as e:
                self.errors.append(f"Cannot load {file}: {e}")
                all_passed = False
                print(f"❌ {file}: Cannot load")
        
        self.validation_results['metadata'] = {
            'passed': all_passed
        }
        
        return all_passed

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        
        print("="*60)
        print("REQUIREMENTS ENGINEERING DATASET VALIDATION")
        print("="*60)
        print(f"Validating dataset in: {self.data_dir}")
        print()
        
        # Run all validation checks
        validation_steps = [
            ("File Structure", self.validate_file_structure),
            ("Participants Data", self.validate_participants_data),
            ("Ground Truth Requirements", self.validate_ground_truth_requirements),
            ("Participant Results", self.validate_participant_results),
            ("Multimedia Data", self.validate_multimedia_data),
            ("Statistical Integrity", self.validate_statistical_integrity),
            ("Metadata", self.validate_metadata)
        ]
        
        passed_count = 0
        total_count = len(validation_steps)
        
        for step_name, validation_func in validation_steps:
            try:
                if validation_func():
                    passed_count += 1
            except Exception as e:
                self.errors.append(f"Validation step '{step_name}' failed with exception: {e}")
                print(f"❌ {step_name}: Exception occurred")
        
        print()
        print("="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        overall_passed = passed_count == total_count and len(self.errors) == 0
        
        print(f"Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print(f"Validation Steps: {passed_count}/{total_count} passed")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n🚨 ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Save validation report
        validation_report = {
            'overall_passed': overall_passed,
            'validation_date': pd.Timestamp.now().isoformat(),
            'steps_passed': passed_count,
            'total_steps': total_count,
            'errors': self.errors,
            'warnings': self.warnings,
            'detailed_results': self.validation_results
        }
        
        report_file = os.path.join(self.data_dir, 'validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"\n📄 Validation report saved to: {report_file}")
        
        return validation_report

def main():
    """Main function to run validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Requirements Engineering Dataset')
    parser.add_argument('--data-dir', default='generated_data', 
                       help='Directory containing the dataset (default: generated_data)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory '{args.data_dir}' does not exist!")
        print("Make sure to generate the dataset first using dataset_generator.py")
        return
    
    # Run validation
    validator = DatasetValidator(args.data_dir)
    report = validator.run_full_validation()
    
    # Exit with appropriate code
    exit_code = 0 if report['overall_passed'] else 1
    exit(exit_code)

if __name__ == "__main__":
    main()
