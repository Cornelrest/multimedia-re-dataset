#!/usr/bin/env python3
"""
Multimedia Requirements Engineering Dataset Generator

This script generates the complete synthetic dataset that reproduces
the statistical findings from the research paper.

Usage:
    python generate_dataset.py [--output-dir OUTPUT_DIR] [--seed SEED]

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University in Zlin
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimediaREDatasetGenerator:
    """Generate complete multimedia requirements engineering dataset"""
    
    def __init__(self, output_dir=".", seed=42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.set_random_seeds()
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Expected results from paper (for validation)
        self.expected_results = {
            'control': {
                'requirements_identified': (78.4, 12.3),
                'precision': (0.76, 0.08),
                'recall': (0.62, 0.10),
                'f1_score': (0.68, 0.07),
                'satisfaction_score': (4.8, 1.2)
            },
            'treatment': {
                'requirements_identified': (96.8, 14.2),
                'precision': (0.82, 0.07),
                'recall': (0.76, 0.11),
                'f1_score': (0.79, 0.08),
                'satisfaction_score': (6.1, 0.9)
            }
        }
        
    def set_random_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def generate_participants_demographics(self):
        """Generate participant demographics data"""
        logger.info("Generating participant demographics...")
        
        participants = []
        participant_id = 1
        
        # Generate 20 educators
        for i in range(20):
            participants.append({
                'participant_id': f'EDU_{participant_id:03d}',
                'role': 'educator',
                'age': np.random.randint(28, 55),
                'gender': np.random.choice(['M', 'F'], p=[0.45, 0.55]),
                'years_experience': np.random.randint(3, 16),
                'technical_background': np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]),
                'university': np.random.choice(['TBU', 'CTU', 'UEP'], p=[0.4, 0.3, 0.3]),
                'department': np.random.choice(['Computer Science', 'Information Systems', 'Engineering', 'Business']),
                'elearning_experience': np.random.choice(['None', 'Basic', 'Intermediate', 'Advanced'], p=[0.1, 0.3, 0.4, 0.2]),
                'group': 'control' if i < 10 else 'treatment'
            })
            participant_id += 1
        
        # Generate 20 students  
        for i in range(20):
            participants.append({
                'participant_id': f'STU_{participant_id:03d}',
                'role': 'student',
                'age': np.random.randint(19, 26),
                'gender': np.random.choice(['M', 'F'], p=[0.55, 0.45]),
                'years_experience': np.random.randint(0, 4),
                'technical_background': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                'university': np.random.choice(['TBU', 'CTU', 'UEP'], p=[0.4, 0.3, 0.3]),
                'department': np.random.choice(['Computer Science', 'Information Systems', 'Engineering', 'Business']),
                'elearning_experience': np.random.choice(['None', 'Basic', 'Intermediate', 'Advanced'], p=[0.05, 0.35, 0.45, 0.15]),
                'group': 'control' if i < 10 else 'treatment'
            })
            participant_id += 1
        
        # Generate 20 administrators
        for i in range(20):
            participants.append({
                'participant_id': f'ADM_{participant_id:03d}',
                'role': 'administrator',
                'age': np.random.randint(25, 50),
                'gender': np.random.choice(['M', 'F'], p=[0.4, 0.6]),
                'years_experience': np.random.randint(2, 12),
                'technical_background': np.random.choice(['Low', 'Medium', 'High'], p=[0.15, 0.6, 0.25]),
                'university': np.random.choice(['TBU', 'CTU', 'UEP'], p=[0.4, 0.3, 0.3]),
                'department': np.random.choice(['IT Services', 'Educational Technology', 'Academic Affairs', 'Distance Learning']),
                'elearning_experience': np.random.choice(['None', 'Basic', 'Intermediate', 'Advanced'], p=[0.05, 0.2, 0.5, 0.25]),
                'group': 'control' if i < 10 else 'treatment'
            })
            participant_id += 1
        
        # Ensure exactly 30 in each group
        df = pd.DataFrame(participants)
        df.loc[:29, 'group'] = 'control'
        df.loc[30:59, 'group'] = 'treatment'
        
        return df
    
    def generate_session_information(self, demographics_df):
        """Generate session information for each participant"""
        logger.info("Generating session information...")
        
        sessions = []
        base_date = datetime(2024, 3, 1)
        
        for _, participant in demographics_df.iterrows():
            participant_id = participant['participant_id']
            group = participant['group']
            
            # Session duration based on group
            if group == 'control':
                duration_minutes = np.random.normal(60, 8)  # Traditional interview
            else:
                duration_minutes = np.random.normal(90, 12)  # Multimedia session
            
            sessions.append({
                'participant_id': participant_id,
                'session_date': (base_date + timedelta(days=np.random.randint(0, 60))).strftime('%Y-%m-%d'),
                'session_duration_minutes': max(30, duration_minutes),  # Minimum 30 minutes
                'completion_status': np.random.choice(['completed', 'completed', 'completed', 'partial'], p=[0.9, 0.05, 0.03, 0.02]),
                'satisfaction_score': np.random.normal(4.8 if group == 'control' else 6.1, 
                                                     1.2 if group == 'control' else 0.9),
                'interviewer_id': f"INT_{np.random.randint(1, 6):02d}",
                'location': np.random.choice(['Lab_A', 'Lab_B', 'Lab_C', 'Remote']),
                'technical_issues': np.random.choice([0, 1], p=[0.85, 0.15]),
                'group': group
            })
        
        return pd.DataFrame(sessions)
    
    def generate_ground_truth_requirements(self):
        """Generate expert-validated ground truth requirements"""
        logger.info("Generating ground truth requirements...")
        
        requirements = []
        req_id = 1
        
        # Functional requirements (73 total)
        functional_categories = [
            'User Management', 'Course Management', 'Assessment', 'Communication', 
            'Content Management', 'Reporting', 'Navigation', 'Search'
        ]
        
        for category in functional_categories:
            num_reqs = np.random.randint(7, 12)
            for i in range(num_reqs):
                requirements.append({
                    'requirement_id': f'FR_{req_id:03d}',
                    'type': 'functional',
                    'category': category,
                    'description': f'System shall provide {category.lower()} functionality for requirement {req_id}',
                    'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
                    'complexity': np.random.choice(['Simple', 'Medium', 'Complex'], p=[0.4, 0.4, 0.2]),
                    'identified_by_control': np.random.choice([0, 1], p=[0.29, 0.71]),  # 71% identification rate
                    'identified_by_treatment': np.random.choice([0, 1], p=[0.08, 0.92]),  # 92% identification rate
                    'expert_confidence': np.random.uniform(0.8, 1.0),
                    'effort_estimate': np.random.randint(1, 8)
                })
                req_id += 1
                if len([r for r in requirements if r['type'] == 'functional']) >= 73:
                    break
            if len([r for r in requirements if r['type'] == 'functional']) >= 73:
                break
        
        # Non-functional requirements (54 total)
        nf_categories = [
            'Performance', 'Usability', 'Security', 'Reliability', 
            'Accessibility', 'Compatibility', 'Maintainability'
        ]
        
        for category in nf_categories:
            num_reqs = np.random.randint(6, 10)
            for i in range(num_reqs):
                requirements.append({
                    'requirement_id': f'NFR_{req_id:03d}',
                    'type': 'non-functional',
                    'category': category,
                    'description': f'System shall meet {category.lower()} requirements for aspect {req_id}',
                    'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.4, 0.4, 0.2]),
                    'complexity': np.random.choice(['Simple', 'Medium', 'Complex'], p=[0.2, 0.5, 0.3]),
                    'identified_by_control': np.random.choice([0, 1], p=[0.52, 0.48]),  # 48% identification rate
                    'identified_by_treatment': np.random.choice([0, 1], p=[0.15, 0.85]),  # 85% identification rate
                    'expert_confidence': np.random.uniform(0.7, 0.95),
                    'effort_estimate': np.random.randint(2, 12)
                })
                req_id += 1
                if len([r for r in requirements if r['type'] == 'non-functional']) >= 54:
                    break
            if len([r for r in requirements if r['type'] == 'non-functional']) >= 54:
                break
        
        return pd.DataFrame(requirements)
    
    def generate_evaluation_results(self, demographics_df):
        """Generate evaluation metrics for both groups"""
        logger.info("Generating evaluation results...")
        
        results = []
        
        # Generate individual participant results
        for _, participant in demographics_df.iterrows():
            participant_id = participant['participant_id']
            group = participant['group']
            stats = self.expected_results[group]
            
            results.append({
                'participant_id': participant_id,
                'group': group,
                'requirements_identified': max(20, np.random.normal(stats['requirements_identified'][0], 
                                                                  stats['requirements_identified'][1])),
                'precision': np.clip(np.random.normal(stats['precision'][0], stats['precision'][1]), 0, 1),
                'recall': np.clip(np.random.normal(stats['recall'][0], stats['recall'][1]), 0, 1),
                'f1_score': np.clip(np.random.normal(stats['f1_score'][0], stats['f1_score'][1]), 0, 1),
                'satisfaction_score': np.clip(np.random.normal(stats['satisfaction_score'][0], 
                                                             stats['satisfaction_score'][1]), 1, 7),
                'functional_req_identified': np.random.randint(35, 68),
                'nonfunctional_req_identified': np.random.randint(15, 50),
                'session_duration': np.random.normal(60 if group == 'control' else 90, 10),
                'analysis_time_minutes': np.random.normal(180 if group == 'control' else 45, 20)
            })
        
        return pd.DataFrame(results)
    
    def generate_multimedia_metadata(self, demographics_df):
        """Generate metadata for multimedia files"""
        logger.info("Generating multimedia metadata...")
        
        multimedia_files = []
        
        # Treatment group participants only
        treatment_participants = demographics_df[demographics_df['group'] == 'treatment']
        
        for _, participant in treatment_participants.iterrows():
            participant_id = participant['participant_id']
            
            # Audio file
            multimedia_files.append({
                'participant_id': participant_id,
                'file_type': 'audio',
                'filename': f'audio_{participant_id}_interview.wav',
                'duration_seconds': np.random.normal(3600, 480),  # ~60 minutes
                'file_size_mb': np.random.normal(350, 50),
                'sample_rate': 44100,
                'channels': 1,
                'format': 'WAV',
                'quality_score': np.random.uniform(0.8, 1.0),
                'transcription_accuracy': np.random.uniform(0.85, 0.98),
                'sentiment_positive': np.random.uniform(0.3, 0.7),
                'sentiment_negative': np.random.uniform(0.1, 0.4),
                'sentiment_neutral': np.random.uniform(0.2, 0.5),
                'keywords_extracted': np.random.randint(25, 60),
                'fps': None,
                'interactions_detected': None,
                'hesitation_events': None,
                'ui_elements_detected': None,
                'attention_regions': None,
                'annotations_count': None,
                'ui_elements_count': None,
                'text_regions': None,
                'ocr_confidence': None
            })
            
            # Video file
            multimedia_files.append({
                'participant_id': participant_id,
                'file_type': 'video',
                'filename': f'video_{participant_id}_session.mp4',
                'duration_seconds': np.random.normal(1800, 300),  # ~30 minutes
                'file_size_mb': np.random.normal(1200, 200),
                'resolution': '1920x1080',
                'fps': 30,
                'format': 'MP4',
                'quality_score': np.random.uniform(0.7, 0.95),
                'interactions_detected': np.random.randint(50, 150),
                'hesitation_events': np.random.randint(5, 25),
                'ui_elements_detected': np.random.randint(15, 40),
                'attention_regions': np.random.randint(8, 20),
                'sample_rate': None,
                'channels': None,
                'transcription_accuracy': None,
                'sentiment_positive': None,
                'sentiment_negative': None,
                'sentiment_neutral': None,
                'keywords_extracted': None,
                'annotations_count': None,
                'ui_elements_count': None,
                'text_regions': None,
                'ocr_confidence': None
            })
            
            # Image files (3-6 per participant)
            num_images = np.random.randint(3, 7)
            for j in range(num_images):
                multimedia_files.append({
                    'participant_id': participant_id,
                    'file_type': 'image',
                    'filename': f'screenshot_{participant_id}_{j+1:02d}.png',
                    'duration_seconds': None,
                    'file_size_mb': np.random.uniform(0.5, 3.0),
                    'resolution': '1920x1080',
                    'format': 'PNG',
                    'quality_score': np.random.uniform(0.8, 1.0),
                    'annotations_count': np.random.randint(1, 8),
                    'ui_elements_count': np.random.randint(5, 20),
                    'text_regions': np.random.randint(2, 12),
                    'ocr_confidence': np.random.uniform(0.8, 0.98),
                    'sample_rate': None,
                    'channels': None,
                    'transcription_accuracy': None,
                    'sentiment_positive': None,
                    'sentiment_negative': None,
                    'sentiment_neutral': None,
                    'keywords_extracted': None,
                    'fps': None,
                    'interactions_detected': None,
                    'hesitation_events': None,
                    'ui_elements_detected': None,
                    'attention_regions': None
                })
        
        return pd.DataFrame(multimedia_files)
    
    def generate_summary_statistics(self, demographics_df, evaluation_df, ground_truth_df, multimedia_df):
        """Generate dataset summary statistics"""
        logger.info("Generating summary statistics...")
        
        summary_stats = {
            'dataset_info': {
                'generation_date': datetime.now().isoformat(),
                'random_seed': self.seed,
                'total_participants': len(demographics_df),
                'control_group_size': len(demographics_df[demographics_df['group'] == 'control']),
                'treatment_group_size': len(demographics_df[demographics_df['group'] == 'treatment'])
            },
            'requirements': {
                'total_requirements': len(ground_truth_df),
                'functional_requirements': len(ground_truth_df[ground_truth_df['type'] == 'functional']),
                'non_functional_requirements': len(ground_truth_df[ground_truth_df['type'] == 'non-functional'])
            },
            'multimedia': {
                'total_files': len(multimedia_df),
                'audio_files': len(multimedia_df[multimedia_df['file_type'] == 'audio']),
                'video_files': len(multimedia_df[multimedia_df['file_type'] == 'video']),
                'image_files': len(multimedia_df[multimedia_df['file_type'] == 'image'])
            },
            'session_stats': {
                'average_duration_control': evaluation_df[evaluation_df['group'] == 'control']['session_duration'].mean(),
                'average_duration_treatment': evaluation_df[evaluation_df['group'] == 'treatment']['session_duration'].mean()
            },
            'key_results': {
                'requirements_improvement': ((evaluation_df[evaluation_df['group'] == 'treatment']['requirements_identified'].mean() - 
                                            evaluation_df[evaluation_df['group'] == 'control']['requirements_identified'].mean()) / 
                                           evaluation_df[evaluation_df['group'] == 'control']['requirements_identified'].mean()) * 100,
                'satisfaction_improvement': ((evaluation_df[evaluation_df['group'] == 'treatment']['satisfaction_score'].mean() - 
                                            evaluation_df[evaluation_df['group'] == 'control']['satisfaction_score'].mean()) / 
                                           evaluation_df[evaluation_df['group'] == 'control']['satisfaction_score'].mean()) * 100
            }
        }
        
        return summary_stats
    
    def save_datasets(self, datasets, summary_stats):
        """Save all generated datasets to files"""
        logger.info("Saving datasets to files...")
        
        # Save CSV files
        for name, df in datasets.items():
            filename = f"{name}.csv"
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"✓ Saved {filename} ({len(df)} rows)")
        
        # Save summary statistics
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"✓ Saved dataset_summary.json")
        
    def validate_generated_data(self, datasets):
        """Validate the generated data meets expected criteria"""
        logger.info("Validating generated data...")
        
        evaluation_df = datasets['evaluation_results']
        ground_truth_df = datasets['ground_truth_requirements']
        
        # Check group sizes
        control_size = len(evaluation_df[evaluation_df['group'] == 'control'])
        treatment_size = len(evaluation_df[evaluation_df['group'] == 'treatment'])
        
        assert control_size == 30, f"Expected 30 control participants, got {control_size}"
        assert treatment_size == 30, f"Expected 30 treatment participants, got {treatment_size}"
        
        # Check requirements counts
        functional_count = len(ground_truth_df[ground_truth_df['type'] == 'functional'])
        nonfunctional_count = len(ground_truth_df[ground_truth_df['type'] == 'non-functional'])
        
        assert functional_count == 73, f"Expected 73 functional requirements, got {functional_count}"
        assert nonfunctional_count == 54, f"Expected 54 non-functional requirements, got {nonfunctional_count}"
        
        # Check statistical properties (within tolerance)
        control = evaluation_df[evaluation_df['group'] == 'control']
        treatment = evaluation_df[evaluation_df['group'] == 'treatment']
        
        tolerance = 0.2  # 20% tolerance for synthetic data
        
        for metric in ['requirements_identified', 'precision', 'recall', 'f1_score', 'satisfaction_score']:
            control_mean = control[metric].mean()
            treatment_mean = treatment[metric].mean()
            
            expected_control = self.expected_results['control'][metric][0]
            expected_treatment = self.expected_results['treatment'][metric][0]
            
            control_diff = abs(control_mean - expected_control) / expected_control
            treatment_diff = abs(treatment_mean - expected_treatment) / expected_treatment
            
            if control_diff > tolerance:
                logger.warning(f"Control {metric} mean {control_mean:.3f} differs from expected {expected_control:.3f} by {control_diff:.1%}")
            
            if treatment_diff > tolerance:
                logger.warning(f"Treatment {metric} mean {treatment_mean:.3f} differs from expected {expected_treatment:.3f} by {treatment_diff:.1%}")
        
        logger.info("✓ Data validation completed")
    
    def generate_complete_dataset(self):
        """Generate the complete dataset"""
        logger.info("Starting dataset generation...")
        logger.info(f"Random seed: {self.seed}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Generate all components
        demographics_df = self.generate_participants_demographics()
        session_df = self.generate_session_information(demographics_df)
        ground_truth_df = self.generate_ground_truth_requirements()
        evaluation_df = self.generate_evaluation_results(demographics_df)
        multimedia_df = self.generate_multimedia_metadata(demographics_df)
        
        # Combine into datasets dictionary
        datasets = {
            'participants_demographics': demographics_df,
            'session_information': session_df,
            'ground_truth_requirements': ground_truth_df,
            'evaluation_results': evaluation_df,
            'multimedia_metadata': multimedia_df
        }
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(
            demographics_df, evaluation_df, ground_truth_df, multimedia_df
        )
        
        # Validate data
        self.validate_generated_data(datasets)
        
        # Save all datasets
        self.save_datasets(datasets, summary_stats)
        
        # Print generation summary
        self.print_generation_summary(datasets, summary_stats)
        
        return datasets, summary_stats
    
    def print_generation_summary(self, datasets, summary_stats):
        """Print summary of generated dataset"""
        print("\n" + "="*60)
        print("DATASET GENERATION SUMMARY")
        print("="*60)
        
        print(f"Total participants: {summary_stats['dataset_info']['total_participants']}")
        print(f"Control group: {summary_stats['dataset_info']['control_group_size']}")
        print(f"Treatment group: {summary_stats['dataset_info']['treatment_group_size']}")
        print(f"Total requirements: {summary_stats['requirements']['total_requirements']}")
        print(f"Functional requirements: {summary_stats['requirements']['functional_requirements']}")
        print(f"Non-functional requirements: {summary_stats['requirements']['non_functional_requirements']}")
        print(f"Multimedia files: {summary_stats['multimedia']['total_files']}")
        
        print(f"\nKey improvements (Treatment vs Control):")
        print(f"Requirements identified: +{summary_stats['key_results']['requirements_improvement']:.1f}%")
        print(f"Stakeholder satisfaction: +{summary_stats['key_results']['satisfaction_improvement']:.1f}%")
        
        print(f"\nFiles generated in: {self.output_dir}")
        for name in datasets.keys():
            print(f"  ✓ {name}.csv")
        print(f"  ✓ dataset_summary.json")
        
        print(f"\nDataset generation completed successfully!")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate multimedia requirements engineering dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py
  python generate_dataset.py --output-dir ./data --seed 123
  python generate_dataset.py --help
        """
    )
    
    parser.add_argument(
        '--output-dir', 
        default='.',
        help='Output directory for generated files (default: current directory)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing dataset without regenerating'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = MultimediaREDatasetGenerator(
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if args.validate_only:
        # Load and validate existing data
        try:
            datasets = {}
            file_names = [
                'participants_demographics',
                'session_information', 
                'ground_truth_requirements',
                'evaluation_results',
                'multimedia_metadata'
            ]
            
            for name in file_names:
                filepath = generator.output_dir / f"{name}.csv"
                datasets[name] = pd.read_csv(filepath)
            
            generator.validate_generated_data(datasets)
            print("✓ Existing dataset validation passed")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return 1
    else:
        # Generate new dataset
        try:
            datasets, summary_stats = generator.generate_complete_dataset()
            return 0
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            return 1

if __name__ == "__main__":
    exit(main())
