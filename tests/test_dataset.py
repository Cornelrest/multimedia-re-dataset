#!/usr/bin/env python3
"""
Test Suite for Requirements Engineering Dataset Generator
========================================================

Comprehensive tests for the dataset generation and validation system.

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University in Zlin
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import modules to test
from dataset_generator import RequirementsDatasetGenerator
from validate_dataset import DatasetValidator


class TestRequirementsDatasetGenerator:
    """Test cases for the dataset generator."""

    @pytest.fixture
    def generator(self):
        """Create a dataset generator instance for testing."""
        return RequirementsDatasetGenerator()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_generator_initialization(self, generator):
        """Test that generator initializes with correct parameters."""
        assert generator.participants_count == 60
        assert generator.control_group_size == 30
        assert generator.treatment_group_size == 30
        assert generator.total_requirements == 127
        assert generator.functional_requirements_total == 73
        assert generator.non_functional_requirements_total == 54

    def test_generate_participants(self, generator):
        """Test participant generation."""
        participants = generator.generate_participants()

        # Check structure
        assert isinstance(participants, pd.DataFrame)
        assert len(participants) == generator.participants_count

        # Check required columns
        required_columns = [
            "participant_id",
            "institution",
            "stakeholder_type",
            "group_assignment",
            "age",
            "gender",
            "experience_years",
            "consent_given",
            "session_date",
            "anonymized_id",
        ]

        for col in required_columns:
            assert col in participants.columns

        # Check group distribution
        group_counts = participants["group_assignment"].value_counts()
        assert group_counts["Control"] == generator.control_group_size
        assert group_counts["Treatment"] == generator.treatment_group_size

        # Check institutions
        assert participants["institution"].nunique() == 3

        # Check stakeholder types
        assert participants["stakeholder_type"].nunique() == 3

        # Check data validity
        assert participants["age"].min() >= 18
        assert participants["age"].max() <= 70
        assert participants["consent_given"].all()

    def test_generate_ground_truth_requirements(self, generator):
        """Test ground truth requirements generation."""
        requirements = generator.generate_ground_truth_requirements()

        # Check structure
        assert isinstance(requirements, pd.DataFrame)
        assert len(requirements) == generator.total_requirements

        # Check required columns
        required_columns = [
            "requirement_id",
            "type",
            "category",
            "priority",
            "complexity",
            "description",
            "expert_consensus",
        ]

        for col in required_columns:
            assert col in requirements.columns

        # Check type distribution
        type_counts = requirements["type"].value_counts()
        assert type_counts["Functional"] == generator.functional_requirements_total
        assert (
            type_counts["Non-Functional"] == generator.non_functional_requirements_total
        )

        # Check data validity
        assert requirements["expert_consensus"].min() >= 0
        assert requirements["expert_consensus"].max() <= 1

        # Check unique IDs
        assert requirements["requirement_id"].nunique() == len(requirements)

    def test_generate_participant_results(self, generator):
        """Test participant results generation."""
        participants = generator.generate_participants()
        results = generator.generate_participant_results(participants)

        # Check structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(participants)

        # Check required columns
        required_columns = [
            "participant_id",
            "group_assignment",
            "requirements_identified",
            "functional_requirements",
            "non_functional_requirements",
            "completeness_score",
            "precision",
            "recall",
            "f1_score",
            "satisfaction_score",
            "total_time_min",
        ]

        for col in required_columns:
            assert col in results.columns

        # Check data ranges
        assert results["completeness_score"].min() >= 0
        assert results["completeness_score"].max() <= 1
        assert results["precision"].min() >= 0
        assert results["precision"].max() <= 1
        assert results["recall"].min() >= 0
        assert results["recall"].max() <= 1
        assert results["f1_score"].min() >= 0
        assert results["f1_score"].max() <= 1
        assert results["satisfaction_score"].min() >= 1
        assert results["satisfaction_score"].max() <= 7

        # Check group differences (statistical properties)
        control_group = results[results["group_assignment"] == "Control"]
        treatment_group = results[results["group_assignment"] == "Treatment"]

        # Treatment should have higher means
        assert (
            treatment_group["requirements_identified"].mean()
            > control_group["requirements_identified"].mean()
        )
        assert treatment_group["precision"].mean() > control_group["precision"].mean()
        assert (
            treatment_group["satisfaction_score"].mean()
            > control_group["satisfaction_score"].mean()
        )

    def test_generate_multimedia_analysis_data(self, generator):
        """Test multimedia analysis data generation."""
        participants = generator.generate_participants()
        multimedia_data = generator.generate_multimedia_analysis_data(participants)

        # Check structure
        assert isinstance(multimedia_data, dict)
        assert "audio_analysis" in multimedia_data
        assert "video_analysis" in multimedia_data
        assert "image_analysis" in multimedia_data

        # Check each modality
        for modality_name, modality_data in multimedia_data.items():
            assert isinstance(modality_data, pd.DataFrame)
            assert len(modality_data) > 0
            assert "participant_id" in modality_data.columns
            assert "confidence_score" in modality_data.columns

            # Check confidence scores are valid
            assert modality_data["confidence_score"].min() >= 0
            assert modality_data["confidence_score"].max() <= 1

    def test_generate_cost_analysis(self, generator):
        """Test cost analysis generation."""
        cost_data = generator.generate_cost_analysis()

        assert isinstance(cost_data, pd.DataFrame)
        assert len(cost_data) > 0

        # Check required columns
        required_columns = ["component", "category", "amount_usd", "frequency"]
        for col in required_columns:
            assert col in cost_data.columns

        # Check categories
        categories = cost_data["category"].unique()
        assert "Setup Cost" in categories
        assert "Per-Project Savings" in categories

        # Check amounts are positive
        assert cost_data["amount_usd"].min() > 0

    def test_generate_complete_dataset(self, generator, temp_dir):
        """Test complete dataset generation."""
        generator.generate_complete_dataset(temp_dir)

        # Check all expected files exist
        expected_files = [
            "participants.csv",
            "ground_truth_requirements.csv",
            "participant_results.csv",
            "audio_analysis.csv",
            "video_analysis.csv",
            "image_analysis.csv",
            "cost_analysis.csv",
            "summary_statistics.json",
            "dataset_metadata.json",
        ]

        for file in expected_files:
            file_path = os.path.join(temp_dir, file)
            assert os.path.exists(file_path), f"File {file} not found"
            assert os.path.getsize(file_path) > 0, f"File {file} is empty"

    def test_reproducibility(self, temp_dir):
        """Test that dataset generation is reproducible with same seed."""
        # Generate dataset twice with same seed
        np.random.seed(42)
        generator1 = RequirementsDatasetGenerator()
        generator1.generate_complete_dataset(os.path.join(temp_dir, "dataset1"))

        np.random.seed(42)
        generator2 = RequirementsDatasetGenerator()
        generator2.generate_complete_dataset(os.path.join(temp_dir, "dataset2"))

        # Compare key files
        df1 = pd.read_csv(os.path.join(temp_dir, "dataset1", "participants.csv"))
        df2 = pd.read_csv(os.path.join(temp_dir, "dataset2", "participants.csv"))

        pd.testing.assert_frame_equal(df1, df2, "Datasets are not reproducible")

    def test_custom_parameters(self):
        """Test that custom statistical parameters are applied."""
        generator = RequirementsDatasetGenerator()

        # Modify parameters
        generator.control_stats["mean_requirements"] = 100.0
        generator.treatment_stats["mean_requirements"] = 150.0

        participants = generator.generate_participants()
        results = generator.generate_participant_results(participants)

        control_group = results[results["group_assignment"] == "Control"]
        treatment_group = results[results["group_assignment"] == "Treatment"]

        # Check that means are approximately as expected (within tolerance)
        control_mean = control_group["requirements_identified"].mean()
        treatment_mean = treatment_group["requirements_identified"].mean()

        assert (
            90 <= control_mean <= 110
        ), f"Control mean {control_mean} not near expected 100"
        assert (
            140 <= treatment_mean <= 160
        ), f"Treatment mean {treatment_mean} not near expected 150"


class TestDatasetValidator:
    """Test cases for the dataset validator."""

    @pytest.fixture
    def validator(self, temp_dir):
        """Create a validator instance with test data."""
        # Generate test dataset first
        generator = RequirementsDatasetGenerator()
        generator.generate_complete_dataset(temp_dir)
        return DatasetValidator(temp_dir)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.data_dir is not None
        assert isinstance(validator.expected_values, dict)
        assert validator.expected_values["total_participants"] == 60

    def test_validate_file_structure(self, validator):
        """Test file structure validation."""
        result = validator.validate_file_structure()
        assert result is True
        assert validator.validation_results["file_structure"]["passed"] is True
        assert len(validator.validation_results["file_structure"]["missing_files"]) == 0

    def test_validate_participants_data(self, validator):
        """Test participants data validation."""
        result = validator.validate_participants_data()
        assert result is True
        assert validator.validation_results["participants"]["passed"] is True
        assert validator.validation_results["participants"]["row_count"] == 60

    def test_validate_ground_truth_requirements(self, validator):
        """Test ground truth requirements validation."""
        result = validator.validate_ground_truth_requirements()
        assert result is True
        assert validator.validation_results["ground_truth"]["passed"] is True
        assert validator.validation_results["ground_truth"]["total_requirements"] == 127
        assert validator.validation_results["ground_truth"]["functional_count"] == 73
        assert (
            validator.validation_results["ground_truth"]["non_functional_count"] == 54
        )

    def test_validate_participant_results(self, validator):
        """Test participant results validation."""
        result = validator.validate_participant_results()
        assert result is True
        assert validator.validation_results["participant_results"]["passed"] is True

    def test_validate_multimedia_data(self, validator):
        """Test multimedia data validation."""
        result = validator.validate_multimedia_data()
        assert result is True
        assert validator.validation_results["multimedia_data"]["passed"] is True

    def test_validate_statistical_integrity(self, validator):
        """Test statistical integrity validation."""
        result = validator.validate_statistical_integrity()
        assert result is True
        assert validator.validation_results["statistical_integrity"]["passed"] is True

    def test_validate_metadata(self, validator):
        """Test metadata validation."""
        result = validator.validate_metadata()
        assert result is True
        assert validator.validation_results["metadata"]["passed"] is True

    def test_run_full_validation(self, validator):
        """Test complete validation suite."""
        report = validator.run_full_validation()

        assert isinstance(report, dict)
        assert "overall_passed" in report
        assert report["overall_passed"] is True
        assert report["steps_passed"] == report["total_steps"]
        assert len(report["errors"]) == 0

    def test_validation_with_missing_files(self, temp_dir):
        """Test validation behavior with missing files."""
        # Create validator with empty directory
        validator = DatasetValidator(temp_dir)

        result = validator.validate_file_structure()
        assert result is False
        assert len(validator.validation_results["file_structure"]["missing_files"]) > 0

    def test_validation_with_corrupted_data(self, temp_dir):
        """Test validation behavior with corrupted data."""
        # Generate dataset first
        generator = RequirementsDatasetGenerator()
        generator.generate_complete_dataset(temp_dir)

        # Corrupt participants file
        corrupted_data = pd.DataFrame({"invalid": [1, 2, 3]})
        corrupted_data.to_csv(os.path.join(temp_dir, "participants.csv"), index=False)

        validator = DatasetValidator(temp_dir)
        result = validator.validate_participants_data()
        assert result is False
        assert len(validator.errors) > 0


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_pipeline(self, temp_dir):
        """Test complete generation and validation pipeline."""
        # Generate dataset
        generator = RequirementsDatasetGenerator()
        generator.generate_complete_dataset(temp_dir)

        # Validate dataset
        validator = DatasetValidator(temp_dir)
        report = validator.run_full_validation()

        # Check validation passes
        assert report["overall_passed"] is True
        assert len(report["errors"]) == 0

        # Check key statistical properties
        results = pd.read_csv(os.path.join(temp_dir, "participant_results.csv"))
        control_group = results[results["group_assignment"] == "Control"]
        treatment_group = results[results["group_assignment"] == "Treatment"]

        # Verify expected improvements
        improvement = (
            (
                treatment_group["requirements_identified"].mean()
                - control_group["requirements_identified"].mean()
            )
            / control_group["requirements_identified"].mean()
        ) * 100

        assert (
            20 <= improvement <= 30
        ), f"Improvement {improvement:.1f}% not in expected range"

    def test_statistical_properties(self, temp_dir):
        """Test that generated data has correct statistical properties."""
        generator = RequirementsDatasetGenerator()
        generator.generate_complete_dataset(temp_dir)

        results = pd.read_csv(os.path.join(temp_dir, "participant_results.csv"))

        # Check group sizes
        group_counts = results["group_assignment"].value_counts()
        assert group_counts["Control"] == 30
        assert group_counts["Treatment"] == 30

        # Check statistical significance would be achieved
        from scipy import stats

        control_group = results[results["group_assignment"] == "Control"]
        treatment_group = results[results["group_assignment"] == "Treatment"]

        t_stat, p_value = stats.ttest_ind(
            treatment_group["requirements_identified"],
            control_group["requirements_identified"],
        )

        assert p_value < 0.001, f"P-value {p_value} not significant"

    def test_data_consistency(self, temp_dir):
        """Test consistency across different data files."""
        generator = RequirementsDatasetGenerator()
        generator.generate_complete_dataset(temp_dir)

        # Load data files
        participants = pd.read_csv(os.path.join(temp_dir, "participants.csv"))
        results = pd.read_csv(os.path.join(temp_dir, "participant_results.csv"))
        audio_data = pd.read_csv(os.path.join(temp_dir, "audio_analysis.csv"))

        # Check participant ID consistency
        assert set(participants["participant_id"]) == set(results["participant_id"])

        # Check treatment group multimedia data
        treatment_participants = participants[
            participants["group_assignment"] == "Treatment"
        ]["participant_id"]
        audio_participants = set(
            audio_data["participant_id"].str[:4]
        )  # First 4 chars of participant ID

        # Audio data should only contain treatment group participants
        for p_id in audio_participants:
            assert any(t_id.startswith(p_id) for t_id in treatment_participants)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_directory_creation(self):
        """Test handling of invalid output directory."""
        generator = RequirementsDatasetGenerator()

        # Try to create dataset in read-only location (this should handle gracefully)
        with pytest.raises(Exception):
            generator.generate_complete_dataset("/invalid/readonly/path")

    def test_missing_data_directory(self):
        """Test validator behavior with missing data directory."""
        validator = DatasetValidator("nonexistent_directory")

        result = validator.validate_file_structure()
        assert result is False
        assert len(validator.errors) > 0

    def test_empty_csv_files(self, temp_dir):
        """Test validation of empty CSV files."""
        # Create empty CSV files
        empty_files = ["participants.csv", "ground_truth_requirements.csv"]

        for file in empty_files:
            with open(os.path.join(temp_dir, file), "w") as f:
                f.write("")  # Empty file

        validator = DatasetValidator(temp_dir)
        validator.validate_file_structure()  # This should detect empty files

        assert len(validator.errors) > 0

    def test_malformed_json_metadata(self, temp_dir):
        """Test validation of malformed JSON metadata."""
        # Create malformed JSON file
        with open(os.path.join(temp_dir, "summary_statistics.json"), "w") as f:
            f.write("{ invalid json ")

        validator = DatasetValidator(temp_dir)
        result = validator.validate_metadata()

        assert result is False
        assert len(validator.errors) > 0


# Pytest configuration and test runner
def test_requirements_available():
    """Test that required packages are available."""
    try:
        import pandas
        import numpy
        import scipy
        import matplotlib
        import seaborn

        assert True
    except ImportError as e:
        pytest.fail(f"Required package not available: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])
