# Contributing to Requirements Engineering Dataset

We welcome contributions to improve this dataset and analysis framework! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports** - Report issues with dataset generation or analysis
2. **Feature Requests** - Suggest new analysis capabilities or improvements
3. **Code Contributions** - Implement new features or fix bugs
4. **Documentation** - Improve documentation, tutorials, or examples
5. **Data Validation** - Help verify and improve data quality
6. **Research Extensions** - Add new multimedia processing techniques

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/dataset.git
   cd dataset
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[dev]  # Install in development mode
   ```

3. **Run Tests to Ensure Everything Works**
   ```bash
   python dataset_generator.py
   python validate_dataset.py
   pytest tests/
   ```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear title** describing the issue
- **Environment details**: OS, Python version, package versions
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Error messages** or log output
- **Sample code** if applicable

**Template:**
```markdown
## Bug Description
Brief description of what went wrong.

## Environment
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python Version: [e.g., 3.9.7]
- Package Version: [e.g., 1.0.0]

## Steps to Reproduce
1. Run command X
2. Execute function Y
3. See error Z

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Output
```
Paste error messages or logs here
```
```

### Feature Requests

For feature requests, please provide:

- **Clear description** of the proposed feature
- **Use case** explaining why this would be valuable
- **Proposed implementation** if you have ideas
- **Examples** of how the feature would work

## 💻 Code Contributions

### Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Check code formatting
   black --check .
   flake8 .
   
   # Validate dataset generation
   python dataset_generator.py
   python validate_dataset.py
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Coding Standards

#### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 127 characters
- Use type hints where appropriate

```python
def analyze_requirements(
    requirements: List[Dict[str, Any]], 
    threshold: float = 0.8
) -> Tuple[int, float]:
    """
    Analyze requirements data.
    
    Args:
        requirements: List of requirement dictionaries
        threshold: Confidence threshold for filtering
        
    Returns:
        Tuple of (count, average_confidence)
    """
    pass
```

#### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings when helpful
- Keep README and guides up to date
- Use clear, concise language

#### Testing Guidelines

- Write unit tests for all new functions
- Aim for >90% test coverage
- Use descriptive test names
- Test edge cases and error conditions

```python
def test_requirements_generation_with_custom_parameters():
    """Test that custom statistical parameters are correctly applied."""
    generator = RequirementsDatasetGenerator()
    generator.control_stats['mean_requirements'] = 100.0
    
    # Generate and validate
    participants = generator.generate_participants()
    assert len(participants) == generator.participants_count
```

### Pull Request Guidelines

#### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

#### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix/feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes or breaking changes documented
```

## 📊 Dataset Improvements

### Adding New Multimedia Techniques

To add new multimedia processing techniques:

1. **Extend the Framework**
   ```python
   class RequirementsDatasetGenerator:
       def generate_new_multimedia_data(self, participants):
           """Add your new multimedia processing method."""
           pass
   ```

2. **Add Validation**
   ```python
   class DatasetValidator:
       def validate_new_multimedia_data(self):
           """Validate your new data format."""
           pass
   ```

3. **Update Documentation**
   - Add method description to user guide
   - Include examples in README
   - Update API documentation

### Data Quality Improvements

Help improve data quality by:

- **Identifying inconsistencies** in generated data
- **Suggesting better statistical models** for data generation
- **Adding validation checks** for edge cases
- **Improving realism** of synthetic data patterns

### Statistical Model Enhancements

To improve statistical accuracy:

- **Validate against additional empirical studies**
- **Add more nuanced statistical relationships**
- **Implement advanced statistical techniques**
- **Add support for different research contexts**

## 📝 Documentation Contributions

### Types of Documentation

1. **Code Documentation**
   - Docstrings for functions and classes
   - Inline comments for complex logic
   - Type hints for better code understanding

2. **User Documentation**
   - Tutorials and how-to guides
   - API reference documentation
   - Example notebooks and scripts

3. **Research Documentation**
   - Methodology explanations
   - Statistical validation details
   - Comparison with other approaches

### Documentation Standards

- **Clear and concise** language
- **Step-by-step instructions** with examples
- **Screenshots or diagrams** where helpful
- **Up-to-date** with current code version
- **Accessible** to users with different experience levels

## 🧪 Research Extensions

### Adding New Research Domains

To extend the framework to new domains:

1. **Define domain-specific requirements**
2. **Adapt statistical parameters**
3. **Add domain-specific validation**
4. **Create domain examples**
5. **Document domain differences**

### Implementing New Analysis Techniques

For new analysis methods:

1. **Research the technique thoroughly**
2. **Implement with proper validation**
3. **Add comprehensive tests**
4. **Document the methodology**
5. **Provide usage examples**

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] GitHub release created
- [ ] PyPI package updated

## 📧 Communication

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: okechukwu@utb.cz for research collaboration

### Community Guidelines

- **Be respectful** and inclusive
- **Stay on topic** in discussions
- **Provide constructive feedback**
- **Help others** when you can
- **Follow the code of conduct**

## 🙏 Recognition

### Contributors

We recognize all contributors in:

- **README.md** contributors section
- **GitHub releases** acknowledgments
- **Academic papers** (for significant contributions)
- **Conference presentations** when appropriate

### Types of Recognition

- **Code contributors**: GitHub commits and PRs
- **Bug reporters**: Issue reports and testing
- **Documentation writers**: Docs improvements
- **Researchers**: Methodological contributions
- **Community helpers**: Support and guidance

## 📋 Development Setup Details

### Advanced Setup

For advanced development features:

```bash
# Install with all development dependencies
pip install -e .[dev,docs]

# Set up pre-commit hooks
pre-commit install

# Configure git hooks for code quality
git config --local core.hooksPath .githooks/
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true
}
```

#### PyCharm Configuration

- Set interpreter to virtual environment
- Configure code style to Black
- Enable pytest as test runner
- Set up flake8 as external tool

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger for complex issues
import pdb; pdb.set_trace()

# Profile performance bottlenecks
import cProfile
cProfile.run('your_function()')
```

## 🔄 Continuous Integration

### GitHub Actions

Our CI pipeline runs:

1. **Code Quality Checks** (linting, formatting)
2. **Unit Tests** (across Python versions and OS)
3. **Integration Tests** (full dataset generation)
4. **Statistical Validation** (verify research findings)
5. **Documentation Build** (ensure docs compile)
6. **Package Testing** (installation and imports)

### Local CI Simulation

```bash
# Run the same checks locally
make ci-check  # or
./scripts/run_ci_checks.sh
```

---

Thank you for contributing to the Requirements Engineering Dataset! Your efforts help advance research in multimedia-enhanced software requirements engineering.

For questions about contributing, please contact: okechukwu@utb.cz