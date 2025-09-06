import numpy as np

# language=Python, caption="Cohen's Kappa Implementation", label="lst:kappa"

def create_binary_matrix(selected_reqs, all_possible_reqs):
    """Convert list of selected requirements to binary vector"""
    return np.array([1 if req in selected_reqs else 0 for req in all_possible_reqs])

def calculate_cohens_kappa(expert1_reqs, expert2_reqs, all_possible_reqs):
    """
    Calculate inter-rater reliability (Cohen's Kappa)
    """
    # Create binary matrices (requirement present/absent)
    expert1_binary = create_binary_matrix(expert1_reqs, all_possible_reqs)
    expert2_binary = create_binary_matrix(expert2_reqs, all_possible_reqs)
    
    # Calculate observed agreement
    observed_agreement = np.mean(expert1_binary == expert2_binary)
    
    # Calculate expected agreement by chance
    p1_positive = np.mean(expert1_binary)
    p2_positive = np.mean(expert2_binary)
    expected_agreement = (p1_positive * p2_positive + 
                         (1 - p1_positive) * (1 - p2_positive))
    
    # Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa

# Example usage
all_reqs = ["Req1", "Req2", "Req3", "Req4"]
expert1 = ["Req1", "Req3"]
expert2 = ["Req1", "Req4"]

print("Cohen's Kappa:", calculate_cohens_kappa(expert1, expert2, all_reqs))
