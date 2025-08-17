#language=Python, caption=Cohen's Kappa Implementation, label=lst:kappa]
def calculate_cohens_kappa(expert1_reqs, expert2_reqs, all_possible_reqs):
    """
    Calculate inter-rater reliability for requirement identification
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
                         (1-p1_positive) * (1-p2_positive))
    
    # Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa

# Results:
# Initial κ = 0.72 (substantial agreement)
# Post-discussion κ = 0.89 (near perfect agreement)
