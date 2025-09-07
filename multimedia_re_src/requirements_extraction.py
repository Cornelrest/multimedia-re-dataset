"""
Requirements Extraction Module for Multimedia Requirements Engineering

This module handles cross-modal validation, requirements classification,
and generation of final requirements from integrated multimedia analysis.

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University, Czech Republic
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter
import re
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some analysis features may be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequirementsExtractor:
    """Main class for extracting and validating requirements from multimedia data."""
    
    def __init__(self):
        """Initialize the RequirementsExtractor."""
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        self.modality_weights = {
            'audio': 0.35,
            'video': 0.35,
            'image': 0.30
        }
        
        # Requirements classification patterns
        self.functional_patterns = [
            r'system (shall|should|must|will) (provide|allow|enable|support)',
            r'user (can|shall be able to|should be able to)',
            r'(function|feature|capability|operation)',
            r'(create|delete|update|modify|generate|process)',
            r'(login|logout|register|authenticate|authorize)',
            r'(search|filter|sort|display|show|hide)'
        ]
        
        self.non_functional_patterns = [
            r'(performance|speed|response time|latency)',
            r'(usability|user.?friendly|intuitive|easy)',
            r'(security|privacy|encryption|authentication)',
            r'(reliability|availability|uptime|stability)',
            r'(scalability|capacity|throughput|load)',
            r'(compatibility|interoperability|standards)',
            r'(maintainability|extensibility|modularity)',
            r'(accessibility|responsive|mobile.?friendly)'
        ]

def extract_requirements_from_multimedia(multimedia_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate requirements from integrated multimedia analysis.
    
    Args:
        multimedia_data: Dictionary containing results from audio, video, and image analysis
        
    Returns:
        Dictionary containing extracted requirements with validation and metadata
    """
    extractor = RequirementsExtractor()
    
    try:
        logger.info("Starting multimedia requirements extraction...")
        
        # Extract individual modality requirements
        audio_requirements = extract_audio_requirements(multimedia_data.get('audio', {}))
        video_requirements = extract_video_requirements(multimedia_data.get('video', {}))
        image_requirements = extract_image_requirements(multimedia_data.get('image', {}))
        
        # Combine all requirements
        all_requirements = {
            'audio': audio_requirements,
            'video': video_requirements,
            'image': image_requirements
        }
        
        # Perform cross-modal validation
        validated_requirements = validate_cross_modal_evidence(all_requirements)
        
        # Classify requirements
        classified_requirements = classify_requirements(validated_requirements)
        
        # Generate final requirement set
        final_requirements = generate_final_requirement_set(classified_requirements)
        
        # Create traceability matrix
        traceability = create_traceability_matrix(final_requirements, multimedia_data)
        
        # Generate quality metrics
        quality_metrics = calculate_extraction_quality_metrics(final_requirements)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'total_requirements': len(final_requirements),
            'requirements': final_requirements,
            'traceability_matrix': traceability,
            'quality_metrics': quality_metrics,
            'source_data_summary': create_source_data_summary(multimedia_data),
            'extraction_metadata': {
                'modalities_processed': list(multimedia_data.keys()),
                'confidence_thresholds': extractor.confidence_thresholds,
                'modality_weights': extractor.modality_weights
            }
        }
        
        logger.info(f"Requirements extraction completed. {len(final_requirements)} requirements extracted.")
        return result
        
    except Exception as e:
        logger.error(f"Requirements extraction failed: {e}")
        return {}

def validate_cross_modal_evidence(all_requirements: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Validate requirements using evidence from multiple modalities.
    
    Args:
        all_requirements: Dictionary of requirements from each modality
        
    Returns:
        List of validated requirements with confidence scores
    """
    logger.info("Performing cross-modal validation...")
    
    validated_requirements = []
    
    # Flatten all requirements with source modality
    flat_requirements = []
    for modality, reqs in all_requirements.items():
        for req in reqs:
            req['source_modality'] = modality
            flat_requirements.append(req)
    
    # Group similar requirements
    requirement_groups = group_similar_requirements(flat_requirements)
    
    # Validate each group
    for group_id, group in enumerate(requirement_groups):
        validated_req = validate_requirement_group(group, group_id)
        if validated_req:
            validated_requirements.append(validated_req)
    
    # Sort by confidence
    validated_requirements.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    logger.info(f"Cross-modal validation completed. {len(validated_requirements)} validated requirements.")
    return validated_requirements

def classify_requirements(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Classify requirements into functional and non-functional categories.
    
    Args:
        requirements: List of validated requirements
        
    Returns:
        List of requirements with classification labels
    """
    logger.info("Classifying requirements...")
    
    extractor = RequirementsExtractor()
    
    for req in requirements:
        description = req.get('description', '').lower()
        
        # Calculate functional score
        functional_score = calculate_pattern_score(description, extractor.functional_patterns)
        
        # Calculate non-functional score
        non_functional_score = calculate_pattern_score(description, extractor.non_functional_patterns)
        
        # Determine classification
        if functional_score > non_functional_score:
            req['classification'] = 'functional'
            req['classification_confidence'] = functional_score
        elif non_functional_score > 0:
            req['classification'] = 'non_functional'
            req['classification_confidence'] = non_functional_score
        else:
            req['classification'] = 'other'
            req['classification_confidence'] = 0.5
        
        # Add sub-classification for non-functional requirements
        if req['classification'] == 'non_functional':
            req['sub_classification'] = classify_non_functional_type(description)
    
    # Generate classification summary
    classification_summary = generate_classification_summary(requirements)
    
    for req in requirements:
        req['classification_summary'] = classification_summary
    
    logger.info("Requirements classification completed.")
    return requirements

def generate_requirements_report(requirements_data: Dict[str, Any], 
                               output_format: str = 'json') -> str:
    """
    Generate a comprehensive requirements report.
    
    Args:
        requirements_data: Complete requirements extraction results
        output_format: Output format ('json', 'html', 'markdown')
        
    Returns:
        Formatted requirements report as string
    """
    logger.info(f"Generating requirements report in {output_format} format...")
    
    if output_format == 'json':
        return generate_json_report(requirements_data)
    elif output_format == 'html':
        return generate_html_report(requirements_data)
    elif output_format == 'markdown':
        return generate_markdown_report(requirements_data)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def create_traceability_matrix(requirements: List[Dict[str, Any]], 
                             multimedia_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create traceability matrix linking requirements to source data.
    
    Args:
        requirements: List of final requirements
        multimedia_data: Original multimedia analysis results
        
    Returns:
        Traceability matrix with mappings
    """
    logger.info("Creating traceability matrix...")
    
    traceability = {
        'requirements_to_sources': {},
        'sources_to_requirements': defaultdict(list),
        'modality_coverage': {},
        'source_files': extract_source_files(multimedia_data)
    }
    
    for req in requirements:
        req_id = req.get('id')
        if not req_id:
            continue
        
        # Map requirement to source modalities and evidence
        sources = []
        supporting_modalities = req.get('supporting_modalities', [])
        
        for modality in supporting_modalities:
            modality_data = multimedia_data.get(modality, {})
            source_info = {
                'modality': modality,
                'evidence_type': determine_evidence_type(req, modality),
                'confidence': req.get(f'{modality}_confidence', 0),
                'source_files': modality_data.get('source_files', []),
                'timestamps': req.get('timestamps', {}).get(modality, [])
            }
            sources.append(source_info)
            
            # Reverse mapping
            traceability['sources_to_requirements'][modality].append({
                'requirement_id': req_id,
                'description': req.get('description', ''),
                'confidence': source_info['confidence']
            })
        
        traceability['requirements_to_sources'][req_id] = sources
    
    # Calculate modality coverage
    total_reqs = len(requirements)
    for modality in ['audio', 'video', 'image']:
        modality_reqs = len(traceability['sources_to_requirements'][modality])
        traceability['modality_coverage'][modality] = {
            'count': modality_reqs,
            'percentage': (modality_reqs / total_reqs * 100) if total_reqs > 0 else 0
        }
    
    logger.info("Traceability matrix created successfully.")
    return traceability

# Helper functions

def extract_audio_requirements(audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from audio analysis results."""
    requirements = []
    
    transcript = audio_data.get('transcript', '')
    sentiment = audio_data.get('sentiment', {})
    keywords = audio_data.get('keywords', [])
    entities = audio_data.get('entities', [])
    
    # Extract requirements from transcript using NLP
    transcript_requirements = extract_requirements_from_text(
        transcript, 'audio_transcript'
    )
    requirements.extend(transcript_requirements)
    
    # Extract requirements based on sentiment analysis
    sentiment_requirements = extract_sentiment_based_requirements(sentiment)
    requirements.extend(sentiment_requirements)
    
    # Extract requirements from keywords and entities
    keyword_requirements = extract_keyword_based_requirements(keywords, entities)
    requirements.extend(keyword_requirements)
    
    # Add audio-specific metadata
    for req in requirements:
        req['modality'] = 'audio'
        req['source_data'] = {
            'transcript_length': len(transcript),
            'keyword_count': len(keywords),
            'entity_count': len(entities),
            'sentiment_score': sentiment.get('compound', 0)
        }
    
    return requirements

def extract_video_requirements(video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from video analysis results."""
    requirements = []
    
    interactions = video_data.get('interactions', [])
    hesitation_points = video_data.get('hesitation_points', [])
    confusion_indicators = video_data.get('confusion_indicators', [])
    ui_elements = video_data.get('ui_elements', [])
    
    # Extract requirements from user interactions
    interaction_requirements = extract_interaction_requirements(interactions)
    requirements.extend(interaction_requirements)
    
    # Extract requirements from hesitation patterns
    hesitation_requirements = extract_hesitation_requirements(hesitation_points)
    requirements.extend(hesitation_requirements)
    
    # Extract requirements from confusion indicators
    confusion_requirements = extract_confusion_requirements(confusion_indicators)
    requirements.extend(confusion_requirements)
    
    # Extract requirements from UI element analysis
    ui_requirements = extract_ui_requirements(ui_elements)
    requirements.extend(ui_requirements)
    
    # Add video-specific metadata
    for req in requirements:
        req['modality'] = 'video'
        req['source_data'] = {
            'interaction_count': len(interactions),
            'hesitation_count': len(hesitation_points),
            'confusion_count': len(confusion_indicators),
            'ui_element_count': len(ui_elements)
        }
    
    return requirements

def extract_image_requirements(image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from image analysis results."""
    requirements = []
    
    if isinstance(image_data, list):
        # Handle multiple images
        for img_data in image_data:
            img_requirements = extract_single_image_requirements(img_data)
            requirements.extend(img_requirements)
    else:
        # Handle single image
        img_requirements = extract_single_image_requirements(image_data)
        requirements.extend(img_requirements)
    
    # Add image-specific metadata
    for req in requirements:
        req['modality'] = 'image'
    
    return requirements

def extract_single_image_requirements(image_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from single image analysis."""
    requirements = []
    
    extracted_text = image_data.get('extracted_text', {})
    ui_elements = image_data.get('ui_elements', [])
    annotations = image_data.get('annotations', [])
    visual_analysis = image_data.get('visual_analysis', {})
    
    # Extract from image requirements (if already processed)
    img_requirements = image_data.get('requirements', [])
    requirements.extend(img_requirements)
    
    # Extract additional requirements from text
    if extracted_text.get('full_text'):
        text_requirements = extract_requirements_from_text(
            extracted_text['full_text'], 'image_ocr'
        )
        requirements.extend(text_requirements)
    
    # Add source data metadata
    for req in requirements:
        req['source_data'] = {
            'text_length': len(extracted_text.get('full_text', '')),
            'ui_element_count': len(ui_elements),
            'annotation_count': len(annotations),
            'image_quality': image_data.get('quality_score', 0)
        }
    
    return requirements

def group_similar_requirements(requirements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group similar requirements for cross-modal validation."""
    if not SKLEARN_AVAILABLE:
        # Fallback: simple keyword-based grouping
        return simple_keyword_grouping(requirements)
    
    # Extract requirement descriptions
    descriptions = [req.get('description', '') for req in requirements]
    
    if not descriptions or all(not desc.strip() for desc in descriptions):
        return [[req] for req in requirements]  # Each requirement in its own group
    
    try:
        # Vectorize descriptions
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group requirements based on similarity threshold
        similarity_threshold = 0.6
        groups = []
        used = [False] * len(requirements)
        
        for i, req in enumerate(requirements):
            if used[i]:
                continue
            
            group = [req]
            used[i] = True
            
            for j in range(i + 1, len(requirements)):
                if used[j]:
                    continue
                
                if similarity_matrix[i][j] > similarity_threshold:
                    group.append(requirements[j])
                    used[j] = True
            
            groups.append(group)
        
        return groups
        
    except Exception as e:
        logger.error(f"Similarity-based grouping failed: {e}")
        return simple_keyword_grouping(requirements)

def simple_keyword_grouping(requirements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Simple keyword-based grouping fallback method."""
    keyword_groups = defaultdict(list)
    
    for req in requirements:
        description = req.get('description', '').lower()
        
        # Extract key terms
        key_terms = extract_key_terms(description)
        
        # Group by most significant term
        if key_terms:
            primary_term = key_terms[0]
            keyword_groups[primary_term].append(req)
        else:
            keyword_groups['misc'].append(req)
    
    return list(keyword_groups.values())

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from requirement description."""
    # Common requirement keywords
    key_patterns = [
        r'button', r'menu', r'field', r'form', r'input', r'search', r'filter',
        r'login', r'register', r'profile', r'dashboard', r'navigation',
        r'performance', r'security', r'usability', r'accessibility'
    ]
    
    found_terms = []
    for pattern in key_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found_terms.append(pattern)
    
    return found_terms

def validate_requirement_group(group: List[Dict[str, Any]], group_id: int) -> Optional[Dict[str, Any]]:
    """Validate a group of similar requirements and merge into single requirement."""
    if not group:
        return None
    
    if len(group) == 1:
        # Single requirement - validate based on individual criteria
        req = group[0]
        confidence = calculate_single_requirement_confidence(req)
        
        if confidence >= 0.4:  # Minimum threshold
            req['id'] = f"REQ-{group_id:03d}"
            req['confidence_score'] = confidence
            req['confidence_level'] = get_confidence_level(confidence)
            req['supporting_modalities'] = [req.get('modality', 'unknown')]
            req['validation_method'] = 'single_modality'
            return req
        else:
            return None
    
    # Multiple requirements - cross-modal validation
    modalities = [req.get('modality') for req in group]
    unique_modalities = list(set(modalities))
    
    # Calculate cross-modal confidence
    cross_modal_confidence = calculate_cross_modal_confidence(group, unique_modalities)
    
    if cross_modal_confidence >= 0.6:  # Higher threshold for multi-modal
        # Merge requirements
        merged_req = merge_requirement_group(group, group_id)
        merged_req['confidence_score'] = cross_modal_confidence
        merged_req['confidence_level'] = get_confidence_level(cross_modal_confidence)
        merged_req['supporting_modalities'] = unique_modalities
        merged_req['validation_method'] = 'cross_modal'
        return merged_req
    
    return None

def calculate_single_requirement_confidence(req: Dict[str, Any]) -> float:
    """Calculate confidence score for single requirement."""
    base_confidence = req.get('confidence', 0.5)
    
    # Adjust based on modality-specific factors
    modality = req.get('modality', 'unknown')
    
    if modality == 'audio':
        # Higher confidence for explicit verbal statements
        if any(word in req.get('description', '').lower() 
               for word in ['need', 'want', 'should', 'must', 'require']):
            base_confidence += 0.2
    
    elif modality == 'video':
        # Higher confidence for behavioral patterns
        if 'hesitation' in req.get('description', '').lower():
            base_confidence += 0.15
        if 'confusion' in req.get('description', '').lower():
            base_confidence += 0.15
    
    elif modality == 'image':
        # Higher confidence for explicit annotations
        if 'annotation' in req.get('extraction_method', ''):
            base_confidence += 0.25
    
    return min(base_confidence, 1.0)

def calculate_cross_modal_confidence(group: List[Dict[str, Any]], 
                                   modalities: List[str]) -> float:
    """Calculate confidence score for cross-modal requirement group."""
    extractor = RequirementsExtractor()
    
    # Base confidence from individual requirements
    individual_confidences = [calculate_single_requirement_confidence(req) for req in group]
    avg_individual_confidence = np.mean(individual_confidences)
    
    # Modality diversity bonus
    modality_bonus = 0
    if len(modalities) >= 3:
        modality_bonus = 0.3  # High confidence for 3+ modalities
    elif len(modalities) == 2:
        modality_bonus = 0.2  # Medium confidence for 2 modalities
    
    # Consistency bonus (similar descriptions across modalities)
    consistency_bonus = calculate_consistency_bonus(group)
    
    # Calculate weighted confidence
    cross_modal_confidence = (
        avg_individual_confidence * 0.6 +
        modality_bonus +
        consistency_bonus * 0.2
    )
    
    return min(cross_modal_confidence, 1.0)

def calculate_consistency_bonus(group: List[Dict[str, Any]]) -> float:
    """Calculate bonus for consistency across modalities."""
    descriptions = [req.get('description', '') for req in group]
    
    if len(descriptions) < 2:
        return 0
    
    # Simple consistency check based on common keywords
    all_words = []
    for desc in descriptions:
        words = set(desc.lower().split())
        all_words.append(words)
    
    # Find common words
    common_words = set.intersection(*all_words) if all_words else set()
    
    # Calculate consistency score
    avg_word_count = np.mean([len(words) for words in all_words])
    consistency_ratio = len(common_words) / avg_word_count if avg_word_count > 0 else 0
    
    return min(consistency_ratio, 0.3)  # Max 30% bonus

def merge_requirement_group(group: List[Dict[str, Any]], group_id: int) -> Dict[str, Any]:
    """Merge a group of similar requirements into single requirement."""
    # Use the most detailed description
    descriptions = [req.get('description', '') for req in group if req.get('description')]
    merged_description = max(descriptions, key=len) if descriptions else 'No description available'
    
    # Combine extraction methods
    methods = list(set(req.get('extraction_method', 'unknown') for req in group))
    
    # Combine source data
    source_data = {}
    for req in group:
        req_source = req.get('source_data', {})
        for key, value in req_source.items():
            if key in source_data:
                if isinstance(value, (int, float)):
                    source_data[key] = max(source_data[key], value)
                else:
                    source_data[key] = value
            else:
                source_data[key] = value
    
    # Get timestamps from all modalities
    timestamps = {}
    for req in group:
        modality = req.get('modality')
        if modality and 'timestamp' in req:
            timestamps[modality] = req['timestamp']
    
    merged_req = {
        'id': f"REQ-{group_id:03d}",
        'description': merged_description,
        'extraction_methods': methods,
        'source_data': source_data,
        'timestamps': timestamps,
        'merged_from': len(group),
        'original_requirements': [req.get('id', f'orig_{i}') for i, req in enumerate(group)]
    }
    
    return merged_req

def get_confidence_level(score: float) -> str:
    """Convert confidence score to level."""
    if score >= 0.8:
        return 'high'
    elif score >= 0.6:
        return 'medium'
    else:
        return 'low'

def calculate_pattern_score(text: str, patterns: List[str]) -> float:
    """Calculate pattern matching score for text."""
    matches = 0
    total_patterns = len(patterns)
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
    
    return matches / total_patterns if total_patterns > 0 else 0

def classify_non_functional_type(description: str) -> str:
    """Classify non-functional requirement sub-type."""
    text = description.lower()
    
    if any(word in text for word in ['performance', 'speed', 'response', 'time']):
        return 'performance'
    elif any(word in text for word in ['usability', 'user', 'friendly', 'intuitive']):
        return 'usability'
    elif any(word in text for word in ['security', 'privacy', 'encryption']):
        return 'security'
    elif any(word in text for word in ['reliability', 'availability', 'uptime']):
        return 'reliability'
    elif any(word in text for word in ['scalability', 'capacity', 'throughput']):
        return 'scalability'
    elif any(word in text for word in ['accessibility', 'responsive', 'mobile']):
        return 'accessibility'
    else:
        return 'other'

def generate_classification_summary(requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary of requirement classifications."""
    total_reqs = len(requirements)
    
    classifications = [req.get('classification', 'unknown') for req in requirements]
    classification_counts = Counter(classifications)
    
    summary = {
        'total_requirements': total_reqs,
        'functional_count': classification_counts.get('functional', 0),
        'non_functional_count': classification_counts.get('non_functional', 0),
        'other_count': classification_counts.get('other', 0),
        'functional_percentage': (classification_counts.get('functional', 0) / total_reqs * 100) if total_reqs > 0 else 0,
        'non_functional_percentage': (classification_counts.get('non_functional', 0) / total_reqs * 100) if total_reqs > 0 else 0
    }
    
    # Sub-classification summary for non-functional requirements
    nf_reqs = [req for req in requirements if req.get('classification') == 'non_functional']
    nf_sub_classifications = [req.get('sub_classification', 'unknown') for req in nf_reqs]
    nf_sub_counts = Counter(nf_sub_classifications)
    
    summary['non_functional_subcategories'] = dict(nf_sub_counts)
    
    return summary

def generate_final_requirement_set(classified_requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate final polished requirement set."""
    final_requirements = []
    
    for req in classified_requirements:
        # Clean and enhance requirement description
        enhanced_description = enhance_requirement_description(req)
        
        # Add MoSCoW priority
        moscow_priority = determine_moscow_priority(req)
        
        # Add effort estimation
        effort_estimate = estimate_implementation_effort(req)
        
        # Create final requirement
        final_req = {
            'id': req.get('id'),
            'title': generate_requirement_title(req),
            'description': enhanced_description,
            'classification': req.get('classification'),
            'sub_classification': req.get('sub_classification'),
            'priority': moscow_priority,
            'confidence_level': req.get('confidence_level'),
            'confidence_score': req.get('confidence_score'),
            'supporting_modalities': req.get('supporting_modalities', []),
            'validation_method': req.get('validation_method'),
            'effort_estimate': effort_estimate,
            'implementation_notes': generate_implementation_notes(req),
            'acceptance_criteria': generate_acceptance_criteria(req),
            'metadata': {
                'extraction_methods': req.get('extraction_methods', []),
                'source_data': req.get('source_data', {}),
                'timestamps': req.get('timestamps', {}),
                'original_requirements': req.get('original_requirements', [])
            }
        }
        
        final_requirements.append(final_req)
    
    # Sort by priority and confidence
    final_requirements.sort(key=lambda x: (
        get_priority_weight(x.get('priority', 'Could')),
        x.get('confidence_score', 0)
    ), reverse=True)
    
    return final_requirements

def enhance_requirement_description(req: Dict[str, Any]) -> str:
    """Enhance and clean requirement description."""
    description = req.get('description', 'No description available')
    
    # Clean up common artifacts
    description = re.sub(r'\s+', ' ', description)  # Multiple whitespace
    description = description.strip()
    
    # Ensure proper sentence structure
    if description and not description[0].isupper():
        description = description.capitalize()
    
    if description and not description.endswith('.'):
        description += '.'
    
    # Add context if from specific modality
    modalities = req.get('supporting_modalities', [])
    if len(modalities) == 1:
        modality = modalities[0]
        if modality == 'audio':
            description = f"Based on user feedback: {description}"
        elif modality == 'video':
            description = f"Based on user behavior analysis: {description}"
        elif modality == 'image':
            description = f"Based on user annotations: {description}"
    
    return description

def generate_requirement_title(req: Dict[str, Any]) -> str:
    """Generate concise title for requirement."""
    description = req.get('description', '')
    classification = req.get('classification', 'functional')
    
    # Extract key action/object from description
    title_patterns = [
        r'(implement|add|create|provide|enable|support|improve|enhance)\s+([^.]{1,50})',
        r'(system|user|interface)\s+(shall|should|must|will)\s+([^.]{1,50})',
        r'([^.]{1,50})\s+(feature|functionality|capability)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            title = match.group().strip()
            if len(title) > 60:
                title = title[:57] + '...'
            return title.capitalize()
    
    # Fallback: use first few words
    words = description.split()[:8]
    title = ' '.join(words)
    if len(title) > 60:
        title = title[:57] + '...'
    
    return title.capitalize()

def determine_moscow_priority(req: Dict[str, Any]) -> str:
    """Determine MoSCoW priority for requirement."""
    confidence = req.get('confidence_score', 0)
    modalities = req.get('supporting_modalities', [])
    classification = req.get('classification', 'functional')
    
    # High confidence + multiple modalities = Must have
    if confidence >= 0.8 and len(modalities) >= 2:
        return 'Must'
    
    # Security/critical functionality = Must have
    description = req.get('description', '').lower()
    if any(word in description for word in ['security', 'authentication', 'critical', 'essential']):
        return 'Must'
    
    # Medium-high confidence = Should have
    if confidence >= 0.6:
        return 'Should'
    
    # Lower confidence = Could have
    return 'Could'

def get_priority_weight(priority: str) -> int:
    """Get numeric weight for priority sorting."""
    weights = {'Must': 4, 'Should': 3, 'Could': 2, 'Won\'t': 1}
    return weights.get(priority, 0)

def estimate_implementation_effort(req: Dict[str, Any]) -> str:
    """Estimate implementation effort for requirement."""
    description = req.get('description', '').lower()
    classification = req.get('classification', 'functional')
    
    # Complex keywords indicate high effort
    high_effort_keywords = ['complex', 'integration', 'algorithm', 'intelligence', 'learning']
    medium_effort_keywords = ['interface', 'dashboard', 'report', 'analysis', 'processing']
    
    if any(word in description for word in high_effort_keywords):
        return 'High (8-13 story points)'
    elif any(word in description for word in medium_effort_keywords):
        return 'Medium (3-5 story points)'
    elif classification == 'non_functional':
        return 'Medium (3-8 story points)'  # Non-functional often more complex
    else:
        return 'Low (1-3 story points)'

def generate_implementation_notes(req: Dict[str, Any]) -> str:
    """Generate implementation notes for requirement."""
    modalities = req.get('supporting_modalities', [])
    
    notes = []
    
    if 'audio' in modalities:
        notes.append("Consider user verbal feedback and preferences")
    
    if 'video' in modalities:
        notes.append("Address user behavior patterns and interaction difficulties")
    
    if 'image' in modalities:
        notes.append("Implement user interface improvements based on annotations")
    
    if len(modalities) > 1:
        notes.append("Cross-modal validation confirms high priority")
    
    classification = req.get('classification')
    if classification == 'non_functional':
        sub_class = req.get('sub_classification', '')
        if sub_class == 'usability':
            notes.append("Focus on user experience and ease of use")
        elif sub_class == 'performance':
            notes.append("Ensure performance benchmarks are established")
    
    return '; '.join(notes) if notes else 'Standard implementation approach'

def generate_acceptance_criteria(req: Dict[str, Any]) -> List[str]:
    """Generate acceptance criteria for requirement."""
    criteria = []
    description = req.get('description', '').lower()
    classification = req.get('classification', 'functional')
    
    # Base criteria
    criteria.append("Implementation matches requirement description")
    criteria.append("Functionality is tested and verified")
    
    # Classification-specific criteria
    if classification == 'functional':
        criteria.append("Feature works as specified")
        criteria.append("User can complete intended task")
    
    elif classification == 'non_functional':
        sub_class = req.get('sub_classification', '')
        if sub_class == 'usability':
            criteria.append("User interface is intuitive and easy to use")
            criteria.append("User task completion time is reasonable")
        elif sub_class == 'performance':
            criteria.append("Performance meets specified benchmarks")
            criteria.append("System response time is acceptable")
        elif sub_class == 'security':
            criteria.append("Security measures are properly implemented")
            criteria.append("Data privacy is maintained")
    
    # Modality-specific criteria
    modalities = req.get('supporting_modalities', [])
    if 'video' in modalities:
        criteria.append("User interaction patterns are improved")
    
    if 'image' in modalities:
        criteria.append("Interface changes address user annotations")
    
    return criteria

def extract_requirements_from_text(text: str, source_type: str) -> List[Dict[str, Any]]:
    """Extract requirements from text using NLP patterns."""
    requirements = []
    
    # Requirement indicating phrases
    requirement_patterns = [
        r'(system|interface|platform)\s+(should|must|shall|will|needs? to)\s+([^.!?]{10,100})',
        r'(user|users?)\s+(should be able to|can|must be able to|need to)\s+([^.!?]{10,100})',
        r'(need|want|require|would like)\s+([^.!?]{10,100})',
        r'(add|include|implement|provide|create)\s+([^.!?]{10,100})',
        r'(improve|enhance|better|fix)\s+([^.!?]{10,100})'
    ]
    
    req_id = 0
    for pattern in requirement_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            req_text = match.group().strip()
            
            # Skip very short or generic matches
            if len(req_text.split()) < 4:
                continue
            
            req_id += 1
            requirements.append({
                'id': f"{source_type}-{req_id}",
                'description': req_text,
                'extraction_method': 'nlp_pattern',
                'source_type': source_type,
                'confidence': 0.7,
                'pattern_matched': pattern
            })
    
    return requirements

def extract_sentiment_based_requirements(sentiment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements based on sentiment analysis."""
    requirements = []
    
    compound_score = sentiment.get('compound', 0)
    
    # Negative sentiment might indicate problems/improvement areas
    if compound_score < -0.3:
        requirements.append({
            'id': 'SENTIMENT-1',
            'description': 'User expressed negative sentiment - investigate and address underlying issues',
            'type': 'non_functional',
            'sub_type': 'usability',
            'extraction_method': 'sentiment_analysis',
            'confidence': abs(compound_score),
            'sentiment_score': compound_score
        })
    
    return requirements

def extract_keyword_based_requirements(keywords: List[str], entities: List[Any]) -> List[Dict[str, Any]]:
    """Extract requirements based on keywords and entities."""
    requirements = []
    
    # Action keywords that suggest requirements
    action_keywords = [word for word in keywords if any(action in word.lower() 
                      for action in ['add', 'need', 'want', 'improve', 'fix', 'create'])]
    
    if action_keywords:
        requirements.append({
            'id': 'KEYWORD-1',
            'description': f'User mentioned action keywords: {", ".join(action_keywords[:5])}',
            'extraction_method': 'keyword_analysis',
            'confidence': 0.6,
            'keywords': action_keywords
        })
    
    return requirements

def extract_interaction_requirements(interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from user interactions."""
    requirements = []
    
    # Analyze interaction patterns
    if len(interactions) > 10:  # Many interactions might indicate complexity
        requirements.append({
            'id': 'INTERACTION-1',
            'description': 'High number of user interactions detected - consider simplifying workflow',
            'type': 'non_functional',
            'sub_type': 'usability',
            'extraction_method': 'interaction_analysis',
            'confidence': 0.7,
            'interaction_count': len(interactions)
        })
    
    return requirements

def extract_hesitation_requirements(hesitation_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from hesitation analysis."""
    requirements = []
    
    if hesitation_points:
        requirements.append({
            'id': 'HESITATION-1',
            'description': 'User hesitation patterns detected - improve interface clarity and guidance',
            'type': 'non_functional',
            'sub_type': 'usability',
            'extraction_method': 'hesitation_analysis',
            'confidence': 0.8,
            'hesitation_count': len(hesitation_points)
        })
    
    return requirements

def extract_confusion_requirements(confusion_indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from confusion analysis."""
    requirements = []
    
    if confusion_indicators:
        requirements.append({
            'id': 'CONFUSION-1',
            'description': 'User confusion indicators detected - enhance user interface and provide better guidance',
            'type': 'non_functional', 
            'sub_type': 'usability',
            'extraction_method': 'confusion_analysis',
            'confidence': 0.9,
            'confusion_count': len(confusion_indicators)
        })
    
    return requirements

def extract_ui_requirements(ui_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from UI element analysis."""
    requirements = []
    
    if len(ui_elements) > 15:  # Too many UI elements
        requirements.append({
            'id': 'UI-DENSITY-1',
            'description': 'High UI element density detected - consider interface simplification',
            'type': 'non_functional',
            'sub_type': 'usability',
            'extraction_method': 'ui_analysis',
            'confidence': 0.7,
            'ui_element_count': len(ui_elements)
        })
    
    return requirements

def determine_evidence_type(req: Dict[str, Any], modality: str) -> str:
    """Determine the type of evidence for a requirement from specific modality."""
    extraction_methods = req.get('extraction_methods', [])
    
    if modality == 'audio':
        if any('transcript' in method for method in extraction_methods):
            return 'verbal_statement'
        elif any('sentiment' in method for method in extraction_methods):
            return 'emotional_response'
        else:
            return 'audio_analysis'
    
    elif modality == 'video':
        if any('hesitation' in method for method in extraction_methods):
            return 'behavioral_hesitation'
        elif any('confusion' in method for method in extraction_methods):
            return 'behavioral_confusion'
        elif any('interaction' in method for method in extraction_methods):
            return 'interaction_pattern'
        else:
            return 'behavioral_analysis'
    
    elif modality == 'image':
        if any('annotation' in method for method in extraction_methods):
            return 'visual_annotation'
        elif any('text' in method for method in extraction_methods):
            return 'textual_annotation'
        else:
            return 'visual_analysis'
    
    return 'unknown'

def extract_source_files(multimedia_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract source file information from multimedia data."""
    source_files = {}
    
    for modality, data in multimedia_data.items():
        if isinstance(data, dict):
            files = data.get('source_files', [])
            if isinstance(files, str):
                files = [files]
            source_files[modality] = files
        elif isinstance(data, list):
            files = []
            for item in data:
                if isinstance(item, dict):
                    item_files = item.get('source_files', [])
                    if isinstance(item_files, str):
                        item_files = [item_files]
                    files.extend(item_files)
            source_files[modality] = files
    
    return source_files

def create_source_data_summary(multimedia_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary of source multimedia data."""
    summary = {
        'modalities_processed': len(multimedia_data),
        'modality_details': {}
    }
    
    for modality, data in multimedia_data.items():
        if isinstance(data, dict):
            modality_summary = {
                'data_type': type(data).__name__,
                'has_transcript': 'transcript' in data,
                'has_analysis_results': bool(data.get('requirements', [])),
                'quality_score': data.get('quality_score', 0)
            }
        elif isinstance(data, list):
            modality_summary = {
                'data_type': 'list',
                'item_count': len(data),
                'has_analysis_results': any(
                    item.get('requirements', []) for item in data 
                    if isinstance(item, dict)
                )
            }
        else:
            modality_summary = {
                'data_type': type(data).__name__,
                'has_analysis_results': False
            }
        
        summary['modality_details'][modality] = modality_summary
    
    return summary

def calculate_extraction_quality_metrics(requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate quality metrics for requirements extraction."""
    if not requirements:
        return {
            'total_requirements': 0,
            'avg_confidence': 0,
            'cross_modal_percentage': 0,
            'classification_distribution': {}
        }
    
    # Basic metrics
    total_reqs = len(requirements)
    confidences = [req.get('confidence_score', 0) for req in requirements]
    avg_confidence = np.mean(confidences)
    
    # Cross-modal validation percentage
    cross_modal_reqs = [req for req in requirements 
                       if len(req.get('supporting_modalities', [])) > 1]
    cross_modal_percentage = len(cross_modal_reqs) / total_reqs * 100
    
    # Classification distribution
    classifications = [req.get('classification', 'unknown') for req in requirements]
    classification_dist = Counter(classifications)
    
    # Confidence level distribution
    confidence_levels = [req.get('confidence_level', 'unknown') for req in requirements]
    confidence_dist = Counter(confidence_levels)
    
    return {
        'total_requirements': total_reqs,
        'avg_confidence': round(avg_confidence, 3),
        'min_confidence': round(min(confidences), 3) if confidences else 0,
        'max_confidence': round(max(confidences), 3) if confidences else 0,
        'cross_modal_percentage': round(cross_modal_percentage, 1),
        'classification_distribution': dict(classification_dist),
        'confidence_level_distribution': dict(confidence_dist),
        'quality_score': round((avg_confidence + cross_modal_percentage/100) / 2, 3)
    }

def generate_json_report(requirements_data: Dict[str, Any]) -> str:
    """Generate JSON format report."""
    return json.dumps(requirements_data, indent=2, default=str)

def generate_markdown_report(requirements_data: Dict[str, Any]) -> str:
    """Generate Markdown format report."""
    requirements = requirements_data.get('requirements', [])
    quality_metrics = requirements_data.get('quality_metrics', {})
    traceability = requirements_data.get('traceability_matrix', {})
    
    report = f"""# Multimedia Requirements Engineering Report

Generated on: {requirements_data.get('timestamp', 'Unknown')}

## Executive Summary

- **Total Requirements**: {quality_metrics.get('total_requirements', 0)}
- **Average Confidence**: {quality_metrics.get('avg_confidence', 0):.3f}
- **Cross-Modal Validation**: {quality_metrics.get('cross_modal_percentage', 0):.1f}%
- **Quality Score**: {quality_metrics.get('quality_score', 0):.3f}

## Requirements Classification

"""
    
    # Classification distribution
    classification_dist = quality_metrics.get('classification_distribution', {})
    for class_type, count in classification_dist.items():
        percentage = (count / quality_metrics.get('total_requirements', 1)) * 100
        report += f"- **{class_type.title()}**: {count} ({percentage:.1f}%)\n"
    
    report += "\n## Requirements List\n\n"
    
    # Requirements details
    for i, req in enumerate(requirements, 1):
        report += f"### REQ-{i:03d}: {req.get('title', 'No Title')}\n\n"
        report += f"**Description**: {req.get('description', 'No description')}\n\n"
        report += f"**Classification**: {req.get('classification', 'Unknown')}\n\n"
        report += f"**Priority**: {req.get('priority', 'Unknown')}\n\n"
        report += f"**Confidence**: {req.get('confidence_level', 'Unknown')} ({req.get('confidence_score', 0):.3f})\n\n"
        
        modalities = req.get('supporting_modalities', [])
        report += f"**Supporting Modalities**: {', '.join(modalities)}\n\n"
        
        criteria = req.get('acceptance_criteria', [])
        if criteria:
            report += "**Acceptance Criteria**:\n"
            for criterion in criteria:
                report += f"- {criterion}\n"
            report += "\n"
        
        report += "---\n\n"
    
    return report

def generate_html_report(requirements_data: Dict[str, Any]) -> str:
    """Generate HTML format report."""
    # Simplified HTML report
    requirements = requirements_data.get('requirements', [])
    quality_metrics = requirements_data.get('quality_metrics', {})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multimedia Requirements Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .requirement {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
            .high-confidence {{ border-left: 4px solid #28a745; }}
            .medium-confidence {{ border-left: 4px solid #ffc107; }}
            .low-confidence {{ border-left: 4px solid #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multimedia Requirements Engineering Report</h1>
            <p>Generated on: {requirements_data.get('timestamp', 'Unknown')}</p>
            <p>Total Requirements: {quality_metrics.get('total_requirements', 0)}</p>
            <p>Quality Score: {quality_metrics.get('quality_score', 0):.3f}</p>
        </div>
    """
    
    for req in requirements:
        confidence_class = req.get('confidence_level', 'low') + '-confidence'
        html += f"""
        <div class="requirement {confidence_class}">
            <h3>{req.get('title', 'No Title')}</h3>
            <p><strong>Description:</strong> {req.get('description', 'No description')}</p>
            <p><strong>Classification:</strong> {req.get('classification', 'Unknown')}</p>
            <p><strong>Priority:</strong> {req.get('priority', 'Unknown')}</p>
            <p><strong>Confidence:</strong> {req.get('confidence_level', 'Unknown')}</p>
            <p><strong>Supporting Modalities:</strong> {', '.join(req.get('supporting_modalities', []))}</p>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html
