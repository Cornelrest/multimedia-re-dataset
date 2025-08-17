# Listing: Video Quality Control Implementation (lst:video-quality)
def process_video_with_quality_control(video_file: str) -> Dict[str, Any]:
    """
    Process video with built-in quality assessment
    """
    quality_metrics = assess_video_quality(video_file)
    
    if quality_metrics['overall_score'] < 0.7:
        logging.warning(f"Low quality video: {video_file}")
        return process_low_quality_video(video_file, quality_metrics)
    
    return process_high_quality_video(video_file)

def assess_video_quality(video_file: str) -> Dict[str, float]:
    """
    Assess video quality across multiple dimensions
    """
    cap = cv2.VideoCapture(video_file)
    
    # Check resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution_score = min(1.0, (width * height) / (1920 * 1080))
    
    # Check frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_score = min(1.0, fps / 30.0)
    
    # Check lighting stability
    lighting_score = assess_lighting_stability(cap)
    
    # Check motion blur
    blur_score = assess_motion_blur(cap)
    
    cap.release()
    
    overall_score = np.mean([resolution_score, fps_score, lighting_score, blur_score])
    
    return {
        'resolution_score': resolution_score,
        'fps_score': fps_score,
        'lighting_score': lighting_score,
        'blur_score': blur_score,
        'overall_score': overall_score
    }
