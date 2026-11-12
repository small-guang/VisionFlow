import numpy as np
import cv2

from core.algorithm_config import algorithm_config
from .quality_reasons import QualityReason






def calculate_pose_from_landmarks(landmarks, image_shape):
    if image_shape is None or len(image_shape) < 2:
        return None

    image_h = float(image_shape[0])
    image_w = float(image_shape[1])
    if image_h <= 0 or image_w <= 0:
        return None

    object_pts = np.array([
        [38.2946, 51.6963, 39.5311], # 左眼
        [73.5318, 51.5014, 36.1781], # 右眼
        [56.0252, 71.7366, 68.4771], # 鼻子
        [41.5493, 92.3655, 35.8782], # 左嘴角
        [70.7299, 92.2041, 36.0337]  # 右嘴角
    ], dtype=np.float64)


    image_pts = np.array(landmarks, dtype=np.float64)
    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        object_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    pitch = np.rad2deg(x)
    yaw = np.rad2deg(y)
    roll = np.rad2deg(z)
    
    return pitch, yaw, roll


def calculate_laplacian_variance(image: np.ndarray) -> float:
    """
    Calculate the variance of the Laplacian of the image, as a measure of blur.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

def is_geometrically_consistent(
    keypoints: np.ndarray,
    bbox: list,
    logger,
    image_shape: tuple[int, ...] | None = None,
    min_face_width: float = algorithm_config.face.quality.min_face_width_px,
    max_pose_abs_degrees: float = algorithm_config.face.quality.max_pose_abs_degrees,
    geometry_std_ratio_threshold: float = algorithm_config.face.quality.geometry_std_ratio_threshold,
) -> tuple[bool, QualityReason | None]:

    if bbox[2]-bbox[0] <= min_face_width:
        reason = QualityReason.FACE_WIDTH_TOO_NARROW
        return False, reason
    
    for point in keypoints:
        if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
            reason = QualityReason.KEYPOINTS_OUTSIDE_BBOX
            return False, reason

    if len(keypoints) >= 5:
        right_eye = keypoints[0]
        left_eye = keypoints[1]
        nose_tip = keypoints[2]
        right_mouth = keypoints[3]
        left_mouth = keypoints[4]

        # 鼻尖关键点在水平方向上处于双眼关键点之间
        eye_x_min = min(left_eye[0], right_eye[0])
        eye_x_max = max(left_eye[0], right_eye[0])
        if not (eye_x_min <= nose_tip[0] <= eye_x_max):
            reason = QualityReason.NOSE_OUTSIDE_EYES_HORIZONTAL
            logger.debug(f"Face rejected: {reason}. Nose_x: {nose_tip[0]}, Eye_x_range: [{eye_x_min}, {eye_x_max}]")
            return False, reason

        # 鼻尖关键点在垂直方向上处于眼睛和嘴角之间
        avg_eye_y = (left_eye[1] + right_eye[1]) / 2
        avg_mouth_y = (left_mouth[1] + right_mouth[1]) / 2
        if not (avg_eye_y < nose_tip[1] < avg_mouth_y):
            reason = QualityReason.NOSE_OUTSIDE_EYES_MOUTH_VERTICAL
            logger.debug(f"Face rejected: {reason}. Nose_y: {nose_tip[1]}, Avg_eye_y: {avg_eye_y}, Avg_mouth_y: {avg_mouth_y}")
            return False, reason


        pose = calculate_pose_from_landmarks(keypoints, image_shape)
        if pose is not None:
            pitch, yaw, roll = pose
            if (
                abs(pitch) > max_pose_abs_degrees
                or abs(yaw) > max_pose_abs_degrees
                or abs(roll) > max_pose_abs_degrees
            ):
                reason = QualityReason.BAD_POSE
                logger.debug(f"Face rejected due to bad pose: pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}")
                return False, reason
 
    distances = []
    for i in range(len(keypoints)):
        for j in range(i + 1, len(keypoints)):
            distances.append(np.linalg.norm(keypoints[i] - keypoints[j]))
    
    if not distances or len(distances) <= 1:
        return True, None

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    if std_distance > mean_distance * geometry_std_ratio_threshold:
        reason = QualityReason.GEOMETRIC_INSTABILITY
        logger.debug(f"Face rejected due to geometric instability (std: {std_distance:.2f}, mean: {mean_distance:.2f})")
        return False, reason
        
    return True, None

def is_high_quality_feature(
    embedding: np.ndarray,
    norm_threshold: float = algorithm_config.face.quality.feature_norm_threshold,
) -> tuple[bool, QualityReason | None]:
    """
    Filter low-quality features based on the L2 norm of the embedding.
    Returns a tuple: (is_high_quality, QualityReason_or_None).
    """
    norm = np.linalg.norm(embedding)
    if norm <= norm_threshold:
        reason = QualityReason.LOW_FEATURE_QUALITY_NORM
        return False, reason
    return True, None

def meets_pre_extraction_quality_criteria(
    face_image: np.ndarray,
    landmarks: np.ndarray,
    bbox: list,
    logger,
    laplacian_threshold: float = algorithm_config.face.quality.laplacian_threshold,
    pose_image_shape: tuple[int, ...] | None = None,
    min_face_width: float = algorithm_config.face.quality.min_face_width_px,
    max_pose_abs_degrees: float = algorithm_config.face.quality.max_pose_abs_degrees,
    geometry_std_ratio_threshold: float = algorithm_config.face.quality.geometry_std_ratio_threshold,
) -> tuple[bool, QualityReason | None]:
    """
    Performs pre-feature-extraction quality checks (blur, geometric consistency).
    Returns a tuple: (is_good_quality, QualityReason_or_None).
    """
    laplacian_var = calculate_laplacian_variance(face_image)
    if laplacian_var < laplacian_threshold:
        reason = QualityReason.BLURRY_LAPLACIAN
        logger.debug(f"Face rejected due to blurriness (Laplacian Var: {laplacian_var:.2f})")
        return False, reason

    pose_image_shape = pose_image_shape or getattr(face_image, "shape", None)
    is_consistent, reason = is_geometrically_consistent(
        landmarks,
        bbox,
        logger,
        pose_image_shape,
        min_face_width=min_face_width,
        max_pose_abs_degrees=max_pose_abs_degrees,
        geometry_std_ratio_threshold=geometry_std_ratio_threshold,
    )
    if not is_consistent:
        return False, reason
        
    return True, None  
