from scipy.spatial import distance


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_eye_landmarks(face_landmarks, eye_indices, image_shape):
    h, w = image_shape
    eye_points = []
    for idx in eye_indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        eye_points.append((x, y))
    return eye_points


def calculate_fatigue_score(
    ear,
    perclos,
    closure_duration
):

    score = (
        (1 - min(ear,0.4)/0.4)*40
        +
        min(perclos,100)*0.4
        +
        min(closure_duration,5)*10
    )

    score = max(
        0,
        min(100,score)
    )

    return round(score,2)


def get_recommendation(score):

    if score < 30:
        return "Driver is focused."

    elif score < 60:
        return "Monitor driver condition."

    return "Take a break immediately."