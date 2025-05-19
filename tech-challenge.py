import cv2
import mediapipe as mp
from deepface import DeepFace
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

video_path = "video.mp4"
output_path = "result.mp4"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Não foi possível abrir o vídeo.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

total_frames = 0
emotion_log = []
activity_log = []
anomaly_log = []
face_expression_sequence = []
expression_threshold = 5

def frame_to_time(frame_number):
    seconds = frame_number / fps
    return str(datetime.utcfromtimestamp(seconds).strftime('%M:%S'))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    if total_frames % 100 == 0:
        print(f"[INFO] Processando frame {total_frames}...")

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    pose_results = pose.process(frame_rgb)
    activity = None
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            activity = f"Bracos levantados"
            activity_log.append((total_frames, activity))
            cv2.putText(frame, activity, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Face & emotion
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        bbox_x = [int(l.x * w) for l in face_results.multi_face_landmarks[0].landmark]
        bbox_y = [int(l.y * h) for l in face_results.multi_face_landmarks[0].landmark]
        x1, x2 = max(min(bbox_x) - 20, 0), min(max(bbox_x) + 20, w)
        y1, y2 = max(min(bbox_y) - 20, 0), min(max(bbox_y) + 20, h)
        face_crop = frame[y1:y2, x1:x2]

        try:
            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)[0]
            emotion = result['dominant_emotion']
        except:
            emotion = "erro"

        emotion_log.append((total_frames, emotion))

        # Melhor detecção de caretas com tolerância de quebra
        if emotion in ["angry", "disgust", "fear", "surprise"]:
            face_expression_sequence.append(total_frames)
        else:
            if len(face_expression_sequence) >= expression_threshold:
                diffs = np.diff(face_expression_sequence)
                if np.all(diffs <= 2):  # tolera 1 frame de quebra
                    start_frame = face_expression_sequence[0]
                    end_frame = face_expression_sequence[-1]
                    anomaly_log.append((start_frame, end_frame, "Careta detectada"))
                    cv2.putText(frame, "Careta detectada", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            face_expression_sequence = []

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
pose.close()
face_mesh.close()

# Contagem e porcentagem de emoções
emotion_counts = Counter([e for _, e in emotion_log if e != "erro"])
emotion_total = sum(emotion_counts.values())

emotion_percentages = {
    emo: round((count / emotion_total) * 100, 1)
    for emo, count in emotion_counts.items()
}

# Gráfico de pizza
if emotion_percentages:
    labels = list(emotion_percentages.keys())
    sizes = list(emotion_percentages.values())
    colors = plt.cm.tab20.colors[:len(labels)]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.axis('equal')
    plt.title("Distribuição das Emoções Detectadas")
    plt.savefig("grafico_emocoes.png")
    plt.close()

# Relatório
with open("relatorio.txt", "w", encoding="utf-8") as f:
    f.write(f"Data da geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total de frames analisados: {total_frames}\n\n")

    f.write("Atividades detectadas:\n")
    for frame_num, act in activity_log:
        f.write(f" - {act} (frame {frame_num}, tempo {frame_to_time(frame_num)})\n")

    f.write("\nDistribuição das emoções detectadas:\n")
    if emotion_percentages:
        for emo, perc in sorted(emotion_percentages.items(), key=lambda x: -x[1]):
            f.write(f" - {emo}: {perc}%\n")
    else:
        f.write(" - Nenhuma emoção detectada com sucesso.\n")

    f.write("\nAnomalias detectadas:\n")
    if anomaly_log:
        for start, end, msg in anomaly_log:
            f.write(f" - {msg} dos frames {start} a {end} (de {frame_to_time(start)} até {frame_to_time(end)})\n")
    else:
        f.write(" - Nenhuma anomalia identificada com consistência suficiente.\n")
