import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Inicializa captura de vídeo
cap = cv2.VideoCapture(0)

# Configura MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Lista de classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N',  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'O']

# Carrega modelo treinado
model = load_model('models/keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Coleta coordenadas extremas
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
            y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

            # Desenha retângulo ao redor da mão
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            try:
                # Recorta e prepara imagem para modelo
                img_crop = img[y_min:y_max, x_min:x_max]
                img_resized = cv2.resize(img_crop, (224, 224))
                img_array = np.asarray(img_resized, dtype=np.float32)
                normalized_img = (img_array / 127.0) - 1
                data[0] = normalized_img

                # Realiza predição
                prediction = model.predict(data, verbose=0)
                class_index = np.argmax(prediction)
                confidence = prediction[0][class_index] * 100
                label = f"{classes[class_index]} ({confidence:.2f}%)"

                # Exibe a label ao lado da mão
                cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            except Exception as e:
                print(f"Erro no processamento da imagem: {e}")
                continue

            # Desenha landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Reconhecimento de Gestos com MediaPipe', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break  # Pressione ESC para sair

# Libera recursos
cap.release()
cv2.destroyAllWindows()
