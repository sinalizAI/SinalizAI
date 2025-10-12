# ==============================================================================
# SCRIPT MOVINET COM JANELAS DE CONTEXTO E GRAVAÇÃO
# ==============================================================================
import os
import tarfile
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# --- 1. CONFIGURAÇÃO ---
MODEL_PATH = "movinet_libras_final_base.keras"
ARCHIVE_PATH = "movinet-tensorflow2-a0-base-kinetics-600-classification-v3.tar.gz"
MODEL_EXTRACT_PATH = "movinet_a0_base_classification"

CLASS_NAMES = sorted([
    'A', 'ABACAXI', 'ABANAR', 'ABANDONAR', 'ABELHA', 'ABENCOAR',
    'ABOBORA', 'ABORTO', 'ABRACO', 'ABRIR_JANELA', 'ABRIR_PORTA',
    'ACABAR', 'ANIMAL_MIMADO', 'A_NOITE_TODA', 'A_TARDE_TODA'
])

# Parâmetros
FRAME_COUNT = 16
HEIGHT = 172
WIDTH = 172
CONFIDENCE_THRESHOLD = 0.70

# Parâmetros da janela de contexto
RECORDING_DURATION = 4    # Duração da gravação em segundos
COOLDOWN_DURATION = 3     # Tempo de espera após a previsão em segundos

# --- 2. FUNÇÃO AUXILIAR DE PRÉ-PROCESSAMENTO ---
def preprocess_frame(frame, image_size=(HEIGHT, WIDTH)):
    """Pré-processa um único frame da webcam."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.float32)
    frame_resized = tf.image.resize_with_pad(frame_tf, image_size[0], image_size[1])
    return frame_resized

# --- 3. SCRIPT PRINCIPAL ---
def main():
    # --- PREPARAÇÃO E CARREGAMENTO DO MODELO ---
    print("--- Verificando a presença do modelo base... ---")
    if not os.path.exists(MODEL_EXTRACT_PATH):
        print(f"--- [INFO] Descompactando o modelo base de '{ARCHIVE_PATH}'... ---")
        if os.path.exists(ARCHIVE_PATH):
            os.makedirs(MODEL_EXTRACT_PATH, exist_ok=True)
            with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall(path=MODEL_EXTRACT_PATH, filter='data')
            print(f"--- [INFO] Modelo base descompactado! ---")
        else:
            print(f"ERRO: O arquivo .tar.gz do modelo base não foi encontrado: {ARCHIVE_PATH}")
            return
    else:
        print(f"--- [INFO] Pasta do modelo base já existe. ---")

    print(f"\n--- Carregando seu modelo final de '{MODEL_PATH}'... ---")
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: O seu arquivo .keras treinado não foi encontrado: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Seu modelo foi carregado! Iniciando a webcam...")
    
    # --- MÁQUINA DE ESTADOS E EXECUÇÃO ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a webcam.")
        return

    # Variáveis da máquina de estados
    current_state = "WAITING" # Estados: WAITING, RECORDING, PROCESSING, COOLDOWN
    recorded_frames = []
    recording_start_time = 0
    cooldown_start_time = 0
    prediction_result = ""

    print("\n Pressione 'G' para iniciar a gravação.")
    print(" Pressione 'Q' para sair.")
    print("\n IMPORTANTE: Clique na janela da webcam para que ela receba os comandos do teclado.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = cv2.flip(frame, 1)

            # Lógica da Máquina de Estados
            if current_state == "WAITING":
                display_text = f"Pressione 'G' para gravar por {RECORDING_DURATION}s"
            
            elif current_state == "RECORDING":
                elapsed = time.time() - recording_start_time
                countdown = RECORDING_DURATION - elapsed
                display_text = f"GRAVANDO... {int(countdown)+1}s"
                
                processed_frame = preprocess_frame(display_frame)
                recorded_frames.append(processed_frame)
                
                if elapsed >= RECORDING_DURATION:
                    current_state = "PROCESSING"

            elif current_state == "PROCESSING":
                display_text = "Processando..."
                
                if len(recorded_frames) > 0:
                    # Amostragem dos frames gravados para o tamanho que o modelo espera (FRAME_COUNT)
                    indices = np.linspace(0, len(recorded_frames) - 1, FRAME_COUNT, dtype=int)
                    sequence_to_predict = [recorded_frames[i] for i in indices]
                    
                    input_tensor = np.expand_dims(sequence_to_predict, axis=0)
                    
                    # Previsão
                    predictions = model.predict(input_tensor, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_index]

                    if confidence > CONFIDENCE_THRESHOLD:
                        predicted_class = CLASS_NAMES[predicted_index]
                        prediction_result = f"{predicted_class} ({confidence:.2f})"
                    else:
                        prediction_result = "Nao identificado"
                else:
                    prediction_result = "Nenhuma gravacao"

                current_state = "COOLDOWN"
                cooldown_start_time = time.time()
                
            elif current_state == "COOLDOWN":
                elapsed = time.time() - cooldown_start_time
                display_text = f"Resultado: {prediction_result}"

                if elapsed >= COOLDOWN_DURATION:
                    current_state = "WAITING"

            # Exibe o estado na tela
            cv2.rectangle(display_frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(display_frame, display_text, (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Reconhecimento de LIBRAS com Janelas (MoViNet)', display_frame)

            # Captura a tecla pressionada
            key = cv2.waitKey(1) & 0xFF

            if current_state == "WAITING" and key == ord('g'):
                current_state = "RECORDING"
                recorded_frames = []
                recording_start_time = time.time()
            
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n--- Câmera desligada. Programa encerrado. ---")

if __name__ == '__main__':
    main()