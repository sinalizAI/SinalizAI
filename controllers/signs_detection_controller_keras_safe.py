
from camera4kivy import Preview
from utils.base_screen import BaseScreen
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
import numpy as np
import cv2
import os
import tensorflow as tf
from pathlib import Path
import time
from collections import deque


SIGNS_CLASSES = sorted([
    'A', 'ABACAXI', 'ABANAR', 'ABANDONAR', 'ABELHA', 'ABENCOAR',
    'ABOBORA', 'ABORTO', 'ABRACO', 'ABRIR_JANELA', 'ABRIR_PORTA',
    'ACABAR', 'ANIMAL_MIMADO', 'A_NOITE_TODA', 'A_TARDE_TODA'
])

class SignsDetectionPreview(Preview):
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_screen = None
        self.current_frame = None
        
    def set_parent_screen(self, screen):
        
        self.parent_screen = screen
        
    def analyze_pixels_callback(self, pixels, image_size, image_pos, image_scale, mirror):
        
        try:

            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(image_size[1], image_size[0], 4)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            

            self.current_frame = bgr
            

            if self.parent_screen:
                self.parent_screen.process_new_frame(bgr)
                
        except Exception as e:
            print(f" Erro na captura de frame: {e}")


class SignsDetectionScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        self.FRAME_COUNT = 16
        self.HEIGHT = 172
        self.WIDTH = 172
        self.CONFIDENCE_THRESHOLD = 0.70
        

        self.RECORDING_DURATION = 4
        self.COOLDOWN_DURATION = 3
        

        self.current_state = "WAITING"
        self.recorded_frames = []
        self.recording_start_time = 0
        self.cooldown_start_time = 0
        self.prediction_result = ""
        

        self.model = None
        self.model_loaded = False
        

        self.preview = None
        

        Clock.schedule_once(self.load_model, 0.1)
    
    def load_model(self, dt):
        
        try:
            print(" Iniciando carregamento do modelo Keras...")
            

            MODEL_PATH = "movinet_libras_final_base.keras"
            ARCHIVE_PATH = "movinet-tensorflow2-a0-base-kinetics-600-classification-v3.tar.gz"
            MODEL_EXTRACT_PATH = "movinet_a0_base_classification"
            

            original_dir = os.getcwd()
            ml_dir = os.path.join("services", "ml")
            os.chdir(ml_dir)
            
            try:

                print("--- Verificando a presença do modelo base... ---")
                if not os.path.exists(MODEL_EXTRACT_PATH):
                    print(f"--- [INFO] Descompactando o modelo base de '{ARCHIVE_PATH}'... ---")
                    if os.path.exists(ARCHIVE_PATH):
                        import tarfile
                        os.makedirs(MODEL_EXTRACT_PATH, exist_ok=True)
                        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
                            tar.extractall(path=MODEL_EXTRACT_PATH, filter='data')
                        print(f"--- [INFO] Modelo base descompactado! ---")
                    else:
                        print(f" ERRO: O arquivo .tar.gz do modelo base não foi encontrado: {ARCHIVE_PATH}")
                        return
                else:
                    print(f"--- [INFO] Pasta do modelo base já existe. ---")

                print(f"\n--- Carregando seu modelo final de '{MODEL_PATH}'... ---")
                if not os.path.exists(MODEL_PATH):
                    print(f" ERRO: O seu arquivo .keras treinado não foi encontrado: {MODEL_PATH}")
                    return
                    

                self.model = tf.keras.models.load_model(MODEL_PATH)
                print(" Seu modelo foi carregado!")
                
                self.model_loaded = True
                
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Modelo Keras carregado - Pressione REC"
                    
            finally:

                os.chdir(original_dir)
                    
        except Exception as e:
            print(f" Erro no carregamento do Keras: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"Erro Keras: {str(e)[:30]}"
    
    def on_enter(self):
        
        super().on_enter()
        print(" Entrando na tela de detecção de sinais TFLite")
        if self.preview is None:
            self.setup_camera()
    
    def setup_camera(self):
        
        try:
            print(" Iniciando camera4kivy...")
            

            if self.preview:
                camera_display = self.ids.get('camera_layout')
                if camera_display:
                    camera_display.remove_widget(self.preview)
            

            self.preview = SignsDetectionPreview()
            

            self.preview.set_parent_screen(self)
            

            Clock.schedule_once(self._connect_camera, 0.1)
            
        except Exception as e:
            print(f" Erro ao configurar câmera: {e}")
    
    def _connect_camera(self, dt):
        
        try:
            camera_display = self.ids.get('camera_layout')
            if camera_display and self.preview:
                camera_display.add_widget(self.preview)

                self.preview.connect_camera(
                    camera_id="0", 
                    filepath_callback=None,
                    enable_analyze_pixels=True,
                    analyze_pixels_resolution=480,
                    mirror=True
                )
                print(" Camera4kivy iniciada")
                

                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Câmera ativa - Pressione REC"
                
            else:
                print(" Erro: camera_layout não encontrado")
                
        except Exception as e:
            print(f" Erro ao conectar câmera: {e}")
    
    def hide_instructions(self):
        
        try:
            instruction_card = self.ids.get('instruction_card')
            if instruction_card:

                instruction_card.parent.remove_widget(instruction_card)
                print(" Instruções removidas")
        except Exception as e:
            print(f" Erro ao remover instruções: {e}")
    
    def process_new_frame(self, frame):
        
        if not self.model_loaded:
            return
        
        try:

            self.update_state_machine(frame)
        except Exception as e:
            print(f" Erro ao processar frame: {e}")
    
    def preprocess_frame(self, frame, image_size=(172, 172)):
        
        try:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.float32)
            frame_resized = tf.image.resize_with_pad(frame_tf, image_size[0], image_size[1])
            return frame_resized
            
        except Exception as e:
            print(f" Erro no pré-processamento: {e}")
            return None
    
    def update_state_machine(self, frame):
        
        if frame is None:
            return
            
        current_time = time.time()
        
        if self.current_state == "WAITING":

            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                if "Pressione 'REC'" not in self.ids.status_label.text:
                    self.ids.status_label.text = f"Pressione 'REC' para iniciar gravação"
        
        elif self.current_state == "RECORDING":
            elapsed = current_time - self.recording_start_time
            countdown = self.RECORDING_DURATION - elapsed
            
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"GRAVANDO... {int(countdown)+1}s"
            

            if len(self.recorded_frames) < 100:
                try:
                    processed_frame = self.preprocess_frame(frame)
                    if processed_frame is not None:
                        self.recorded_frames.append(processed_frame)
                        print(f" Frame {len(self.recorded_frames)} capturado")
                except Exception as e:
                    print(f" Erro ao processar frame: {e}")
            
            if elapsed >= self.RECORDING_DURATION:
                print(f" Gravação finalizada. {len(self.recorded_frames)} frames capturados")
                self.current_state = "PROCESSING"
        
        elif self.current_state == "PROCESSING":
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = "Processando..."
            
            if len(self.recorded_frames) >= self.FRAME_COUNT and self.model_loaded:
                try:
                    if len(self.recorded_frames) >= self.FRAME_COUNT and self.model_loaded:
                try:
                    print(f" Processando {len(self.recorded_frames)} frames com Keras...")
                    

                    indices = np.linspace(0, len(self.recorded_frames) - 1, self.FRAME_COUNT, dtype=int)
                    sequence_to_predict = [self.recorded_frames[i] for i in indices]
                    
                    input_tensor = np.expand_dims(sequence_to_predict, axis=0)
                    
                    print(f" Shape do tensor: {input_tensor.shape}")
                    

                    predictions = self.model.predict(input_tensor, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_index]
                    
                    print(f" Predição Keras: índice={predicted_index}, confiança={confidence:.3f}")
                    
                    if confidence > self.CONFIDENCE_THRESHOLD:
                        predicted_class = SIGNS_CLASSES[predicted_index]
                        self.prediction_result = f"{predicted_class} ({confidence:.2f})"
                        print(f" Resultado: {self.prediction_result}")
                    else:
                        self.prediction_result = "Não identificado"
                        print(f" Confiança baixa: {confidence:.3f}")
                        
                except Exception as pred_error:
                    print(f" Erro na predição Keras: {pred_error}")
                    import traceback
                    traceback.print_exc()
                    self.prediction_result = "Erro na predição Keras"
                        
                except Exception as pred_error:
                    print(f" Erro na predição Keras: {pred_error}")
                    import traceback
                    traceback.print_exc()
                    self.prediction_result = "Erro na predição Keras"
                finally:

                    self.recorded_frames.clear()
                    import gc
                    gc.collect()
            else:
                self.prediction_result = "Poucos frames gravados"
                print(f" Poucos frames: {len(self.recorded_frames)}/{self.FRAME_COUNT}")

            self.current_state = "COOLDOWN"
            self.cooldown_start_time = current_time
                
        elif self.current_state == "COOLDOWN":
            elapsed = current_time - self.cooldown_start_time
            
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"Resultado: {self.prediction_result}"
            if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                self.ids.result_label.text = self.prediction_result
            
            if elapsed >= self.COOLDOWN_DURATION:
                self.current_state = "WAITING"
                if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                    self.ids.result_label.text = ""
                print(" Pronto para nova gravação")
    
    def start_recording(self):
        
        if not self.model_loaded:
            print(" Modelo não carregado, não é possível gravar")
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = "Erro: Modelo não carregado"
            return
            
        if self.current_state == "WAITING":
            print(" Iniciando gravação...")
            self.current_state = "RECORDING"
            self.recorded_frames = []
            self.recording_start_time = time.time()
        else:
            print(f" Não é possível gravar no estado atual: {self.current_state}")
    
    def start_manual_recording(self):
        
        self.start_recording()
    
    def on_leave(self):
        
        super().on_leave()
        print(" Saindo da detecção de sinais...")
        

        if self.preview:
            try:
                camera_display = self.ids.get('camera_layout')
                if camera_display:
                    camera_display.remove_widget(self.preview)
                self.preview = None
                print(" Câmera desconectada")
            except:
                pass
    
    def go_back(self):
        
        print(" Voltando para home...")
        self.manager.current = 'home'