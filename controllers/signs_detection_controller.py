
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
        

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        

        self.preview = None
        

        Clock.schedule_once(self.load_model, 0.1)
    
    def load_model(self, dt):
        
        try:
            print(" Iniciando carregamento do modelo TFLite...")
            

            model_path = os.path.join("services", "ml", "modelo_video.tflite")
            
            print(f" Verificando arquivo: {model_path}")
            print(f"   Arquivo existe: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                print("� Carregando modelo TensorFlow Lite...")
                

                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                

                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f" Modelo TFLite carregado com sucesso!")
                print(f" Input shape: {self.input_details[0]['shape']}")
                print(f" Output shape: {self.output_details[0]['shape']}")
                print(f" Input dtype: {self.input_details[0]['dtype']}")
                print(f" Output dtype: {self.output_details[0]['dtype']}")
                
                self.model_loaded = True
                
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Modelo TFLite carregado - Câmera ativa"
                    
            else:
                print(f" Modelo TFLite não encontrado: {model_path}")
                self.model_loaded = False
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Erro: Modelo TFLite não encontrado"
                    
        except Exception as e:
            print(f" Erro no carregamento do TFLite: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"Erro TFLite: {str(e)[:30]}"
    
    def on_enter(self):
        
        super().on_enter()
        print(" Entrando na tela de detecção de sinais")
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
                    self.ids.status_label.text = "Câmera ativa - Aguardando sinal"
                
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
            print(f" Erro no processamento do frame: {e}")
    
    def preprocess_frame(self, frame, image_size=(172, 172)):
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.float32)
        frame_resized = tf.image.resize_with_pad(frame_tf, image_size[0], image_size[1])
        return frame_resized
    
    def process_frame(self, dt):
        
        if not self.preview or not self.model_loaded:
            return
        
        try:

            frame = self.preview.get_frame()
            if frame is None:
                return
            

            if hasattr(frame, 'get_region'):

                import io
                frame_bytes = io.BytesIO()
                frame.save(frame_bytes, fmt='png')
                frame_bytes.seek(0)
                frame_array = np.frombuffer(frame_bytes.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            elif isinstance(frame, bytes):

                frame_array = np.frombuffer(frame, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:

                self.update_state_machine(frame)
                
        except Exception as e:
            print(f"Erro no processamento do frame: {e}")

            self.fallback_frame_capture()
    
    def fallback_frame_capture(self):
        
        try:

            if not hasattr(self, 'backup_cap'):
                self.backup_cap = cv2.VideoCapture(0)
            
            ret, frame = self.backup_cap.read()
            if ret and frame is not None:
                self.update_state_machine(frame)
        except Exception as e:
            print(f"Erro no método de backup: {e}")
    
    def preprocess_frame(self, frame, image_size=(172, 172)):
        
        try:

            if frame is None or frame.size == 0:
                return None
                

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            frame_resized = cv2.resize(frame_rgb, image_size, interpolation=cv2.INTER_AREA)
            

            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            return frame_normalized
            
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
            
            if len(self.recorded_frames) >= self.FRAME_COUNT and self.model:
                try:
                    print(f" Processando {len(self.recorded_frames)} frames...")
                    

                    indices = np.linspace(0, len(self.recorded_frames) - 1, self.FRAME_COUNT, dtype=int)
                    sequence_to_predict = [self.recorded_frames[i] for i in indices]
                    

                    input_array = np.array(sequence_to_predict, dtype=np.float32)
                    input_tensor = np.expand_dims(input_array, axis=0)
                    
                    print(f" Shape do tensor: {input_tensor.shape}")
                    

                    expected_shape = (1, self.FRAME_COUNT, self.HEIGHT, self.WIDTH, 3)
                    if input_tensor.shape != expected_shape:
                        print(f" Shape incorreto: esperado {expected_shape}, obtido {input_tensor.shape}")
                        self.prediction_result = "Erro: Shape incorreto"
                    else:

                        import time as _time
                        start_pred = _time.time()
                        try:
                            print(f" Salvando dados para predição isolada...")
                            import tempfile
                            import pickle
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                                pickle.dump(input_tensor, temp_file)
                                temp_path = temp_file.name
                            print(f" Dados salvos em: {temp_path}")
                            import subprocess
                            import sys
                            prediction_script = f'''
                        
                except Exception as pred_error:
                    print(f" Erro na predição: {pred_error}")
                    self.prediction_result = "Erro na predição"
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
                            result = subprocess.run(
                                [sys.executable, "-c", prediction_script],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            print(f" Resultado do processo: {result.stdout.strip()}")
                            if result.returncode == 0 and "," in result.stdout:
                                output_line = result.stdout.strip().split('\n')[-1]
                                if "," in output_line and not output_line.startswith("ERROR:"):
                                    predicted_index, confidence = output_line.split(',')
                                    predicted_index = int(predicted_index)
                                    confidence = float(confidence)
                                    print(f" Predição: índice={predicted_index}, confiança={confidence:.3f}")
                                    if confidence > self.CONFIDENCE_THRESHOLD:
                                        predicted_class = SIGNS_CLASSES[predicted_index]
                                        self.prediction_result = f"{predicted_class} ({confidence:.2f})"
                                        print(f" Resultado: {self.prediction_result}")
                                    else:
                                        self.prediction_result = "Não identificado"
                                        print(f" Confiança baixa: {confidence:.3f}")
                                else:
                                    self.prediction_result = "Erro no processo"
                                    print(f" Saída inválida: {result.stdout}")
                            else:
                                self.prediction_result = "Erro no processo"
                                print(f" Processo falhou: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            print(f" Timeout na predição")
                            self.prediction_result = "Timeout na predição"
                        except Exception as process_error:
                            print(f" Erro no processo: {process_error}")
                            import traceback
                            traceback.print_exc()
                            self.prediction_result = "Erro no processo"
                        finally:
                            try:
                                if 'temp_path' in locals():
                                    os.unlink(temp_path)
                            except:
                                pass
                        end_pred = _time.time()
                        tempo_resposta = end_pred - start_pred
                        print(f" Tempo de resposta do modelo: {tempo_resposta:.3f} segundos")
                        from utils.benchmark_logger import log_benchmark
                        log_benchmark('modelo_video', tempo_resposta, {'resultado': self.prediction_result})

                        if hasattr(self, '_benchmark_start'):
                            tempo_total = _time.time() - self._benchmark_start
                            print(f" Tempo total (input->resultado sinais): {tempo_total:.3f} segundos")
                            log_benchmark('tradutor_sinais_total', tempo_total, {'resultado': self.prediction_result})

                            try:
                                home_screen = self.manager.get_screen('home')
                                if hasattr(home_screen, 'log_signs_result'):
                                    home_screen.log_signs_result(self.prediction_result)
                            except Exception as e:
                                print(f"Erro ao chamar log_signs_result: {e}")
                            self._benchmark_start = _time.time()
            
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
            self._benchmark_start = time.time()
        else:
            print(f" Não é possível gravar no estado atual: {self.current_state}")
    
    def start_manual_recording(self):
        
        self.start_recording()
    
    def on_leave(self):
        
        super().on_leave()
        print(" Saindo da detecção de sinais...")
        

        if self.preview:
            try:
                self.ids.camera_layout.remove_widget(self.preview)
                self.preview = None
                print(" Câmera desconectada")
            except:
                pass
    
    def go_back(self):
        
        print(" Voltando para home...")
        self.manager.current = 'home'