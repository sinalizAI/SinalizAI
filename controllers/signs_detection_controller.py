"""
Controller para tela de detecção de sinais LIBRAS usando MoViNet
"""
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

# Classes de sinais LIBRAS do modelo treinado
SIGNS_CLASSES = sorted([
    'A', 'ABACAXI', 'ABANAR', 'ABANDONAR', 'ABELHA', 'ABENCOAR',
    'ABOBORA', 'ABORTO', 'ABRACO', 'ABRIR_JANELA', 'ABRIR_PORTA',
    'ACABAR', 'ANIMAL_MIMADO', 'A_NOITE_TODA', 'A_TARDE_TODA'
])

class SignsDetectionPreview(Preview):
    """Preview personalizado para captura de sequência de frames"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_screen = None
        self.current_frame = None
        
    def set_parent_screen(self, screen):
        """Define a tela pai para comunicação"""
        self.parent_screen = screen
        
    def analyze_pixels_callback(self, pixels, image_size, image_pos, image_scale, mirror):
        """Processa frame da câmera"""
        try:
            # Converte pixels RGBA para BGR OpenCV
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(image_size[1], image_size[0], 4)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Armazena frame atual
            self.current_frame = bgr
            
            # Notifica a tela pai que tem um novo frame
            if self.parent_screen:
                self.parent_screen.process_new_frame(bgr)
                
        except Exception as e:
            print(f"❌ Erro na captura de frame: {e}")


class SignsDetectionScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parâmetros do modelo (baseado no modelo TFLite)
        self.FRAME_COUNT = 16
        self.HEIGHT = 172
        self.WIDTH = 172
        self.CONFIDENCE_THRESHOLD = 0.70
        
        # Parâmetros da janela de contexto
        self.RECORDING_DURATION = 4
        self.COOLDOWN_DURATION = 3
        
        # Estados da máquina de estados
        self.current_state = "WAITING"
        self.recorded_frames = []
        self.recording_start_time = 0
        self.cooldown_start_time = 0
        self.prediction_result = ""
        
        # Modelo TensorFlow Lite
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        
        # Preview da câmera
        self.preview = None
        
        # Carregar modelo
        Clock.schedule_once(self.load_model, 0.1)
    
    def load_model(self, dt):
        """Carrega o modelo TensorFlow Lite para detecção de sinais"""
        try:
            print("📥 Iniciando carregamento do modelo TFLite...")
            
            # Caminho do modelo TensorFlow Lite
            model_path = os.path.join("services", "ml", "modelo_video.tflite")
            
            print(f"🔍 Verificando arquivo: {model_path}")
            print(f"   Arquivo existe: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                print("� Carregando modelo TensorFlow Lite...")
                
                # Carrega o modelo TFLite
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                # Obtém detalhes de entrada e saída
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"✅ Modelo TFLite carregado com sucesso!")
                print(f"📊 Input shape: {self.input_details[0]['shape']}")
                print(f"📊 Output shape: {self.output_details[0]['shape']}")
                print(f"📊 Input dtype: {self.input_details[0]['dtype']}")
                print(f"📊 Output dtype: {self.output_details[0]['dtype']}")
                
                self.model_loaded = True
                
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Modelo TFLite carregado - Câmera ativa"
                    
            else:
                print(f"❌ Modelo TFLite não encontrado: {model_path}")
                self.model_loaded = False
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Erro: Modelo TFLite não encontrado"
                    
        except Exception as e:
            print(f"❌ Erro no carregamento do TFLite: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"Erro TFLite: {str(e)[:30]}"
    
    def on_enter(self):
        """Chamado quando a tela é exibida"""
        super().on_enter()
        print("🎬 Entrando na tela de detecção de sinais")
        if self.preview is None:
            self.setup_camera()
    
    def setup_camera(self):
        """Configura a câmera IGUAL ao detection_controller_camera4kivy.py"""
        try:
            print("📷 Iniciando camera4kivy...")
            
            # Remove preview anterior se existir
            if self.preview:
                camera_display = self.ids.get('camera_layout')
                if camera_display:
                    camera_display.remove_widget(self.preview)
            
            # Cria novo preview personalizado
            self.preview = SignsDetectionPreview()
            
            # Define a tela pai no preview
            self.preview.set_parent_screen(self)
            
            # Conecta com a câmera usando o mesmo método do detection.py
            Clock.schedule_once(self._connect_camera, 0.1)
            
        except Exception as e:
            print(f"❌ Erro ao configurar câmera: {e}")
    
    def _connect_camera(self, dt):
        """Conecta a câmera ao preview - IGUAL ao detection_controller_camera4kivy.py"""
        try:
            camera_display = self.ids.get('camera_layout')
            if camera_display and self.preview:
                camera_display.add_widget(self.preview)
                # Conecta com análise de pixels habilitada
                self.preview.connect_camera(
                    camera_id="0", 
                    filepath_callback=None,
                    enable_analyze_pixels=True,
                    analyze_pixels_resolution=480,
                    mirror=True
                )
                print("✅ Camera4kivy iniciada")
                
                # Atualiza status
                if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                    self.ids.status_label.text = "Câmera ativa - Aguardando sinal"
                
            else:
                print("❌ Erro: camera_layout não encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao conectar câmera: {e}")
    
    def hide_instructions(self):
        """Esconde o card de instruções"""
        try:
            instruction_card = self.ids.get('instruction_card')
            if instruction_card:
                # Remove o card das instruções
                instruction_card.parent.remove_widget(instruction_card)
                print("✅ Instruções removidas")
        except Exception as e:
            print(f"❌ Erro ao remover instruções: {e}")
    
    def process_new_frame(self, frame):
        """Processa novo frame da câmera (chamado pelo preview)"""
        if not self.model_loaded:
            return
        
        try:
            # Atualiza a máquina de estados
            self.update_state_machine(frame)
            
        except Exception as e:
            print(f"❌ Erro no processamento do frame: {e}")
    
    def preprocess_frame(self, frame, image_size=(172, 172)):
        """Pré-processa um único frame da webcam (baseado no teste_janela.py)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.float32)
        frame_resized = tf.image.resize_with_pad(frame_tf, image_size[0], image_size[1])
        return frame_resized
    
    def process_frame(self, dt):
        """Processa frame da câmera em tempo real"""
        if not self.preview or not self.model_loaded:
            return
        
        try:
            # Captura o frame atual da preview
            frame = self.preview.get_frame()
            if frame is None:
                return
            
            # Se o frame for uma texture, converte para numpy array
            if hasattr(frame, 'get_region'):
                # É uma texture, precisa converter
                import io
                frame_bytes = io.BytesIO()
                frame.save(frame_bytes, fmt='png')
                frame_bytes.seek(0)
                frame_array = np.frombuffer(frame_bytes.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            elif isinstance(frame, bytes):
                # Frame em bytes
                frame_array = np.frombuffer(frame, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Atualiza a máquina de estados
                self.update_state_machine(frame)
                
        except Exception as e:
            print(f"Erro no processamento do frame: {e}")
            # Tenta método alternativo de captura
            self.fallback_frame_capture()
    
    def fallback_frame_capture(self):
        """Método alternativo de captura de frames se o principal falhar"""
        try:
            # Usa OpenCV diretamente como backup
            if not hasattr(self, 'backup_cap'):
                self.backup_cap = cv2.VideoCapture(0)
            
            ret, frame = self.backup_cap.read()
            if ret and frame is not None:
                self.update_state_machine(frame)
        except Exception as e:
            print(f"Erro no método de backup: {e}")
    
    def preprocess_frame(self, frame, image_size=(172, 172)):
        """Pré-processa um único frame da webcam com melhor gerenciamento de memória"""
        try:
            # Valida se o frame é válido
            if frame is None or frame.size == 0:
                return None
                
            # Converte para RGB usando NumPy (mais estável)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensiona usando OpenCV (mais eficiente)
            frame_resized = cv2.resize(frame_rgb, image_size, interpolation=cv2.INTER_AREA)
            
            # Normaliza para [0, 1]
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            return frame_normalized
            
        except Exception as e:
            print(f"❌ Erro no pré-processamento: {e}")
            return None
    
    def update_state_machine(self, frame):
        """Atualiza a máquina de estados de gravação e predição com melhor controle"""
        if frame is None:
            return
            
        current_time = time.time()
        
        if self.current_state == "WAITING":
            # Atualiza status apenas se necessário
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                if "Pressione 'REC'" not in self.ids.status_label.text:
                    self.ids.status_label.text = f"Pressione 'REC' para iniciar gravação"
        
        elif self.current_state == "RECORDING":
            elapsed = current_time - self.recording_start_time
            countdown = self.RECORDING_DURATION - elapsed
            
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"GRAVANDO... {int(countdown)+1}s"
            
            # Processa e armazena o frame (COM LIMITE)
            if len(self.recorded_frames) < 100:  # Limite máximo de frames
                try:
                    processed_frame = self.preprocess_frame(frame)
                    if processed_frame is not None:
                        self.recorded_frames.append(processed_frame)
                        print(f"📹 Frame {len(self.recorded_frames)} capturado")
                except Exception as e:
                    print(f"❌ Erro ao processar frame: {e}")
            
            if elapsed >= self.RECORDING_DURATION:
                print(f"⏱️ Gravação finalizada. {len(self.recorded_frames)} frames capturados")
                self.current_state = "PROCESSING"
        
        elif self.current_state == "PROCESSING":
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = "Processando..."
            
            if len(self.recorded_frames) >= self.FRAME_COUNT and self.model:
                try:
                    print(f"🔮 Processando {len(self.recorded_frames)} frames...")
                    
                    # Seleciona frames uniformemente espaçados
                    indices = np.linspace(0, len(self.recorded_frames) - 1, self.FRAME_COUNT, dtype=int)
                    sequence_to_predict = [self.recorded_frames[i] for i in indices]
                    
                    # Cria tensor de entrada com validação de shape
                    input_array = np.array(sequence_to_predict, dtype=np.float32)
                    input_tensor = np.expand_dims(input_array, axis=0)
                    
                    print(f"📊 Shape do tensor: {input_tensor.shape}")
                    
                    # Valida shape antes da predição
                    expected_shape = (1, self.FRAME_COUNT, self.HEIGHT, self.WIDTH, 3)
                    if input_tensor.shape != expected_shape:
                        print(f"❌ Shape incorreto: esperado {expected_shape}, obtido {input_tensor.shape}")
                        self.prediction_result = "Erro: Shape incorreto"
                    else:
                        # SOLUÇÃO RADICAL: Executa predição em processo separado para evitar crash
                        try:
                            print(f"🔧 Salvando dados para predição isolada...")
                            
                            # Salva o tensor em arquivo temporário
                            import tempfile
                            import pickle
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                                pickle.dump(input_tensor, temp_file)
                                temp_path = temp_file.name
                            
                            print(f"📁 Dados salvos em: {temp_path}")
                            
                            # Executa predição em processo separado usando subprocess
                            import subprocess
                            import sys
                            
                            # Script de predição isolado
                            prediction_script = f'''
import os
os.chdir("{os.getcwd()}/services/ml")
import tensorflow as tf
import numpy as np
import pickle

# Configurações seguras
tf.config.set_soft_device_placement(True)

try:
    # Carrega dados
    with open("{temp_path}", "rb") as f:
        input_tensor = pickle.load(f)
    
    # Carrega modelo
    model = tf.keras.models.load_model("movinet_libras_final_base.keras", compile=False)
    
    # Predição
    predictions = model.predict(input_tensor, verbose=0, batch_size=1)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    
    print(f"{{predicted_index}},{{confidence}}")
    
except Exception as e:
    print(f"ERROR:{{e}}")
finally:
    # Remove arquivo temporário
    try:
        os.unlink("{temp_path}")
    except:
        pass
'''
                            
                            # Executa em processo separado
                            result = subprocess.run(
                                [sys.executable, "-c", prediction_script],
                                capture_output=True,
                                text=True,
                                timeout=30  # Timeout de 30 segundos
                            )
                            
                            print(f"🔍 Resultado do processo: {result.stdout.strip()}")
                            
                            if result.returncode == 0 and "," in result.stdout:
                                # Parse do resultado
                                output_line = result.stdout.strip().split('\n')[-1]
                                if "," in output_line and not output_line.startswith("ERROR:"):
                                    predicted_index, confidence = output_line.split(',')
                                    predicted_index = int(predicted_index)
                                    confidence = float(confidence)
                                    
                                    print(f"🎯 Predição: índice={predicted_index}, confiança={confidence:.3f}")
                                    
                                    if confidence > self.CONFIDENCE_THRESHOLD:
                                        predicted_class = SIGNS_CLASSES[predicted_index]
                                        self.prediction_result = f"{predicted_class} ({confidence:.2f})"
                                        print(f"✅ Resultado: {self.prediction_result}")
                                    else:
                                        self.prediction_result = "Não identificado"
                                        print(f"❌ Confiança baixa: {confidence:.3f}")
                                else:
                                    self.prediction_result = "Erro no processo"
                                    print(f"❌ Saída inválida: {result.stdout}")
                            else:
                                self.prediction_result = "Erro no processo"
                                print(f"❌ Processo falhou: {result.stderr}")
                                
                        except subprocess.TimeoutExpired:
                            print(f"❌ Timeout na predição")
                            self.prediction_result = "Timeout na predição"
                        except Exception as process_error:
                            print(f"❌ Erro no processo: {process_error}")
                            import traceback
                            traceback.print_exc()
                            self.prediction_result = "Erro no processo"
                        finally:
                            # Limpa arquivo temporário se ainda existir
                            try:
                                if 'temp_path' in locals():
                                    os.unlink(temp_path)
                            except:
                                pass
                        
                except Exception as pred_error:
                    print(f"❌ Erro na predição: {pred_error}")
                    self.prediction_result = "Erro na predição"
                finally:
                    # LIMPA MEMÓRIA após processamento
                    self.recorded_frames.clear()
                    import gc
                    gc.collect()
            else:
                self.prediction_result = "Poucos frames gravados"
                print(f"❌ Poucos frames: {len(self.recorded_frames)}/{self.FRAME_COUNT}")

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
                print("🔄 Pronto para nova gravação")
    
    def start_recording(self):
        """Inicia a gravação"""
        if not self.model_loaded:
            print("⚠️ Modelo não carregado, não é possível gravar")
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = "Erro: Modelo não carregado"
            return
            
        if self.current_state == "WAITING":
            print("🎬 Iniciando gravação...")
            self.current_state = "RECORDING"
            self.recorded_frames = []
            self.recording_start_time = time.time()
        else:
            print(f"⚠️ Não é possível gravar no estado atual: {self.current_state}")
    
    def start_manual_recording(self):
        """Inicia gravação manual através do botão REC"""
        self.start_recording()
    
    def on_leave(self):
        """Chamado quando sai da tela"""
        super().on_leave()
        print("📹 Saindo da detecção de sinais...")
        
        # Remove a câmera
        if self.preview:
            try:
                self.ids.camera_layout.remove_widget(self.preview)
                self.preview = None
                print("📹 Câmera desconectada")
            except:
                pass
    
    def go_back(self):
        """Volta para a tela anterior"""
        print("🏠 Voltando para home...")
        self.manager.current = 'home'