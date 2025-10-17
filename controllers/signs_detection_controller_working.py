"""
Controller para tela de detecção de sinais LIBRAS usando implementação do teste_janela.py
"""
from camera4kivy import Preview
from utils.base_screen import BaseScreen
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.spinner import MDSpinner
from kivy.animation import Animation
from kivy.clock import Clock
import numpy as np
import cv2
import os
import tensorflow as tf
from pathlib import Path
import time
from collections import deque
import traceback
import sys
import datetime

# Classes de sinais LIBRAS do modelo treinado - IGUAL ao teste_janela.py
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
        """Processa frame da câmera - SEM modificar buffer de pixels"""
        try:
            # Converte pixels RGBA para BGR OpenCV
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(image_size[1], image_size[0], 4)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Armazena frame para processamento
            self.current_frame = bgr.copy()
            
            # Processa frame para máquina de estados
            if self.parent_screen and self.parent_screen.model_loaded:
                self.parent_screen.update_state_machine(bgr)
                
        except Exception as e:
            print(f"❌ Erro na captura de frame: {e}")


class SignsDetectionScreen(BaseScreen):
    def update_feedback_display(self):
        """
        Atualiza o feedback visual na interface do usuário de forma segura e enxuta.
        """
        from kivy.clock import Clock

        def _do_update(dt):
            try:
                print("[DEBUG] update_feedback_display chamado!")
                if not hasattr(self, 'ids'):
                    print("[DEBUG] self.ids não existe!")
                    return
                print(f"[DEBUG] ids disponíveis: {list(self.ids.keys())}")
                feedback_label = self.ids.get('feedback_label')
                result_label = self.ids.get('result_label')
                side_result_label = self.ids.get('side_result_label')
                print(f"[DEBUG] feedback_label: {feedback_label}")
                print(f"[DEBUG] result_label: {result_label}")
                print(f"[DEBUG] side_result_label: {side_result_label}")
                if not feedback_label or not result_label or not side_result_label:
                    print("[DEBUG] Alguma label não foi encontrada!")
                    return
                print(f"[DEBUG] current_state: {self.current_state}")
                print(f"[DEBUG] prediction_result: {self.prediction_result}")
                if self.current_state == "WAITING":
                    feedback_label.text = "Pressione REC para gravar"
                    result_label.opacity = 0
                    # limpa texto do result_label para evitar exibição de valor antigo
                    try:
                        result_label.text = ""
                    except Exception:
                        pass
                    side_result_label.text = ""
                    side_result_label.opacity = 0
                elif self.current_state == "RECORDING":
                    elapsed = time.time() - getattr(self, 'recording_start_time', 0)
                    countdown = max(0, int(getattr(self, 'RECORDING_DURATION', 4) - elapsed) + 1)
                    feedback_label.text = f"Gravando... {countdown}s"
                    result_label.opacity = 0
                    side_result_label.text = ""
                    side_result_label.opacity = 0
                elif self.current_state == "PROCESSING":
                    feedback_label.text = "Processando..."
                    result_label.opacity = 0
                    side_result_label.text = ""
                    side_result_label.opacity = 0
                elif self.current_state == "COOLDOWN":
                    result = getattr(self, 'prediction_result', None)
                    print(f"[DEBUG] Exibindo na side_result_label: {result}")
                    side_result_label.opacity = 1  # Sempre visível durante cooldown
                    if result is not None and str(result).strip():
                        side_result_label.text = str(result)
                        print(f"[RESULTADO] {result}")
                    else:
                        side_result_label.text = "Nenhum sinal detectado"
            except Exception as e:
                print(f"[ERRO] Falha ao atualizar feedback: {e}")
        # Agendar a atualização na thread principal
        Clock.schedule_once(_do_update, 0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parâmetros do modelo - IGUAL ao teste_janela.py
        self.FRAME_COUNT = 16
        self.HEIGHT = 172
        self.WIDTH = 172
        self.CONFIDENCE_THRESHOLD = 0.50  
        # Parâmetros da janela de contexto
        self.RECORDING_DURATION = 4
        self.COOLDOWN_DURATION = 30 # 
        # Estados da máquina de estados
        self.current_state = "WAITING"
        self.recorded_frames = []
        self.recording_start_time = 0
        self.cooldown_start_time = 0
        self.prediction_result = ""
        # Sistema de predição isolada
        self.model_loaded = False
        # Preview da câmera
        self.preview = None
        self.camera_connected = False
        # Proteção contra timeout
        self.processing_timeout = 30  # segundos (aumentado para evitar timeout prematuro)
        # Carregar modelo
        Clock.schedule_once(self.load_model, 0.1)
    
    def load_model(self, dt):
        """Simula carregamento do modelo - A predição real será feita em processo isolado"""
        try:
            print("📥 Configurando predição em processo isolado...")
            
            # Verifica se os arquivos necessários existem
            MODEL_PATH = os.path.join("services", "ml", "movinet_libras_final_base.keras")
            SCRIPT_PATH = os.path.join("services", "ml", "isolated_prediction.py")
            
            if os.path.exists(MODEL_PATH) and os.path.exists(SCRIPT_PATH):
                print("✅ Arquivos necessários encontrados!")
                self.model_loaded = True
                
                # Agenda atualização do status na thread principal
                Clock.schedule_once(self.update_status_success, 0)
            else:
                print(f"❌ Arquivos não encontrados:")
                print(f"   Modelo: {os.path.exists(MODEL_PATH)}")
                print(f"   Script: {os.path.exists(SCRIPT_PATH)}")
                self.model_loaded = False
                Clock.schedule_once(lambda dt: self.update_status_error("Arquivos não encontrados"), 0)
                    
        except Exception as e:
            print(f"❌ Erro na configuração: {e}")
            self.model_loaded = False
            Clock.schedule_once(lambda dt: self.update_status_error(str(e)), 0)
    
    def update_status_success(self, dt):
        """Atualiza status de sucesso na thread principal"""
        if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
            self.ids.status_label.text = "Sistema configurado - Pressione REC"
    
    def update_status_error(self, error_msg):
        """Atualiza status de erro na thread principal"""
        def update(dt):
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                self.ids.status_label.text = f"Erro Keras: {error_msg[:30]}"
        return update
    
    def on_enter(self):
        """Chamado quando a tela é exibida"""
        super().on_enter()
        print("🎬 Entrando na tela de detecção de sinais Keras")
        if self.preview is None:
            self.setup_camera()
    
    def setup_camera(self):
        """Configura a câmera IGUAL ao detection_controller_camera4kivy.py"""
        try:
            print("📷 Iniciando camera4kivy...")
            
            # Remove preview anterior se existir
            if self.preview:
                camera_display = self.ids.get('camera_display')
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
            camera_display = self.ids.get('camera_display')
            if camera_display and self.preview:
                camera_display.add_widget(self.preview)
                # Conecta com análise de pixels habilitada E canvas instructions
                self.preview.connect_camera(
                    camera_id="0", 
                    filepath_callback=None,
                    enable_analyze_pixels=True,
                    analyze_pixels_resolution=480,
                    mirror=True
                )
                print("✅ Camera4kivy iniciada")
                # Atualiza status inicial
                self.update_feedback_display()
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
    
    def show_processing_indicator(self):
        """Mostra indicador de processamento com animação"""
        try:
            processing_card = self.ids.get('processing_card')
            loading_spinner = self.ids.get('loading_spinner')
            
            if processing_card:
                # Mostra card diretamente sem animação complexa
                processing_card.opacity = 1
                
            if loading_spinner:
                # Ativa spinner diretamente
                loading_spinner.active = True
                loading_spinner.opacity = 1
                
            print("🔄 Indicador de processamento ativado")
        except Exception as e:
            print(f"❌ Erro ao mostrar indicador: {e}")
    
    def hide_processing_indicator(self):
        """Esconde indicador de processamento com animação"""
        try:
            processing_card = self.ids.get('processing_card')
            loading_spinner = self.ids.get('loading_spinner')
            
            if processing_card:
                # Esconde card diretamente
                processing_card.opacity = 0
                
            if loading_spinner:
                # Desativa spinner diretamente
                loading_spinner.active = False
                loading_spinner.opacity = 0
                
            print("✅ Indicador de processamento desativado")
        except Exception as e:
            print(f"❌ Erro ao esconder indicador: {e}")
    
    def show_result_with_animation(self, result_text):
        """Mostra resultado com animação chamativa"""
        try:
            result_label = self.ids.get('result_label')
            if result_label:
                result_label.text = result_text
                
                # Animação de "pulse" mais simples - apenas tamanho
                anim1 = Animation(font_size="40sp", duration=0.3)
                anim2 = Animation(font_size="32sp", duration=0.3)
                anim1.bind(on_complete=lambda *args: anim2.start(result_label))
                anim1.start(result_label)
                
                # Muda cor baseada no resultado
                if "Não identificado" in result_text or "Erro" in result_text or "Timeout" in result_text:
                    result_label.text_color = [1, 0.5, 0, 1]  # Laranja para erro
                else:
                    result_label.text_color = [0, 1, 0, 1]  # Verde para sucesso
                    
                print(f"📢 Resultado exibido: {result_text}")
        except Exception as e:
            print(f"❌ Erro ao mostrar resultado: {e}")
            # Fallback simples sem animação
            try:
                result_label = self.ids.get('result_label')
                if result_label:
                    result_label.text = result_text
                    if "Não identificado" in result_text or "Erro" in result_text or "Timeout" in result_text:
                        result_label.text_color = [1, 0.5, 0, 1]
                    else:
                        result_label.text_color = [0, 1, 0, 1]
            except:
                pass
    
    def draw_feedback_on_frame(self, frame):
        """Desenha feedback visual diretamente no frame - IGUAL ao detection_controller"""
        try:
            current_time = time.time()
            height, width = frame.shape[:2]
            
            # Cores e configurações visuais
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            padding = 10
            
            # Calcula texto e cor baseado no estado atual
            if self.current_state == "WAITING":
                display_text = "Pressione REC para gravar"
                bg_color = (80, 120, 255)  # Azul
            elif self.current_state == "RECORDING":
                elapsed = current_time - self.recording_start_time
                countdown = max(0, self.RECORDING_DURATION - elapsed)
                display_text = f"GRAVANDO... {int(countdown)+1}s"
                bg_color = (0, 0, 255)  # Vermelho
            elif self.current_state == "PROCESSING":
                display_text = "Processando..."
                bg_color = (0, 165, 255)  # Laranja
            elif self.current_state == "COOLDOWN":
                display_text = f"Resultado: {self.prediction_result}"
                bg_color = (0, 255, 0)  # Verde
            else:
                display_text = "Sistema configurado"
                bg_color = (128, 128, 128)  # Cinza
            
            # Calcula tamanho do texto
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
            
            # Posição do texto (topo da tela)
            text_x = padding
            text_y = text_height + padding
            
            # Desenha fundo do texto - IGUAL ao detection_controller
            bg_x1 = text_x - padding//2
            bg_y1 = text_y - text_height - baseline - padding//2
            bg_x2 = text_x + text_width + padding//2
            bg_y2 = text_y + baseline + padding//2
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            
            # Cor do texto contrastante (branco sempre para boa legibilidade)
            text_color = (255, 255, 255)
            
            # Desenha o texto
            cv2.putText(frame, display_text, (text_x, text_y), font, font_scale, text_color, thickness)
            
            # Se estiver no cooldown, desenha resultado grande no centro
            if self.current_state == "COOLDOWN" and self.prediction_result:
                # Texto grande no centro para resultado
                large_font_scale = 2.0
                large_thickness = 3
                
                result_text = self.prediction_result
                (result_width, result_height), result_baseline = cv2.getTextSize(
                    result_text, font, large_font_scale, large_thickness)
                
                # Posição central
                center_x = (width - result_width) // 2
                center_y = height // 2
                
                # Fundo do resultado
                result_bg_x1 = center_x - padding
                result_bg_y1 = center_y - result_height - result_baseline - padding
                result_bg_x2 = center_x + result_width + padding
                result_bg_y2 = center_y + result_baseline + padding
                
                # Cor baseada no resultado
                if "Sinal não identificado" in self.prediction_result or "Erro" in self.prediction_result:
                    result_bg_color = (0, 165, 255)  # Laranja
                else:
                    result_bg_color = (0, 255, 0)  # Verde
                
                cv2.rectangle(frame, (result_bg_x1, result_bg_y1), (result_bg_x2, result_bg_y2), result_bg_color, -1)
                cv2.putText(frame, result_text, (center_x, center_y), font, large_font_scale, (255, 255, 255), large_thickness)
                           
        except Exception as e:
            print(f"❌ Erro ao desenhar feedback: {e}")

    def preprocess_frame(self, frame, image_size=(172, 172)):
        """Pré-processa um único frame da webcam - IGUAL ao teste_janela.py"""
        try:
            # EXATAMENTE IGUAL ao teste_janela.py
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.float32)
            frame_resized = tf.image.resize_with_pad(frame_tf, image_size[0], image_size[1])
            return frame_resized
            
        except Exception as e:
            print(f"❌ Erro no pré-processamento: {e}")
            return None

    def update_state_machine(self, frame=None):
        """Atualiza a máquina de estados de gravação e predição - IGUAL ao teste_janela.py"""
        if frame is None:
            return
            
        current_time = time.time()
        
        if self.current_state == "WAITING":
            # Atualiza feedback visual
            self.update_feedback_display()
        
        elif self.current_state == "RECORDING":
            elapsed = current_time - self.recording_start_time
            countdown = self.RECORDING_DURATION - elapsed
            
            # Atualiza feedback visual com countdown
            self.update_feedback_display()
            
            # Processa e armazena o frame - IGUAL ao teste_janela.py
            try:
                processed_frame = self.preprocess_frame(frame)
                if processed_frame is not None:
                    self.recorded_frames.append(processed_frame)
                    print(f"📹 Frame {len(self.recorded_frames)} capturado")
            except Exception as e:
                print(f"❌ Erro ao processar frame: {e}")
            
            if elapsed >= self.RECORDING_DURATION:
                print(f"⏱️ Gravação finalizada. {len(self.recorded_frames)} frames capturados")
                # Entrando em processamento: inicializa o timer de processamento
                self.current_state = "PROCESSING"
                # garante que o timeout de processamento comece a contar agora
                try:
                    self.processing_start_time = current_time
                except Exception:
                    self.processing_start_time = time.time()
                self.update_feedback_display()  # Atualiza para "Processando..."
        
        elif self.current_state == "PROCESSING":
            # Atualiza feedback visual
            self.update_feedback_display()
            
            # Verifica timeout
            if not hasattr(self, 'processing_start_time'):
                self.processing_start_time = current_time
            elif current_time - self.processing_start_time > self.processing_timeout:
                print("⚠️ Timeout na predição, pulando para cooldown")
                self.prediction_result = "Timeout"
                self.current_state = "COOLDOWN"
                self.cooldown_start_time = current_time
                self.update_feedback_display()  # Atualiza feedback
                return
            
            if len(self.recorded_frames) > 0 and self.model_loaded:
                try:
                    print(f"🔮 Processando {len(self.recorded_frames)} frames com Keras...")
                    
                    # EXATAMENTE IGUAL ao teste_janela.py - Amostragem dos frames gravados
                    indices = np.linspace(0, len(self.recorded_frames) - 1, self.FRAME_COUNT, dtype=int)
                    sequence_to_predict = [self.recorded_frames[i] for i in indices]
                    
                    input_tensor = np.expand_dims(sequence_to_predict, axis=0)
                    
                    print(f"📊 Shape do tensor: {input_tensor.shape}")
                    
                    # PREDIÇÃO EM PROCESSO ISOLADO - EVITA SEGMENTATION FAULT
                    try:
                        print(f"🔧 Executando predição em processo isolado...")

                        import pickle
                        import tempfile
                        import subprocess

                        # Cria arquivos temporários
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as input_tmp:
                            input_file = input_tmp.name
                            pickle.dump(input_tensor, input_tmp)

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as output_tmp:
                            output_file = output_tmp.name

                        # Executa predição em processo separado
                        script_path = os.path.join("services", "ml", "isolated_prediction.py")
                        # Use o mesmo interpretador Python que está executando a aplicação
                        import sys
                        python_cmd = sys.executable
                        print(f"🔧 Usando python para predição isolada: {python_cmd}")
                        print(f"🔧 script_path (absoluto): {os.path.abspath(script_path)} exists={os.path.exists(script_path)}")

                        try:
                            result = subprocess.run([
                                python_cmd, script_path, input_file, output_file
                            ], capture_output=True, text=True, timeout=int(self.processing_timeout))
                        except FileNotFoundError as fnf:
                            # Erro comum ao usar caminho fixo em outra máquina (Windows)
                            print(f"❌ Executável não encontrado ao tentar executar o subprocesso: {fnf}")
                            self.prediction_result = "Erro: executável Python não encontrado"
                            # limpa frames e entra em cooldown imediatamente
                            try:
                                self.recorded_frames.clear()
                            except Exception:
                                pass
                            try:
                                if hasattr(self, 'processing_start_time'):
                                    del self.processing_start_time
                            except Exception:
                                try:
                                    del self.processing_start_time
                                except Exception:
                                    pass
                            self.current_state = "COOLDOWN"
                            self.cooldown_start_time = current_time
                            self.update_feedback_display()
                            Clock.schedule_once(self.cooldown_tick, 0.1)
                            return

                        print(f"🔧 Processo isolado stdout: {result.stdout}")
                        if result.stderr:
                            print(f"🔧 Processo isolado stderr: {result.stderr}")

                        # Lê resultado
                        if os.path.exists(output_file):
                            with open(output_file, 'rb') as f:
                                prediction_result = pickle.load(f)

                            if prediction_result['success']:
                                predictions = prediction_result['predictions']
                                predicted_index = np.argmax(predictions[0])
                                confidence = predictions[0][predicted_index]

                                print(f"🎯 Predição isolada: índice={predicted_index}, confiança={confidence:.3f}")

                                if confidence > self.CONFIDENCE_THRESHOLD:
                                    predicted_class = SIGNS_CLASSES[predicted_index]
                                    self.prediction_result = f"{predicted_class} ({confidence:.2f})"
                                    print(f"✅ Resultado: {self.prediction_result}")
                                else:
                                    self.prediction_result = "Sinal não identificado"
                                    print(f"❌ Confiança baixa: {confidence:.3f}")
                            else:
                                print(f"❌ Erro na predição isolada: {prediction_result['error']}")
                                self.prediction_result = "Erro na predição"
                        else:
                            print("❌ Arquivo de resultado não encontrado")
                            self.prediction_result = "Erro: sem resultado"

                        # Limpa arquivos temporários
                        try:
                            os.unlink(input_file)
                            os.unlink(output_file)
                        except:
                            pass

                    except subprocess.TimeoutExpired:
                        print("❌ Timeout na predição isolada")
                        self.prediction_result = "Timeout"
                    except Exception as pred_exception:
                        print(f"❌ Erro específico na predição isolada: {pred_exception}")
                        traceback.print_exc()
                        # Salva diagnóstico detalhado para inspeção (útil em Windows)
                        try:
                            diag_lines = []
                            diag_lines.append(f"Timestamp: {datetime.datetime.utcnow().isoformat()}Z")
                            diag_lines.append(f"Exception: {repr(pred_exception)}")
                            diag_lines.append("Traceback:")
                            diag_lines.append(traceback.format_exc())
                            py_exec = sys.executable
                            diag_lines.append(f"Python executable: {py_exec}")
                            try:
                                import platform
                                diag_lines.append(f"Python version: {platform.python_version()} {platform.platform()}")
                            except Exception:
                                pass
                            script_abspath = os.path.abspath(script_path) if 'script_path' in locals() else 'N/A'
                            diag_lines.append(f"script_path: {script_abspath}")
                            diag_lines.append(f"script_exists: {os.path.exists(script_abspath) if script_abspath!='N/A' else 'N/A'}")
                            if 'input_file' in locals():
                                try:
                                    diag_lines.append(f"input_file: {input_file} size={os.path.getsize(input_file)}")
                                except Exception:
                                    diag_lines.append(f"input_file: {input_file} (size unknown)")
                            if 'output_file' in locals():
                                diag_lines.append(f"output_file: {output_file}")
                            if 'result' in locals():
                                try:
                                    diag_lines.append("--- subprocess stdout ---")
                                    diag_lines.append(getattr(result, 'stdout', '<no stdout>'))
                                    diag_lines.append("--- subprocess stderr ---")
                                    diag_lines.append(getattr(result, 'stderr', '<no stderr>'))
                                except Exception:
                                    pass
                            logname = f"prediction_diag_{int(time.time())}.log"
                            with open(logname, 'w', encoding='utf-8') as lf:
                                lf.write('\n'.join(str(x) for x in diag_lines))
                            print(f"📝 Diagnostic saved to {os.path.abspath(logname)}")
                        except Exception as diag_exc:
                            print(f"❌ Falha ao gravar diagnóstico: {diag_exc}")
                        # Em caso de erro grave, entrar em cooldown para evitar loop de erros
                        self.prediction_result = "Erro na predição"
                        # limpa frames e marca cooldown imediatamente
                        try:
                            self.recorded_frames.clear()
                        except Exception:
                            pass
                        try:
                            if hasattr(self, 'processing_start_time'):
                                del self.processing_start_time
                        except Exception:
                            try:
                                del self.processing_start_time
                            except Exception:
                                pass
                        # define cooldown e atualiza UI
                        self.current_state = "COOLDOWN"
                        self.cooldown_start_time = current_time
                        self.update_feedback_display()
                        Clock.schedule_once(self.cooldown_tick, 0.1)
                        # garante que a função saia logo que possível
                        return
                        
                except Exception as pred_error:
                    print(f"❌ Erro na predição Keras: {pred_error}")
                    import traceback
                    traceback.print_exc()
                    # Também gravar diagnóstico resumido
                    try:
                        short_log = f"prediction_error_short_{int(time.time())}.log"
                        with open(short_log, 'w', encoding='utf-8') as sf:
                            sf.write(f"Time: {datetime.datetime.utcnow().isoformat()}Z\n")
                            sf.write(f"Error: {repr(pred_error)}\n")
                            sf.write(traceback.format_exc())
                            sf.write(f"\nPython: {sys.executable}\n")
                        print(f"📝 Short diagnostic saved to {os.path.abspath(short_log)}")
                    except Exception:
                        pass
                    self.prediction_result = "Erro na predição Keras"
                finally:
                    # LIMPA MEMÓRIA após processamento
                    self.recorded_frames.clear()
                    import gc
                    gc.collect()
                    # remover o marker de tempo de processamento para não causar timeout em execuções seguintes
                    try:
                        if hasattr(self, 'processing_start_time'):
                            delattr(self, 'processing_start_time')
                    except Exception:
                        try:
                            del self.processing_start_time
                        except Exception:
                            pass
            else:
                self.prediction_result = "Sinal não identificado"
                print(f"❌ Poucos frames: {len(self.recorded_frames)}")

            self.current_state = "COOLDOWN"
            self.cooldown_start_time = current_time
            if self.prediction_result is None:
                self.prediction_result = ""
            self.update_feedback_display()  # Atualiza feedback com resultado
            # Inicia ciclo de cooldown separado
            Clock.schedule_once(self.cooldown_tick, 0.1)

    def cooldown_tick(self, dt):
        current_time = time.time()
        # Se não estivermos mais em COOLDOWN, cancelar qualquer agendamento pendente
        if self.current_state != "COOLDOWN":
            print(f"[COOLDOWN DEBUG] cooldown_tick chamado, mas estado atual é '{self.current_state}' — cancelando agendamento")
            # Limpa resultado antigo para evitar exibição em WAITING
            try:
                self.prediction_result = ""
                if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                    self.ids.result_label.text = ""
                if hasattr(self, 'ids') and hasattr(self.ids, 'side_result_label'):
                    self.ids.side_result_label.text = ""
            except Exception:
                pass
            return

        # Protege caso cooldown_start_time não exista
        try:
            elapsed = current_time - self.cooldown_start_time
        except Exception:
            elapsed = 0
        print(f"[COOLDOWN DEBUG] elapsed: {elapsed:.2f}s, state: {self.current_state}")
        if elapsed >= self.COOLDOWN_DURATION:
            print("[COOLDOWN DEBUG] Finalizando cooldown, voltando para WAITING")
            self.current_state = "WAITING"
            # Limpa o resultado anterior para não mostrar mensagens de erro na tela de espera
            try:
                self.prediction_result = ""
                if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                    self.ids.result_label.text = ""
            except Exception:
                pass
            self.update_feedback_display()  # Volta para estado inicial
            # garantir que qualquer processing_start_time antigo seja removido
            try:
                if hasattr(self, 'processing_start_time'):
                    delattr(self, 'processing_start_time')
            except Exception:
                try:
                    del self.processing_start_time
                except Exception:
                    pass
            print("🔄 Pronto para nova gravação")
        else:
            # Continua agendando enquanto estiver no cooldown
            self.update_feedback_display()
            Clock.schedule_once(self.cooldown_tick, 0.1)
                
        # O ciclo de cooldown agora é controlado por cooldown_tick
    
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
            # Limpa resultado anterior para evitar reuso de erro antigo
            try:
                self.prediction_result = ""
                if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                    self.ids.result_label.text = ""
                if hasattr(self, 'ids') and hasattr(self.ids, 'side_result_label'):
                    self.ids.side_result_label.text = ""
            except Exception:
                pass
            self.recording_start_time = time.time()
            # Limpa qualquer timer de processamento pendente
            try:
                if hasattr(self, 'processing_start_time'):
                    delattr(self, 'processing_start_time')
            except Exception:
                try:
                    del self.processing_start_time
                except Exception:
                    pass
            self.update_feedback_display()  # Atualiza para "GRAVANDO..."
        else:
            print(f"⚠️ Não é possível gravar no estado atual: {self.current_state}")
    
    def start_manual_recording(self):
        """Inicia gravação manual através do botão REC"""
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
            self.prediction_result = ""
            self.prediction_confidence = 0.0
            self.update_feedback_display()
            # self._schedule_feedback_update()  # Removido: método não existe
        else:
            print(f"⚠️ Não é possível gravar no estado atual: {self.current_state}")

    def process_prediction_result(self, palavra, conf):
        # Exibe o resultado na tela
        self.prediction_result = f"{palavra} ({conf*100:.1f}%)"
        self.prediction_confidence = conf
        self.current_state = "COOLDOWN"
        self.cooldown_start_time = time.time()
        self.update_feedback_display()
    # self._cancel_feedback_update()  # Removido: método não existe

    def show_result_with_animation(self, result_text):
        # Exibe o resultado com destaque
        if hasattr(self, 'ids'):
            self.ids.result_label.text = result_text
            self.ids.result_label.opacity = 1
    
    def on_leave(self):
        """Chamado quando sai da tela"""
        super().on_leave()
        print("📹 Saindo da detecção de sinais...")
    # self._cancel_feedback_update()  # Removido: método não existe
        # Limpa qualquer resultado pendente ao sair da tela
        try:
            self.prediction_result = ""
            if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                self.ids.result_label.text = ""
            if hasattr(self, 'ids') and hasattr(self.ids, 'side_result_label'):
                self.ids.side_result_label.text = ""
        except Exception:
            pass
        # Limpa estado primeiro
        self.current_state = "WAITING"
        self.recorded_frames.clear()
        # Remove a câmera de forma segura
        if self.preview:
            try:
                camera_display = self.ids.get('camera_layout')
                if camera_display and self.preview in camera_display.children:
                    camera_display.remove_widget(self.preview)
                # Desconecta a câmera se estiver conectada
                if hasattr(self.preview, 'disconnect_camera'):
                    self.preview.disconnect_camera()
                self.preview = None
                print("📹 Câmera desconectada")
            except Exception as e:
                print(f"❌ Erro ao desconectar câmera: {e}")
        # Força coleta de lixo
        import gc
        gc.collect()
    
    def go_back(self):
        """Volta para a tela anterior"""
        print("🏠 Voltando para home...")
        
        # Limpa recursos antes de sair
        self.current_state = "WAITING"
        # Limpa qualquer resultado pendente
        try:
            self.prediction_result = ""
            if hasattr(self, 'ids') and hasattr(self.ids, 'result_label'):
                self.ids.result_label.text = ""
            if hasattr(self, 'ids') and hasattr(self.ids, 'side_result_label'):
                self.ids.side_result_label.text = ""
        except Exception:
            pass
        if hasattr(self, 'recorded_frames'):
            self.recorded_frames.clear()
        
        try:
            self.manager.current = 'home'
        except Exception as e:
            print(f"❌ Erro ao voltar: {e}")
            # Força navegação
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'home'), 0.1)