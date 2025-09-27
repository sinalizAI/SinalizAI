"""
Controller que executa o detect.py diretamente em janela separada
Como se fosse executado do terminal da IDE
"""
import subprocess
import os
import sys
from pathlib import Path
from kivy.clock import Clock
from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class CameraScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detect_process = None
        self.is_detecting = False
        
    def on_enter(self):
        """Chamado quando a tela é exibida - inicia detecção automaticamente"""
        print("🔍 CameraController.on_enter() chamado")
        super().on_enter()
        # Aguardar um momento para os IDs serem carregados e então iniciar automaticamente
        Clock.schedule_once(self.init_ui, 0.1)
        Clock.schedule_once(lambda dt: self.start_camera(), 1.0)  # Iniciar após 1 segundo
        print("🔍 on_enter() concluído")
        
    def init_ui(self, dt):
        """Inicializa a UI após IDs serem carregados"""
        print("🔍 init_ui chamado")
        self.update_status("Preparando detecção...")
        self.update_instructions("Aguardando comando para iniciar...")
    
    def on_leave(self):
        """Chamado quando a tela é deixada - garantir que processo seja parado"""
        print("🔍 CameraController.on_leave() chamado")
        self.stop_camera()
        super().on_leave()
    
    def start_camera(self):
        """Inicia a detecção executando detect.py em janela separada"""
        if self.is_detecting:
            return
            
        try:
            print("🚀 Preparando para executar detect.py...")
            self.update_status("Iniciando detecção...")
            self.update_instructions("Verificando sistema...")
            
            # Caminhos
            ROOT_DIR = Path(__file__).parent.parent
            detect_script = ROOT_DIR / "services" / "ml" / "yolov5" / "detect.py"
            model_path = ROOT_DIR / "services" / "ml" / "alfabeto.pt"
            
            # Verificar se arquivos existem
            if not detect_script.exists():
                self.update_status("Erro: Script de detecção não encontrado")
                show_message(f"Erro: detect.py não encontrado em {detect_script}")
                return
                
            if not model_path.exists():
                self.update_status("Erro: Modelo não encontrado")
                show_message(f"Erro: modelo não encontrado em {model_path}")
                return
            
            self.update_status("Ativando ambiente conda...")
            self.update_instructions("Preparando ambiente Python...")
            
            # Preparar comando com ativação do conda
            conda_env = "kivymd_app"  # nome do seu ambiente conda
            cmd = [
                "conda", "run", "-n", conda_env,
                "python", str(detect_script),
                "--weights", str(model_path),
                "--source", "0",  # webcam
                "--view-img",  # mostrar janela
                "--conf-thres", "0.5",
                "--line-thickness", "2"
            ]
            
            print(f"📝 Comando: {' '.join(cmd)}")
            
            self.update_status("Iniciando câmera...")
            self.update_instructions("Abrindo janela da câmera...")
            
            # Executar em processo separado (janela independente)
            self.detect_process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR / "services" / "ml" / "yolov5"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.is_detecting = True
            
            # Atualizar interface
            self.update_status("✅ Detecção ativa!")
            self.update_instructions("• A janela da câmera foi aberta\n• Posicione sua mão na frente da câmera\n• O sistema detectará os sinais de LIBRAS\n• Pressione 'Q' na janela da câmera para sair")
            self.update_button_text("Parar Detecção")
            
            print("✅ detect.py executado! Janela OpenCV deve abrir.")
            
            # Agendar verificação do processo
            Clock.schedule_interval(self.check_process, 2.0)
            
        except Exception as e:
            print(f"❌ Erro ao executar detect.py: {e}")
            self.update_status("❌ Erro ao iniciar detecção")
            self.update_instructions(f"Erro: {str(e)}")
            show_message(f"Erro ao iniciar detecção: {e}")
            self.is_detecting = False
    
    def check_process(self, dt):
        """Verifica se o processo ainda está rodando"""
        if self.detect_process is None:
            return False
            
        # Verificar se processo terminou
        if self.detect_process.poll() is not None:
            print("🔚 Processo detect.py terminou")
            self.detect_process = None
            self.is_detecting = False
            Clock.unschedule(self.check_process)
            
            # Atualizar interface
            self.update_status("Detecção finalizada")
            self.update_instructions("A janela da câmera foi fechada")
            self.update_button_text("Iniciar Novamente")
            
            return False
            
        return True
    
    def stop_camera(self):
        """Para o processo detect.py e mata todos os processos relacionados"""
        if self.detect_process is not None:
            try:
                print("🛑 Parando processo detect.py...")
                self.update_status("Parando detecção...")
                
                # Obter PID do processo principal
                main_pid = self.detect_process.pid
                print(f"PID do processo principal: {main_pid}")
                
                # Primeiro, tentar terminar graciosamente
                self.detect_process.terminate()
                
                # Aguardar um pouco para término gracioso
                try:
                    self.detect_process.wait(timeout=2)
                    print("Processo terminado graciosamente")
                except subprocess.TimeoutExpired:
                    print("⚡ Forçando término do processo...")
                    self.detect_process.kill()
                    
                # Matar qualquer processo Python relacionado ao detect.py que ainda esteja rodando
                try:
                    subprocess.run([
                        "pkill", "-f", "detect.py"
                    ], check=False, capture_output=True)
                    print("Matando processos detect.py restantes...")
                except:
                    pass
                
                # Matar processos OpenCV/webcam que possam estar presos
                try:
                    subprocess.run([
                        "pkill", "-f", "python.*detect"
                    ], check=False, capture_output=True)
                    print("Matando processos Python relacionados...")
                except:
                    pass
                    
                self.detect_process = None
                self.is_detecting = False
                Clock.unschedule(self.check_process)
                
                # Atualizar interface
                self.update_status("Detecção parada")
                self.update_instructions("A detecção foi interrompida")
                self.update_button_text("Iniciar Novamente")
                
                print("✅ Processo parado completamente")
                
            except Exception as e:
                print(f"❌ Erro ao parar processo: {e}")
                self.is_detecting = False
        else:
            print("Nenhum processo ativo para parar")
    
    def close_camera(self):
        """Botão X pressionado - para detecção e volta"""
        self.stop_detect()
        self.go_back()
    
    def go_back(self):
        """Volta para a tela anterior, parando qualquer detecção ativa"""
        print("🔍 go_back chamado")
        
        # Parar detecção se estiver ativa
        if self.is_detecting:
            print("Parando detecção antes de voltar...")
            self.stop_camera()
        
        try:
            from kivy.app import App
            app = App.get_running_app()
            if app and hasattr(app, 'root') and app.root:
                app.root.current = 'home'
        except Exception as e:
            print(f"❌ Erro ao voltar: {e}")
    
    def toggle_camera(self):
        """Alterna entre iniciar e parar a detecção"""
        print(f"🔍 toggle_camera chamado. is_detecting: {self.is_detecting}")
        
        if self.is_detecting:
            print("Parando detecção...")
            self.stop_camera()
        else:
            print("Iniciando detecção...")
            self.start_camera()
    
    def update_status(self, text):
        """Atualiza o texto de status na interface"""
        try:
            print(f"🔍 update_status: {text}")
            print(f"🔍 self.ids existe: {hasattr(self, 'ids')}")
            if hasattr(self, 'ids') and hasattr(self.ids, 'status_label'):
                print(f"🔍 status_label encontrado, atualizando...")
                self.ids.status_label.text = text
                print(f"🔍 status_label atualizado para: {text}")
            else:
                print(f"🔍 status_label não encontrado")
        except Exception as e:
            print(f"❌ Erro em update_status: {e}")
    
    def update_instructions(self, text):
        """Atualiza as instruções na interface"""
        try:
            if hasattr(self.ids, 'instructions_label'):
                self.ids.instructions_label.text = text
        except:
            pass
            
    def update_button_text(self, text):
        """Atualiza o texto do botão"""
        try:
            if hasattr(self.ids, 'camera_button'):
                self.ids.camera_button.text = text
        except:
            pass