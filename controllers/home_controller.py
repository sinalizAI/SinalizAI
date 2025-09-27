from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class HomeScreen(BaseScreen):
    def translate_alphabet(self):
        """Chama a tela de reconhecimento de alfabeto LIBRAS"""
        try:
            from kivy.app import App
            app = App.get_running_app()
            app.root.current = 'camera'
        except Exception as e:
            print(f"Erro ao abrir câmera: {e}")
            show_message("Não foi possível abrir a câmera para reconhecimento.")
