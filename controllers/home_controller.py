from utils.base_screen import BaseScreen
from utils.message_helper import show_message

class HomeScreen(BaseScreen):
    def translate_alphabet(self):
        """Chama a tela de reconhecimento de alfabeto LIBRAS"""
        try:
            # Usa o método correto do BaseScreen para navegar
            self.manager.current = 'detection'
        except Exception as e:
            print(f"Erro ao abrir câmera: {e}")
            show_message("Não foi possível abrir a câmera para reconhecimento.")
