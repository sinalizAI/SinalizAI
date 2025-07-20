from helpers.base_screen import BaseScreen
from helpers.message_helper import show_exit_dialog

class ProfileScreen(BaseScreen):
    def confirmar_saida(self):
        show_exit_dialog(self)
