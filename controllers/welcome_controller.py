from kivymd.uix.screen import MDScreen
from utils.base_screen import BaseScreen
from kivy.uix.screenmanager import SlideTransition
from utils.message_helper import show_message

class WelcomeScreen(BaseScreen):
    def go_to_login(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "login"

    def go_to_register(self):
        if not self.ids.checkbox_termos.active:
            show_message("Você precisa aceitar os Termos de Serviço e a Política de Privacidade.")
        else:
            self.manager.transition = SlideTransition(direction='left', duration=0.0)
            self.manager.current = "register"
