from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import SlideTransition
from kivymd.toast import toast

class BaseScreen(MDScreen):
    
    def go_to_welcome(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "welcome"
    
    def go_to_home(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "home"

    def go_to_profile(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "profile"

    def go_to_faq(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "faq"

    def go_to_feedback(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "feedback"

    def go_to_policy(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "policy"

    def go_to_terms(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "terms"
    
    def go_to_about(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "aboutus"

    def go_to_edit(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "edit"
        
    #def go_to_loading(self):
        self.manager.previous_screen = self.manager.current  # Salva a tela anterior
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "loading"
    
    # Função para voltar à tela anterior
    def go_to_back(self):
        if hasattr(self.manager, 'previous_screen') and self.manager.previous_screen:
            self.manager.transition = SlideTransition(direction='right', duration=0.0)
            self.manager.current = self.manager.previous_screen
