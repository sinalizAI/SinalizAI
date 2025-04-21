from kivymd.uix.screen import MDScreen
from controllers.base_screen import BaseScreen
from kivy.uix.screenmanager import SlideTransition

class WelcomeScreen(BaseScreen):
    
    def go_to_login(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "login"

    def go_to_register(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "register"
    
