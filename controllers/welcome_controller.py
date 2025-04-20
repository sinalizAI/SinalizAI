from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import SlideTransition

class WelcomeScreen(MDScreen):
    
    def go_to_login(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "login"

    def go_to_register(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "register"
    
    def go_to_policy(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "policy"

    def go_to_terms(self):
        self.manager.transition = SlideTransition(direction='left', duration=0.0)
        self.manager.current = "terms"
