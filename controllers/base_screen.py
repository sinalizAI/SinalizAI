from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import SlideTransition

class BaseScreen(MDScreen):
    def go_to_welcome(self):
        self.manager.transition = SlideTransition(direction='right', duration=0.0)
        self.manager.current = "welcome"
