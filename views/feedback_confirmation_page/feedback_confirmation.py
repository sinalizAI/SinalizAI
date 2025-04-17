from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window


Builder.load_file("feedback_confirmation.kv")

class Feedback_ConfirmationScreen(Screen):
    pass

class   Feedback_confirmation(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return Feedback_ConfirmationScreen() 

if __name__ == '__main__':
    Feedback_confirmation().run()