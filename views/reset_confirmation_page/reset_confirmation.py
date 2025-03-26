from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

# Carregar o arquivo KV
Builder.load_file("reset_confirmation.kv")

class ConfirmationScreen(Screen):
    pass

class Confirmation(MDApp):
    def build(self):
        Window.size = (360, 640)
        return ConfirmationScreen()

if __name__ == '__main__':
    Confirmation().run()
