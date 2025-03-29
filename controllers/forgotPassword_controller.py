from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/forgot_password_page/forgot_password.kv")

class ForgotScreen(Screen):
    pass

class ForgotPassword(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return ForgotScreen()

if __name__ == '__main__':
 ForgotPassword().run()