from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window


Builder.load_file("login.kv")

class LoginScreen(Screen):
    pass

class Login(MDApp):
    def build(self):
        Window.size = (360, 640) 
        return LoginScreen()

if __name__ == '__main__':
 Login().run()