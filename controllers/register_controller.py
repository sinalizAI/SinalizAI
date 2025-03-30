from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/register_page/register.kv")

class RegisterScreen(Screen):
    pass

class Registre(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return RegisterScreen()     

if __name__ == '__main__':
    Registre().run()