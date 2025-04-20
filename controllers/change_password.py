from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/change_password_page/change_password.kv")

class ChangePasswordScreen(Screen):
    pass

class ChangePassword(MDApp):
    def build(self):
        Window.size = (360, 640) 
        return ChangePasswordScreen()

if __name__ == '__main__':
    ChangePassword().run()