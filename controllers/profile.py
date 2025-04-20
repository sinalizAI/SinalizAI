from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("views/profile_page/profile.kv")

class ProfileScreen(Screen):
    pass

class Profile(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return ProfileScreen()     
    
if __name__ == '__main__':
     Profile().run()
