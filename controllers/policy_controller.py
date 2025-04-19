from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("policy.kv")

class PolicyScreen(Screen):
    pass

class Policy(MDApp):
    def build(self):
        Window.size = (360, 640)  
        return PolicyScreen()     

if __name__ == '__main__':
     Policy().run()
