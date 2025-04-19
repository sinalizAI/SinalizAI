from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Builder.load_file("edit_profile.kv")

class EditScreen(Screen):
    pass

class Edit(MDApp):
    def build(self):
        Window.size = (360, 640) 
        return EditScreen()

if __name__ == '__main__':
    Edit().run()