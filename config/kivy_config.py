




import os
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_GL_BACKEND'] = 'gl'

from kivy.config import Config

Config.set('input', 'mtdev', '')

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

Config.set('graphics', 'multisamples', '0')


if __name__ == "__main__":
    import main