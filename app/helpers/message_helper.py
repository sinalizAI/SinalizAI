from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, Line
from kivy.animation import Animation

from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.utils import get_color_from_hex

_snackbar = None
_exit_dialog = None

def show_message(text, duration=2):
    Clock.schedule_once(lambda dt: _show_snackbar(text, duration), 0.05)

def _show_snackbar(text, duration):
    global _snackbar

    if _snackbar and _snackbar.parent:
        _snackbar.dismiss()

    core_label = CoreLabel(text=text, font_size=dp(14))
    core_label.refresh()
    text_width = core_label.texture.size[0]

    min_w = dp(160)
    max_w = Window.width * 0.95
    padding = dp(32)
    final_w = min(max(text_width + padding, min_w), max_w)
    h = dp(48)

    label = MDLabel(
        text=text,
        halign="center",
        valign="middle",
        theme_text_color="Custom",
        text_color=(0, 0, 0, 1),
        font_style="Body2",
        size_hint=(None, None),
        size=(final_w, h),
        text_size=(final_w, None),
        shorten=False,
        max_lines=1,
    )

    class SnackbarBox(MDSnackbar):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with self.canvas.before:
                Color(0, 0, 0, 0.5)
                self.border = Line(
                    rounded_rectangle=(self.x, self.y, self.width, self.height, dp(16)),
                    width=2
                )
            self.bind(pos=self._upd, size=self._upd)

        def _upd(self, *l):
            self.border.rounded_rectangle = (self.x, self.y, self.width, self.height, dp(16))

        def open(self):
            self.opacity = 0
            super().open()
            Animation(opacity=1, duration=0.2).start(self)
            Clock.schedule_once(lambda dt: self.dismiss(), self.duration)

        def dismiss(self, *args):
            anim = Animation(opacity=0, duration=0.2)
            anim.bind(on_complete=lambda *x: super(SnackbarBox, self).dismiss())
            anim.start(self)

    _snackbar = SnackbarBox(
        label,
        duration=duration,
        pos_hint={"center_x": 0.5, "y": 0.05},
        size_hint=(None, None),
        size=(final_w, h),
        md_bg_color=(1, 1, 1, 0.85),
        radius=[dp(16)] * 4,
        elevation=0,
    )
    _snackbar.open()


def show_exit_dialog(screen_instance):
    global _exit_dialog

    if not _exit_dialog:
        _exit_dialog = MDDialog(
            text="Você deseja sair do aplicativo?",
            buttons=[
                MDFlatButton(
                    text="SAIR",
                    text_color=(1, 0, 0, 1),  # vermelho
                    md_bg_color=(0, 0, 0, 0),
                    on_release=lambda *args: _confirm_exit(screen_instance)
                ),
                MDFlatButton(
                    text="CANCELAR",
                    text_color=get_color_from_hex("#2196F3"),  # azul
                    on_release=lambda *args: _exit_dialog.dismiss()
                ),
            ],
            radius=[dp(16)] * 4,
        )
    _exit_dialog.open()

def _confirm_exit(screen_instance):
    global _exit_dialog
    _exit_dialog.dismiss()
    screen_instance.go_to_welcome()
