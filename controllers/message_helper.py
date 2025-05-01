from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, Line
from kivy.animation import Animation

_snackbar = None

def show_message(text, duration=2):
    Clock.schedule_once(lambda dt: _show_snackbar(text, duration), 0.05)

def _show_snackbar(text, duration):
    global _snackbar

    # Fecha qualquer snackbar anterior
    if _snackbar and _snackbar.parent:
        _snackbar.dismiss()

    # Medir a largura do texto
    core_label = CoreLabel(text=text, font_size=dp(14))
    core_label.refresh()
    text_width = core_label.texture.size[0]

    # Limites de largura
    min_w = dp(160)
    max_w = Window.width * 0.95
    padding = dp(32)
    final_w = min(max(text_width + padding, min_w), max_w)
    h = dp(48)

    # Conteúdo: MDLabel que ocupa toda a largura, sem wrap
    label = MDLabel(
        text=text,
        halign="center",
        valign="middle",
        theme_text_color="Custom",
        text_color=(0, 0, 0, 1),
        font_style="Body2",
        size_hint=(None, None),
        size=(final_w, h),
        text_size=(final_w, None),  # largura fixa, altura livre => nunca quebra
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
            # fade in
            self.opacity = 0
            super().open()
            Animation(opacity=1, duration=0.2).start(self)
            Clock.schedule_once(lambda dt: self.dismiss(), self.duration)

        def dismiss(self, *args):
            # fade out
            anim = Animation(opacity=0, duration=0.2)
            anim.bind(on_complete=lambda *x: super(SnackbarBox, self).dismiss())
            anim.start(self)

    # Criar e abrir
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
