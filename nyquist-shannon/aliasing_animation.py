import numpy as np
from manim import (
    BLACK,
    DL,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Axes,
    Create,
    FadeIn,
    Line,
    Rectangle,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
    config,
)

SIGNAL_FREQ = 3
ALIAS_FREQ = 2 
ANALOG_COLOR = "#4A90E2"
ALIAS_COLOR = "#E74C3C"
SAMPLE_COLOR = "#FF8C00"
NOTE_LINE_COLOR = "#D0E7FF"
NOTE_MARGIN_COLOR = "#FF9A9A"
TEXT_FONT_SIZE = 24


class AliasingVisualisation(Scene):
    def make_notebook_background(self):
        w, h = config.frame_width, config.frame_height

        page = Rectangle(
            width=w, height=h, fill_color=WHITE, fill_opacity=1, stroke_width=0
        )

        lines = VGroup()
        for y in np.arange(-h / 2, h / 2, 0.5):
            lines.add(
                Line(
                    LEFT * w / 2 + UP * y,
                    RIGHT * w / 2 + UP * y,
                    color=NOTE_LINE_COLOR,
                    stroke_width=2,
                )
            )

        margin_x = -w / 2 + 1.5
        margin = Line(
            UP * h / 2 + RIGHT * margin_x,
            DOWN * h / 2 + RIGHT * margin_x,
            color=NOTE_MARGIN_COLOR,
            stroke_width=2,
        )

        bg = VGroup(page, lines, margin)
        bg.set_z_index(-10)
        self.add(bg)

    def construct(self):
        self.camera.background_color = WHITE
        self.make_notebook_background()

        margin_x = -config.frame_width / 2 + 1.5

        # Setup Axes
        axes = (
            Axes(
                x_range=[0, 1, 0.1],
                y_range=[-1.5, 1.5, 0.5],
                x_length=10,
                y_length=5,
                axis_config={
                    "include_tip": False,
                    "color": BLACK,
                    "stroke_width": 2,
                },
            )
            .set_color(BLACK)
            .shift(DOWN * 1.0)
        )
        self.play(Create(axes), run_time=2.0)

        # Plot Analog Signal
        analog_signal = axes.plot(
            lambda t: np.sin(2 * np.pi * SIGNAL_FREQ * t),
            x_range=[0, 1],
            color=ANALOG_COLOR,
            stroke_width=4,
        )
        signal_label = Text(
            "Original Signal: 3 Hz", font_size=TEXT_FONT_SIZE, color=ANALOG_COLOR
        )
        signal_label.move_to([margin_x + 0.2, 3.5 + 0.05, 0], aligned_edge=DL)
        self.play(Create(analog_signal), Write(signal_label), run_time=2.0)
        self.wait()

        # Sampling Animation Loop
        current_rate = 20
        sampled_graph = axes.plot_line_graph(
            x_values=np.linspace(0, 1, current_rate + 1),
            y_values=np.sin(
                2 * np.pi * SIGNAL_FREQ * np.linspace(0, 1, current_rate + 1)
            ),
            line_color=SAMPLE_COLOR,
            vertex_dot_radius=0.05,
            vertex_dot_style={"color": SAMPLE_COLOR},
            stroke_width=3,
        )
        rate_text = Text(
            f"Sampling Rate: {current_rate} Hz",
            font_size=TEXT_FONT_SIZE,
            color=SAMPLE_COLOR,
        )
        rate_text.move_to([margin_x + 0.2, 3.0 + 0.05, 0], aligned_edge=DL)
        self.play(FadeIn(sampled_graph), Write(rate_text), run_time=2.0)
        self.wait()

        rates_to_test = [15, 10, 5]

        for new_rate in rates_to_test:
            new_x = np.linspace(0, 1, new_rate + 1)
            new_y = np.sin(2 * np.pi * SIGNAL_FREQ * new_x)

            new_sampled_graph = axes.plot_line_graph(
                x_values=new_x,
                y_values=new_y,
                line_color=SAMPLE_COLOR,
                vertex_dot_radius=0.05,
                vertex_dot_style={"color": SAMPLE_COLOR},
                stroke_width=3,
            )
            new_rate_text = Text(
                f"Sampling Rate: {new_rate} Hz",
                font_size=TEXT_FONT_SIZE,
                color=SAMPLE_COLOR,
            )
            new_rate_text.move_to([margin_x + 0.2, 3.0 + 0.05, 0], aligned_edge=DL)
            self.play(
                ReplacementTransform(sampled_graph, new_sampled_graph),
                Transform(rate_text, new_rate_text),
                run_time=1.5,
            )
            sampled_graph = new_sampled_graph
            self.wait(1.0)

            # Aliasing Reveal
            if new_rate == 5:
                self.play(analog_signal.animate.set_stroke(opacity=0.3))

                alias_explanation = Text(
                    "Aliasing! The samples form a 2 Hz wave",
                    font_size=TEXT_FONT_SIZE,
                    color=ALIAS_COLOR,
                )
                alias_explanation.move_to(
                    [margin_x + 0.2, 2.5 + 0.05, 0], aligned_edge=DL
                )

                self.play(Write(alias_explanation))

                alias_sine = axes.plot(
                    lambda t: -np.sin(2 * np.pi * ALIAS_FREQ * t),
                    x_range=[0, 1],
                    color=ALIAS_COLOR,
                    stroke_width=4,
                )

                self.play(Create(alias_sine), run_time=2)
                self.play(sampled_graph.animate.scale(1.2).scale(1 / 1.2))
                self.wait(3.0)
