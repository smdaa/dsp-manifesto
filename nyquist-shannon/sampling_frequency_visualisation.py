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
    FadeOut,
    Line,
    MathTex,
    Rectangle,
    ReplacementTransform,
    FadeTransform,
    Scene,
    VGroup,
    Write,
    config,
)

ANALOG_COLOR = "#4A90E2"
SPECTRUM_COLOR = "#E74C3C"
SAMPLE_COLOR = "#FF8C00"
NYQUIST_COLOR = "#2ECC71"
NOTE_LINE_COLOR = "#D0E7FF"
NOTE_MARGIN_COLOR = "#FF9A9A"


class SamplingFrequencyVisualisation(Scene):
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

        def x_t(t):
            return np.sinc(t)

        def X_f(f):
            return np.where(np.abs(f) <= 0.5, 1.0, 0.0)

        signal_name = r"\mathrm{sinc}(t)"
        B = 0.5

        T_MIN, T_MAX = -4.0, 4.0
        F_MIN, F_MAX = -3.0, 3.0

        time_axes = Axes(
            x_range=[T_MIN, T_MAX, 1],
            y_range=[-0.4, 1.2, 0.2],
            x_length=7.5,
            y_length=2.2,
            axis_config={"include_tip": False, "color": BLACK, "stroke_width": 2},
        ).set_color(BLACK)

        time_axes.to_edge(UP, buff=0.8)

        time_axis_labels = time_axes.get_axis_labels(
            MathTex("t").scale(0.8), MathTex("x(t)").scale(0.8)
        )

        time_graph = time_axes.plot(
            x_t, x_range=[T_MIN, T_MAX], color=ANALOG_COLOR, stroke_width=4
        )

        time_title = MathTex(r"x(t) = " + signal_name)
        time_title.set_color(ANALOG_COLOR).scale(0.7)
        time_title.move_to([margin_x + 0.2, 3.5, 0], aligned_edge=DL)

        freq_axes = Axes(
            x_range=[F_MIN, F_MAX, 1],
            y_range=[0, 3.5, 1.0],
            x_length=7.5,
            y_length=2.2,
            axis_config={"include_tip": False, "color": BLACK, "stroke_width": 2},
        ).set_color(BLACK)

        freq_axes.next_to(time_axes, DOWN, buff=1.2)

        freq_axis_labels = freq_axes.get_axis_labels(
            MathTex(r"f").scale(0.8), MathTex(r"|X(f)|").scale(0.8)
        )

        original_spectrum = freq_axes.plot(
            X_f,
            x_range=[F_MIN, F_MAX],
            color=SPECTRUM_COLOR,
            stroke_width=4,
            use_smoothing=False,
            discontinuities=[-B, B],
        )

        freq_title = MathTex(r"|X(f)|")
        freq_title.set_color(SPECTRUM_COLOR).scale(0.7)
        freq_title.move_to([margin_x + 0.2, 0.9, 0], aligned_edge=DL)

        self.play(Create(time_axes), Write(time_axis_labels), run_time=1.5)
        self.play(Write(time_title), run_time=0.8)
        self.play(Create(time_graph), run_time=2.0)
        self.wait(0.5)

        self.play(Create(freq_axes), Write(freq_axis_labels), run_time=1.5)
        self.play(Write(freq_title), run_time=0.8)
        self.play(Create(original_spectrum), run_time=2.0)
        self.wait(0.8)

        def make_samples(fs):
            n_min = int(np.floor(T_MIN * fs))
            n_max = int(np.ceil(T_MAX * fs))
            t_samples = np.arange(n_min, n_max + 1) / fs
            t_samples = t_samples[(t_samples >= T_MIN) & (t_samples <= T_MAX)]
            y_samples = x_t(t_samples)
            return time_axes.plot_line_graph(
                x_values=t_samples,
                y_values=y_samples,
                line_color=SAMPLE_COLOR,
                vertex_dot_radius=0.06,
                vertex_dot_style={"color": SAMPLE_COLOR, "fill_opacity": 1},
                stroke_width=2,
            )

        def compute_sampled_spectrum(fs):
            def X_s(f):
                total = np.zeros_like(f)
                for k in range(-10, 11):
                    total += X_f(f - k * fs)
                return total

            return X_s

        def make_sampled_spectrum(fs):
            X_s = compute_sampled_spectrum(fs)
            return freq_axes.plot(
                X_s,
                x_range=[F_MIN, F_MAX],
                color=SPECTRUM_COLOR,
                stroke_width=4,
                use_smoothing=False,
            )

        def make_nyquist_markers(fs):
            fn = fs / 2
            markers = VGroup()
            for f in [-fn, fn]:
                if F_MIN <= f <= F_MAX:
                    line = Line(
                        freq_axes.c2p(f, 0),
                        freq_axes.c2p(f, 3.0),
                        color=NYQUIST_COLOR,
                        stroke_width=2,
                        stroke_opacity=0.6,
                    )
                    label = MathTex(r"f_s/2").scale(0.5).set_color(NYQUIST_COLOR)
                    label.next_to(line, UP, buff=0.1)
                    markers.add(VGroup(line, label))
            return markers

        def make_fs_text(fs):
            fs_str = str(int(round(fs))) if abs(fs - round(fs)) < 1e-6 else f"{fs:.1f}"
            tex = MathTex(r"f_s = " + fs_str + r"~\text{Hz}")
            tex.set_color(SAMPLE_COLOR).scale(0.7)
            tex.move_to([margin_x + 0.2, 0.3, 0], aligned_edge=DL)
            return tex

        def make_status_text(fs):
            fn = fs / 2
            if fn > B:
                text = MathTex(r"\text{No aliasing: } f_s/2 > B", color=NYQUIST_COLOR)
            else:
                text = MathTex(r"\text{Aliasing: } f_s/2 < B", color=SPECTRUM_COLOR)
            text.scale(0.6)
            text.move_to([margin_x + 0.2, -0.3, 0], aligned_edge=DL)
            return text

        self.play(FadeOut(time_title), FadeOut(freq_title), run_time=0.6)

        fs_sequence = [2.0, 1.5, 1.0, 0.8, 0.6]

        fs = fs_sequence[0]
        samples = make_samples(fs)
        sampled_spectrum = make_sampled_spectrum(fs)
        nyquist = make_nyquist_markers(fs)
        fs_text = make_fs_text(fs)
        status = make_status_text(fs)

        self.play(FadeIn(samples), run_time=1.0)
        self.play(
            ReplacementTransform(original_spectrum, sampled_spectrum),
            FadeIn(nyquist),
            Write(fs_text),
            FadeIn(status),
            run_time=2.0,
        )
        self.wait(1.2)

        for new_fs in fs_sequence[1:]:
            new_samples = make_samples(new_fs)
            new_sampled_spectrum = make_sampled_spectrum(new_fs)
            new_nyquist = make_nyquist_markers(new_fs)
            new_fs_text = make_fs_text(new_fs)
            new_status = make_status_text(new_fs)

            self.play(
                ReplacementTransform(samples, new_samples),
                FadeTransform(sampled_spectrum, new_sampled_spectrum),
                FadeTransform(nyquist, new_nyquist),
                FadeTransform(fs_text, new_fs_text),
                FadeTransform(status, new_status),
                run_time=2.0,
            )

            samples = new_samples
            sampled_spectrum = new_sampled_spectrum
            nyquist = new_nyquist
            fs_text = new_fs_text
            status = new_status

            self.wait(2.0)

        self.wait(2.0)
