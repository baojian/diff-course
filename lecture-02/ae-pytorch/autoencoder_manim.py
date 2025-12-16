# autoencoder_manim.py
from manim import *

class AutoencoderArchitecture(Scene):
    def construct(self):
        title = Text("Standard Autoencoder", font_size=44).to_edge(UP)
        self.play(Write(title))

        # ---- helpers ----
        def make_layer(n: int, label: str):
            nodes = VGroup(*[
                Circle(radius=0.18, stroke_width=2)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.15)

            lab = Text(label, font_size=22).next_to(nodes, DOWN, buff=0.25)
            return nodes, lab

        sizes  = [6, 4, 3, 2, 3, 4, 6]
        labels = ["x", "h₁", "h₂", "z", "h₂′", "h₁′", "x̂"]

        layers_nodes = []
        layers_labs  = []

        for n, lab in zip(sizes, labels):
            nodes, t = make_layer(n, lab)
            layers_nodes.append(nodes)
            layers_labs.append(t)

        layers = VGroup(*layers_nodes).arrange(RIGHT, buff=1.2).shift(DOWN * 0.4)

        # re-attach labels after positioning
        for nodes, t in zip(layers_nodes, layers_labs):
            t.next_to(nodes, DOWN, buff=0.25)

        # arrows between layers
        arrows = VGroup()
        for left, right in zip(layers_nodes[:-1], layers_nodes[1:]):
            a = Arrow(
                left.get_right(),
                right.get_left(),
                buff=0.10,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.12,
            )
            arrows.add(a)

        # braces for encoder/decoder
        enc_group = VGroup(*layers_nodes[:4])   # x, h1, h2, z
        dec_group = VGroup(*layers_nodes[3:])   # z, h2', h1', x̂

        enc_brace = Brace(enc_group, direction=UP, buff=0.15)
        dec_brace = Brace(dec_group, direction=UP, buff=0.15)
        enc_text  = Text("Encoder", font_size=24).next_to(enc_brace, UP, buff=0.15)
        dec_text  = Text("Decoder", font_size=24).next_to(dec_brace, UP, buff=0.15)

        # ---- animate build ----
        # show layers + labels
        for nodes, t in zip(layers_nodes, layers_labs):
            self.play(LaggedStart(*[Create(c) for c in nodes], lag_ratio=0.06), run_time=0.8)
            self.play(FadeIn(t, shift=0.2 * DOWN), run_time=0.25)

        # show arrows + braces
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08), run_time=1.0)
        self.play(FadeIn(enc_brace), FadeIn(enc_text), FadeIn(dec_brace), FadeIn(dec_text), run_time=0.6)

        # ---- animate "information flow" ----
        dot = Dot(radius=0.06)
        dot.move_to(layers_nodes[0].get_center())
        self.play(FadeIn(dot), run_time=0.2)

        for a in arrows:
            self.play(MoveAlongPath(dot, a), run_time=0.55)

        # emphasize bottleneck + reconstruction
        self.play(Indicate(layers_nodes[3], scale_factor=1.05), run_time=0.6)   # z
        self.play(Indicate(layers_nodes[-1], scale_factor=1.05), run_time=0.6) # x̂
        self.play(FadeOut(dot), run_time=0.2)

        self.wait(0.8)
