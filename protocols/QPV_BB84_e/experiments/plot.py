"""
Author: Danilo de Goede, modified by Julian Verweij.
stddev layout from:
https://tex.stackexchange.com/questions/337318/decent-looking-plot-with-standard-deviation

plots.py:
This file contains functionality to generate LaTeX code for plotting graphs
"""

import uuid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class Plot():
    def __init__(self, title, xlabel, ylabel):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.plots = []

    def get_coords(self, x_list, y_list):
        return ''.join([f"({x}, {y})" for x, y in zip(x_list, y_list)])

    def add_plot(self, xdata, ydata, label, colour, stddevs=None, matplotlib=False):
        self.plots.append(f"""\\addplot[
        color={colour}
    ]
    coordinates {{{self.get_coords(xdata, ydata)}}};
    \\addlegendentry{{{label}}}""")

        if stddevs is not None:
            name = uuid.uuid4()
            for side in ['top', 'bot']:
                self.plots.append(
                    f"""
    \\addplot[
        name path={name}{side},
        opacity=0.5,
        color={colour}!30
    ]
    coordinates {{{self.get_coords(xdata, [x[0] + x[1] for x in zip(ydata, stddevs)]) if side == 'top' else
           self.get_coords(xdata, [x[0] - x[1] for x in zip(ydata, stddevs)])}}};""")

            self.plots.append(
                f"""
    \\addplot[{colour}!30, fill opacity=0.5] fill between[of={name}top and {name}bot];""")

        if matplotlib:
            plt.plot(xdata, ydata, label=label, c=colour)

            if stddevs is not None:
                plt.fill_between(xdata, ydata - stddevs, ydata + stddevs, facecolor=colour, alpha=0.5)

    def generate_latex_code(self):
        return f"""\\begin{{tikzpicture}}
    \\begin{{axis}}[
        width=0.8\\textwidth,
        height=0.6\\textwidth,
        align=center,
        xlabel={{{self.xlabel}}},
        ylabel={{{self.ylabel}}},
        ymajorgrids=true,
        grid style=dashed,
    ]
    {''.join([x for x in self.plots])}
    \\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{self.title}}}"""

    def plot_matplotlib(self):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()
