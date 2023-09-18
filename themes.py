import tkinter as tk
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['path.simplify'] = True
plt.rcParams['text.color'] = '#AAA'
plt.rcParams['axes.edgecolor'] = '#AAA'
plt.rcParams['axes.labelcolor'] = '#AAA'
plt.rcParams['xtick.color'] = '#AAA'
plt.rcParams['ytick.color'] = '#AAA'
plt.rcParams['grid.color'] = '#222'
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.figsize'] = (16, 9.6)


class Tk(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(background='#000')
        self.geometry('+0+0')

class Frame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(background='#222')

class Label(tk.Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(foreground='#AAA', background='#222')

class Entry(tk.Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(foreground='#AAA', background='#000', insertbackground='#AAA', justify='left')
        self.config(width=4, border=0)

class Button(tk.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(foreground='#AAA', background='#333', activeforeground='#AAA', activebackground='#444')
        self.config(width=10)

class OptionMenu(tk.OptionMenu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(foreground='#AAA', background='#333', activeforeground='#AAA', activebackground='#444')
        self['menu'].config(foreground='#AAA', background='#222', activeforeground='#AAA', activebackground='#444')
        self['menu'].config(borderwidth=0)
        self.config(highlightthickness=0)
        self.config(width=9)
