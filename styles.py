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
plt.rcParams['figure.figsize'] = (16, 9.6)


class Tk(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry('+0+0')
        self.config(bg='#000')

class Frame(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(bg='#222')

class Label(tk.Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(fg='#AAA', bg='#222')

class Entry(tk.Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(width=3, bd=0, fg='#AAA', bg='#000', insertbackground='#AAA', justify='left')

class Button(tk.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(width=10, fg='#AAA', bg='#333', activeforeground='#AAA', activebackground='#444')

class OptionMenu(tk.OptionMenu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(width=9)
        self.config(highlightthickness=0, fg='#AAA', bg='#333', activeforeground='#AAA', activebackground='#444')
        self['menu'].config(borderwidth=0, fg='#AAA', bg='#222', activeforeground='#AAA', activebackground='#444')
