import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from scipy import fft
from time import sleep, perf_counter
from camera import DummyCamera, TCE1304U, SK2048U3HW


plt.style.use('dark_background')
plt.rcParams['path.simplify'] = True
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['text.color'] = '#AAA'
plt.rcParams['axes.edgecolor'] = '#AAA'
plt.rcParams['axes.labelcolor'] = '#AAA'
plt.rcParams['xtick.color'] = '#AAA'
plt.rcParams['ytick.color'] = '#AAA'
plt.rcParams['grid.color'] = '#222'
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (14, 6)


class Tk(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry('1600x993+0+0')
        self.config(bg='#222')

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
        self.config(width=4, bd=0, fg='#AAA', bg='#000', insertbackground='#AAA', justify='right')

class Button(tk.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(width=10, fg='#AAA', bg='#333', activeforeground='#AAA', activebackground='#444')

class OptionMenu(tk.OptionMenu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(width=10)
        self.config(highlightthickness=0, fg='#AAA', bg='#333', activeforeground='#AAA', activebackground='#444')
        self['menu'].config(borderwidth=0, fg='#AAA', bg='#222', activeforeground='#AAA', activebackground='#444')


class Updater(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera = None
        self.cameras = {}

        self.cameras['2048x14um'] = DummyCamera(2048, 14)
        self.cameras['3648x08um'] = DummyCamera(3648, 8)
        self.cameras['10680x3um'] = DummyCamera(10680, 2.625)

        for c in self.cameras.values():
            c.set_dummy_signal((520, 0.2), (532, 0.5), (544, 0.3))

        try: self.cameras['TCE-1304-U'] = TCE1304U()
        except: pass
        try: self.cameras['SK2048U3HW'] = SK2048U3HW()
        except: pass

        self.raw_data = []
        self.fft_data = []
        self.paused = True
        self.running = True

    def start_camera(self, handler, camera, exposure_time):
        self.handler = handler
        self.camera = self.cameras.get(camera)
        self.camera.set_exposure_time(exposure_time)

    def set_accum_count(self, accum_count, spectrum):
        self._min = spectrum._min
        self._max = spectrum._max
        self.raw_data = np.zeros((accum_count, 2*self._max))
        self.fft_data = np.zeros((accum_count, 2*self._max))

    def get_line(self):
        data = self.camera.get_line()
        data_fft = fft.rfft(data)
        data_fft[:self._min] = np.zeros(self._min)
        data_ifft = np.real(fft.irfft(data_fft))
        return data_ifft

    def run(self):
        counter = 0

        while self.running:
            camera = self.camera
            if self.paused or not camera:
                sleep(0.01)
                continue

            data = self.camera.get_line()
            if counter >= len(self.raw_data):
                counter = 0

            if camera == self.camera:
                self.raw_data[counter,:len(data)] = data
                self.fft_data[counter] = 10 * np.abs(fft.fft(self.raw_data[counter]) / len(data))
                y1 = np.average(self.raw_data, axis=0)[:len(data)]
                y2 = np.average(self.fft_data, axis=0)[self._min:self._max]
                self.handler(y1, y2)
                counter += 1


class Spectrum(FigureCanvasTkAgg):
    def _inv(self, x): return np.divide(1, x, where=x!=0)
    def _raman(self, x): return 1e7 * (1 / self.center - self._inv(x))
    def _invraman(self, x): return self._inv(1 / self.center - x / 1e7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center = 532
        self._min = 1<<5
        self._max = 1<<15

        self.ax1 = self.figure.add_subplot(313)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(311)

        self.ax1.grid()
        self.ax1.set_xlabel('Sensor Position (mm)')
        self.ax1.set_xticks(np.arange(-14, 15, 1))
        self.ax1.set_xlim(-14.6, 14.6)
        self.ax1.set_ylim(-0.01, 1.01)
        self.ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.ax2.grid()
        self.ax2.set_xscale('function', functions=(self._inv, self._inv))
        self.ax2.set_xticks(1/np.linspace(1/400, 1/12000, 30))
        self.ax2.set_xlim(12000, 500)
        self.ax2.set_ylim(-0.01, 1.01)
        self.ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.ax3.grid()
        self.ax3.set_xticks(np.arange(-700, 701, 50))
        self.ax3.set_xlim(-700, 700)
        self.ax3.set_ylim(-0.01, 1.01)
        self.ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        self.ax3.set_xlabel('Raman Shift (cm⁻¹)')    

        self.ax3.xaxis.tick_top()
        self.ax3.xaxis.set_label_position('top') 
        self.ax4 = self.ax3.secondary_xaxis('bottom', functions=(self._invraman, self._raman))
        self.ax4.set_xticks(np.arange(500, 1001, 2))
        self.ax4.xaxis.set_minor_locator(AutoMinorLocator())

        self.line1, = self.ax1.plot((  -15,   15), (0, 0), linewidth=0.5, color='#0F4', animated=True)
        self.line2, = self.ax2.plot((  500,  1e7), (0, 0), linewidth=0.5, color='#F80', animated=True)
        self.line3, = self.ax3.plot((-3000, 3000), (0, 0), linewidth=0.5, color='#F80', animated=True)

        self.ax1.add_line(self.line1)
        self.ax2.add_line(self.line2)
        self.ax3.add_line(self.line3)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def set_xaxis(self, center, camera=None):
        if camera:
            x1 = np.arange(-camera.n>>1, camera.n>>1) * camera.d
            self.ax1.set_xlim(x1[0], x1[-1])
            self.line1.set_data(x1, np.zeros(len(x1)))

        self.center = center
        x2 = 1/np.linspace(2/(500*2*self._max), 2/500, self._max*2)[self._min-1:self._max-1]
        x3 = self._raman(x2)
        self.line2.set_data(x2, np.zeros(len(x2)))
        self.line3.set_data(x3, np.zeros(len(x3)))

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)
        self.reset_required = False

    def set_ydata(self, y1, y2):
        self.restore_region(self.background)

        if len(y1) == len(self.line1.get_ydata()):
            self.line1.set_ydata(y1)
            self.line2.set_ydata(y2)
            self.line3.set_ydata(y2)

            self.ax1.draw_artist(self.line1)
            self.ax2.draw_artist(self.line2)
            self.ax3.draw_artist(self.line3)
            self.blit(self.figure.bbox)

    def auto_scale(self):
        ymax1 = min(max(1.3 * np.max(np.abs(self.line1.get_ydata())), 1e-3), 1)
        ymax2 = max(1.3 * np.max(self.line2.get_ydata()), 1e-4)

        self.ax1.set_ylim(-0.01*ymax1, 1.01*ymax1)
        self.ax2.set_ylim(-0.01*ymax2, 1.01*ymax2)
        self.ax3.set_ylim(-0.01*ymax2, 1.01*ymax2)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

        self.ax1.draw_artist(self.line1)
        self.ax2.draw_artist(self.line2)
        self.ax3.draw_artist(self.line3)
        self.blit(self.figure.bbox)


class MainWindow(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Fourier Transform Raman Spectroscopy')
        self.bind('<Configure>', self.resize)
        self.protocol('WM_DELETE_WINDOW', self.quit)

        self.updater = Updater()
        self.spectrum = Spectrum(Figure(), self)
        self.controls = Frame(self)

        self.camera_type = tk.StringVar(self)
        self.accum_count = tk.IntVar(self, 1)
        self.exposure_time = tk.DoubleVar(self, 10)
        self.spectrum_center = tk.IntVar(self, self.spectrum.center)

        self.camera_type_widget = OptionMenu(self.controls, self.camera_type, *self.updater.cameras.keys())
        self.accum_count_widget = Entry(self.controls, textvariable=self.accum_count)
        self.exposure_time_widget = Entry(self.controls, textvariable=self.exposure_time)
        self.spectrum_center_widget = Entry(self.controls, textvariable=self.spectrum_center)

        self.camera_type.trace('w', self.select_camera)
        self.accum_count_widget.bind('<Return>', self.set_accum_count)
        self.exposure_time_widget.bind('<Return>', self.set_exposure_time)
        self.spectrum_center_widget.bind('<Return>', self.set_spectrum_center)
        self.updater.start()

        self.spectrum.get_tk_widget().pack(fill='both', expand=True)
        self.controls.pack(fill='x')

        Label(self.controls, text='  Camera Type:').pack(side='left')
        self.camera_type_widget.pack(side='left')
        Label(self.controls, text=',  Accumulation =').pack(side='left')
        self.accum_count_widget.pack(side='left')
        Label(self.controls, text=',  Exposure Time =').pack(side='left')
        self.exposure_time_widget.pack(side='left')
        Label(self.controls, text='ms,    Reyleigh λ =').pack(side='left')
        self.spectrum_center_widget.pack(side='left')
        Label(self.controls, text='nm').pack(side='left')

        self.button = []
        self.button.append(Button(self.controls, text='Auto scale', command=self.spectrum.auto_scale))
        self.button.append(Button(self.controls, text='Pause', command=self.pause_camera))
        self.button.append(Button(self.controls, text='Save as...', command=self.save_plot))
        self.button.append(Button(self.controls, text='Quit', command=self.quit))
        for b in reversed(self.button):
            b.pack(padx=1, pady=0, side='right')

    def set_accum_count(self, *args):
        try:
            accum_count = self.accum_count.get()
            camera = self.updater.cameras.get(self.camera_type.get())
            if accum_count > 0:
                if camera != self.updater.camera or accum_count != len(self.updater.raw_data):
                    self.updater.set_accum_count(self.accum_count.get(), self.spectrum)
        except tk.TclError: pass

    def set_exposure_time(self, *args):
        try:
            exposure_time = self.exposure_time.get()
            if exposure_time >= 0.1 and self.updater.camera and exposure_time != self.updater.camera.exposure_time:
                self.updater.camera.set_exposure_time(self.exposure_time.get())
        except tk.TclError: pass

    def set_spectrum_center(self, *args):
        self.spectrum.set_xaxis(self.spectrum_center.get())

    def select_camera(self, *args):
        self.updater.paused = True
        camera = self.camera_type.get()
        if camera in self.updater.cameras.keys():
            self.set_accum_count()
            self.updater.start_camera(self.spectrum.set_ydata, camera, self.exposure_time.get())
            self.spectrum.set_xaxis(self.spectrum_center.get(), self.updater.camera)
            self.updater.paused = False
            self.button[1]['text'] = 'Pause'

    def pause_camera(self):
        if self.updater.camera:
            if self.updater.paused:
                if self.spectrum.reset_required:
                    self.spectrum.set_xaxis(self.spectrum_center.get())
                self.updater.paused = False
                self.button[1]['text'] = 'Pause'
            else:
                self.updater.paused = True
                self.button[1]['text'] = 'Resume'

    def save_plot(self):
        self.updater.paused = True
        self.button[1]['text'] = 'Resume'
        filetypes = [('All files', '*.*'), ('CSV data files', '*.csv'), ('PNG image files', '*.png')]
        filename = tk.filedialog.asksaveasfilename(defaultextension='.png', filetypes=filetypes)
        if filename:
            if filename.endswith('.png'):
                self.spectrum.figure.savefig(filename)
            elif filename.endswith('.csv'):
                data = zip(self.spectrum.line2.get_xdata(), self.spectrum.line2.get_ydata())
                data = '\n'.join([f'{a},{b}' for a, b in reversed(list(data)) if a <= 800])
                with open(filename, 'w') as f:
                    f.write(data)
                    f.close()
    
    def resize(self, *args):
        self.spectrum.reset_required = True
        if not self.updater.paused:
            self.updater.paused = True
            self.button[1]['text'] = 'Resume'

    def quit(self):
        self.updater.running = False
        self.updater.join(0.1)
        if self.updater.is_alive():
            self.after(1000, self.quit)
        else:
            super().quit()


if __name__ == '__main__':
    MainWindow().mainloop()
