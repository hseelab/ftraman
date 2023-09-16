import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from scipy import fft
from time import sleep, perf_counter
from styles import Tk, Frame, Label, Entry, Button, OptionMenu
from camera import DummyCam, SK2048U3, TCE1304U


class Updater(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera = None
        self.cameras = {}

        for camera in [SK2048U3, TCE1304U]:
            try:
                self.cameras[camera.__name__] = camera()
            except: pass

        self.cameras['2048x14um']  = DummyCam(2048, 14)
        self.cameras['3648x8.0um'] = DummyCam(3648, 8)
        self.cameras['4096x7.0um'] = DummyCam(4096, 7)
        self.cameras['8192x3.5um'] = DummyCam(8192, 3.5)

        self.raw_data = []
        self.fft_data = []
        self.paused = True
        self.running = True

    def close(self):
        for camera in self.cameras.values():
            camera.close_camera()

    def set_dummy_signal(self, *peaks, width=0.5):
        for camera in self.cameras.values():
            if camera.is_dummy:
                camera.set_dummy_signal(*peaks, width=width)

    def set_camera(self, camera, camera_gain, exposure_time, plotter):
        self.plotter = plotter
        self.camera = self.cameras.get(camera)
        self.camera.set_camera_gain(camera_gain)
        self.camera.set_exposure_time(exposure_time)
        self.raw_data.fill(0)

    def set_buffer(self, accum_count, min_idx, max_idx):
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.raw_data = np.zeros((accum_count, 2*self.max_idx))
        self.fft_data = np.zeros((accum_count, 2*self.max_idx))

    def filter(self, data):
        data_fft = fft.rfft(data)
        data_fft[:self.min_idx] = np.zeros(self.min_idx)
        data_ifft = np.real(fft.irfft(data_fft))
        return data_ifft

    def run(self):
        counter = 0

        while self.running:
            camera = self.camera
            if self.paused or not camera:
                sleep(0.01)
                continue

            data = self.camera.get_frame()
            if counter >= len(self.raw_data):
                counter = 0

            if camera == self.camera:
                self.raw_data[counter,:len(data)] = data
                self.fft_data[counter] = 10 * np.abs(fft.fft(self.raw_data[counter]) / len(data))
                y1 = np.average(self.raw_data, axis=0)[:len(data)]
                y2 = np.average(self.fft_data, axis=0)[self.min_idx:self.max_idx]
                self.plotter(y1, y2)
                counter += 1


class Spectrum(FigureCanvasTkAgg):
    def _inv(self, x): return np.divide(1, x, where=x!=0)
    def _raman(self, x): return 1e7 * (1 / self.center - self._inv(x))
    def _invraman(self, x): return self._inv(1 / self.center - x / 1e7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cutoff = 1
        self.center = 1
        self.min_idx = 1<<6
        self.max_idx = 1<<15

        self.ax1 = self.figure.add_subplot(313)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(311)

        self.ax1.grid()
        self.ax1.set_xlabel('Sensor Position (mm)')
        self.ax1.set_xticks(np.arange(-14, 15, 1))
        self.ax1.set_xlim(-14.6, 14.6)
        self.ax1.set_ylim(-0.01, 1.01)
        self.ax1.ticklabel_format(axis='y')

        self.ax2.grid()
        self.ax2.set_xscale('function', functions=(self._inv, self._inv))
        self.ax2.set_xticks(1/np.linspace(1/300, 1/12000, 40))
        self.ax2.set_xlim(1e7, 300)
        self.ax2.set_ylim(-0.01, 1.01)
        self.ax2.ticklabel_format(axis='y')

        self.ax3.grid()
        self.ax3.set_xticks(np.arange(-600, 601, 50))
        self.ax3.set_xlim(-600, 600)
        self.ax3.set_ylim(-0.01, 1.01)
        self.ax3.ticklabel_format(axis='y')
        self.ax3.set_xlabel('Raman Shift (cm⁻¹)')    

        self.ax3.xaxis.tick_top()
        self.ax3.xaxis.set_label_position('top') 
        self.ax4 = self.ax3.secondary_xaxis('bottom', functions=(self._invraman, self._raman))
        self.ax4.set_xticks(np.arange(300, 1001, 2))
        self.ax4.xaxis.set_minor_locator(AutoMinorLocator())

        self.line1, = self.ax1.plot(( -15,  15), (0, 0), linewidth=0.5, color='#0F4', animated=True)
        self.line2, = self.ax2.plot(( 300, 1e7), (0, 0), linewidth=0.5, color='#F80', animated=True)
        self.line3, = self.ax3.plot((-600, 600), (0, 0), linewidth=0.5, color='#F80', animated=True)
        self.ax1.add_line(self.line1)
        self.ax2.add_line(self.line2)
        self.ax3.add_line(self.line3)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def set_plot(self, cutoff, center, camera=None):
        if camera:
            x1 = np.arange(-camera.pixel_count>>1, camera.pixel_count>>1) * camera.pixel_pitch / 1000
            self.ax1.set_xlim(x1[0], x1[-1])
            self.line1.set_data(x1, np.zeros(len(x1)))

        self.cutoff = cutoff
        self.center = center
        self.ax2.set_xlim(1e7, cutoff)
        x2 = 1/np.linspace(1/(cutoff*self.max_idx), 2/cutoff, self.max_idx*2)[self.min_idx-1:self.max_idx-1]
        x3 = self._raman(x2)
        self.line2.set_data(x2, np.zeros(len(x2)))
        self.line3.set_data(x3, np.zeros(len(x3)))

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def plot(self, y1, y2):
        try:
            self.restore_region(self.background)
            self.line1.set_ydata(y1)
            self.line2.set_ydata(y2)
            self.line3.set_ydata(y2)
            self.ax1.draw_artist(self.line1)
            self.ax2.draw_artist(self.line2)
            self.ax3.draw_artist(self.line3)
            self.blit(self.figure.bbox)
        except ValueError: pass

    def auto_scale(self, *args):
        ymax1 = min(max(1.3 * np.max(np.abs(self.line1.get_ydata())), 1e-2), 1)
        ymax2 = max(1.3 * np.max(self.line2.get_ydata()), 1e-2)
        self.ax1.set_ylim(-0.01*ymax1, 1.01*ymax1)
        self.ax2.set_ylim(-0.01*ymax2, 1.01*ymax2)
        self.ax3.set_ylim(-0.01*ymax2, 1.01*ymax2)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)
        self.ax1.draw_artist(self.line1)
        self.ax2.draw_artist(self.line2)
        self.ax3.draw_artist(self.line3)
        self.blit(self.figure.bbox)


class App(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Multi-channel Fourier Transform Raman Spectrometer')
        self.protocol('WM_DELETE_WINDOW', self.quit)
        self.spectrum = Spectrum(Figure(), self)
        self.controls = Frame(self)
        self.updater = Updater()
        self.updater.start()

        self.camera_type = tk.StringVar(self)
        self.accum_count = tk.IntVar(self, 1)
        self.camera_gain = tk.DoubleVar(self, 20)
        self.exposure_time = tk.DoubleVar(self, 1)
        self.cutoff = tk.IntVar(self, 500)
        self.center = tk.IntVar(self, 532)
        self.dummy_signals = [tk.DoubleVar(self, x) for x in [0.2, 522, 0.5, 532, 0.3, 542, 0.4]]

        camera_type_widget = OptionMenu(self.controls, self.camera_type, *self.updater.cameras.keys())
        accum_count_widget = Entry(self.controls, textvariable=self.accum_count)
        camera_gain_widget = Entry(self.controls, textvariable=self.camera_gain)
        exposure_time_widget = Entry(self.controls, textvariable=self.exposure_time)
        cutoff_widget = Entry(self.controls, textvariable=self.cutoff)
        center_widget = Entry(self.controls, textvariable=self.center)
        dummy_signal_widgets = [Entry(self.controls, textvariable=x) for x in self.dummy_signals]

        self.spectrum.get_tk_widget().pack()
        self.controls.pack(fill='x', side='bottom')

        Label(self.controls, text=' Camera:').pack(side='left')
        camera_type_widget.pack(side='left', pady=3)
        Label(self.controls, text='  Accum =').pack(side='left')
        accum_count_widget.pack(side='left')
        Label(self.controls, text='  Gain =').pack(side='left')
        camera_gain_widget.pack(side='left')
        Label(self.controls, text='  Exposure =').pack(side='left')
        exposure_time_widget.pack(side='left')
        Label(self.controls, text='ms,\t Spectrum:  λₘᵢₙ =').pack(side='left')
        cutoff_widget.pack(side='left')
        Label(self.controls, text='nm  λ₀ =').pack(side='left')
        center_widget.pack(side='left')
        Label(self.controls, text='nm,\t Dummy:').pack(side='left')

        for text, widget in zip(['A₁', 'λ₁', 'A₂', 'λ₂', 'A₃', 'λ₃', 'FWHM'], dummy_signal_widgets):
            Label(self.controls, text=f' {text} =').pack(side='left')
            widget.pack(side='left')

        self.camera_type.trace('w', self.select_camera)
        for event in ['<Return>', '<FocusOut>']:
            accum_count_widget.bind(event, self.set_accum_count)
            camera_gain_widget.bind(event, self.set_camera_gain)
            exposure_time_widget.bind(event, self.set_exposure_time)
            cutoff_widget.bind(event, self.set_plot)
            center_widget.bind(event, self.set_plot)
            for widget in dummy_signal_widgets:
                widget.bind(event, self.set_dummy_signal)

        self.bind('<Control-a>', self.spectrum.auto_scale)
        self.bind('<Control-p>', self.pause_camera)
        self.bind('<Control-s>', self.save_plot)
        self.bind('<Control-q>', self.quit)
        self.buttons = []
        self.buttons.append(Button(self.controls, text='Auto scale', command=self.spectrum.auto_scale))
        self.buttons.append(Button(self.controls, text='Pause', command=self.pause_camera))
        self.buttons.append(Button(self.controls, text='Save as...', command=self.save_plot))
        self.buttons.append(Button(self.controls, text='Quit', command=self.quit))
        for button in reversed(self.buttons):
            button.pack(padx=2, pady=3, side='right')

        self.set_dummy_signal()
        self.set_plot(None)

    def select_camera(self, *args):
        self.updater.paused = True
        camera = self.camera_type.get()

        if camera in self.updater.cameras.keys():
            self.set_accum_count()
            self.updater.set_camera(camera, self.camera_gain.get(), self.exposure_time.get(), self.spectrum.plot)
            self.spectrum.set_plot(self.cutoff.get(), self.center.get(), self.updater.camera)
            self.updater.paused = False
            self.buttons[1]['text'] = 'Pause'
            self.title(self.title().split(' - ')[0] + ' - ' + str(self.updater.camera))

    def set_accum_count(self, *args):
        try:
            accum_count = self.accum_count.get()
            camera = self.updater.cameras.get(self.camera_type.get())
            if accum_count > 0:
                if camera != self.updater.camera or accum_count != len(self.updater.raw_data):
                    self.updater.set_buffer(self.accum_count.get(), self.spectrum.min_idx, self.spectrum.max_idx)
        except tk.TclError: pass

    def set_camera_gain(self, *args):
        try:
            camera_gain = self.camera_gain.get()
            if self.updater.camera and camera_gain != self.updater.camera.camera_gain:
                self.updater.camera.set_camera_gain(self.camera_gain.get())
        except tk.TclError: pass

    def set_exposure_time(self, *args):
        try:
            exposure_time = self.exposure_time.get()
            if self.updater.camera and exposure_time != self.updater.camera.exposure_time:
                self.updater.camera.set_exposure_time(self.exposure_time.get())
        except tk.TclError: pass

    def set_plot(self, event, *args):
        try:
            if self.cutoff.get() != self.spectrum.cutoff or self.center.get() != self.spectrum.center or \
                    event and event.keycode == 13:
                self.set_dummy_signal()
                self.spectrum.set_plot(self.cutoff.get(), self.center.get())
        except tk.TclError: pass

    def set_dummy_signal(self, *args):
        try:
            s = [s.get() for s in self.dummy_signals]
            self.updater.set_dummy_signal(self.cutoff.get(), (s[0], s[1]), (s[2], s[3]), (s[4], s[5]), width=s[6])
        except tk.TclError: pass

    def pause_camera(self, *args):
        if self.updater.camera:
            if self.updater.paused:
                self.updater.paused = False
                self.buttons[1]['text'] = 'Pause'
            else:
                self.updater.paused = True
                self.buttons[1]['text'] = 'Resume'

    def save_plot(self, *args):
        self.updater.paused = True
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

        if self.buttons[1]['text'] == 'Pause':
            self.updater.paused = False
    
    def quit(self, *args):
        if self.updater.paused:
            self.updater.running = False
            self.updater.join(0.1)
            if self.updater.is_alive():
                self.after(100, self.quit)
            else:
                self.updater.close()
                super().quit()
        else:
            self.updater.paused = True
            if tk.messagebox.askyesno('McFT Raman Spectrometer', 'Do you want to close the application?'):
                self.after(100, self.quit)
            elif self.buttons[1]['text'] == 'Pause':
                self.updater.paused = False


if __name__ == '__main__':
    App().mainloop()
