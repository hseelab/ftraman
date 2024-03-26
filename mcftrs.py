import numpy as np
import tkinter as tk
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from os.path import splitext
from scipy import fft, signal
from time import sleep, perf_counter
from themes import Tk, Frame, Label, Entry, Button, OptionMenu
from camera import DummyCam, DV420, ZL41W, SK2048, TCE1304


class Updater(Thread):
    def __init__(self, ctemp, stemp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = True
        self.running = True
        self.camera = None
        self.cameras = {}
        self.ctemp = ctemp
        self.stemp = stemp

        print('Detecting cameras ...')
        for c in [DV420, ZL41W, SK2048, TCE1304]:
            try:
                print(c.__name__, end=' camera: ')
                self.cameras[c.__name__] = c()
                print('Detected!')
            except:
                print('Not detected!')

        self.cameras['1024x26um' ] = DummyCam(1024, 26)
        self.cameras['2048x6.5um'] = DummyCam(2048, 6.5)
        self.cameras['2048x14um' ] = DummyCam(2048, 14)
        self.cameras['3648x8.0um'] = DummyCam(3648, 8)

    def close(self):
        for camera in self.cameras.values():
            camera.close_camera()

    def set_camera(self, camera, gain, exposure):
        self.camera = camera
        self.camera.set_gain(gain)
        self.camera.set_exposure(exposure)
        if self.camera.get_temperature:
            self.stemp.set(self.camera.get_temperature()[1])

    def run(self):
        while self.running:
            if not self.paused and self.camera:
                self.handler(self.camera.get_frame())
            else:
                sleep(0.1)

            if self.camera and self.camera.get_temperature:
                self.ctemp.set(self.camera.get_temperature()[0])


class Image(FigureCanvasTkAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax = self.figure.add_subplot()
        self.image = self.ax.imshow(np.zeros((2, 2)), aspect='auto', cmap=cm.gray)
        self.image.set_extent((0, 100, 0, 100))

    def set_data(self, image):
        self.image.set_data(image)
        self.image.norm.autoscale([0, np.amax(image)])
        self.draw()


class Spectra(FigureCanvasTkAgg):
    def _inv(self, λ): return np.divide(1, λ, where=λ!=0)
    def _raman(self, λ): return 1e7 * (1 / self.λ_0 - self._inv(λ))
    def _invraman(self, Δ): return self._inv(1 / self.λ_0 - Δ / 1e7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logscale = False
        self.pixel_count = 2
        self.pixel_pitch = 14336
        self.λ_min = 1
        self.λ_0 = 1

        self.ax1 = self.figure.add_subplot(313)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(311)
        self.ax4 = self.ax3.secondary_xaxis('bottom', functions=(self._invraman, self._raman))

        self.ax1.set_xlabel('Sensor Position (mm)')
        self.ax2.set_xscale('function', functions=(self._inv, self._inv))
        self.ax3.set_xlabel('Raman Shift (cm⁻¹)')

        self.ax1.set_xticks(np.arange(-14, 15, 1))
        self.ax2.set_xticks(1/np.linspace(1/400, 1/12000, 30))
        self.ax3.set_xticks(np.arange(-1200, 1201, 100))
        self.ax3.set_xlim(-1200, 1200)

        self.ax3.xaxis.tick_top()
        self.ax3.xaxis.set_label_position('top')
        self.ax4.minorticks_on()

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()

        self.line1, = self.ax1.plot((0, 1), (1, 1), color='#0F4', animated=True)
        self.line2, = self.ax2.plot((0, 1), (1, 1), color='#F80', animated=True)
        self.line3, = self.ax3.plot((0, 1), (1, 1), color='#F80', animated=True)

        self.ax1.add_line(self.line1)
        self.ax2.add_line(self.line2)
        self.ax3.add_line(self.line3)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def set_axes(self, λ_min=None, λ_0=None, camera=None):
        if λ_min: self.λ_min = λ_min
        if λ_0:   self.λ_0   = λ_0

        if camera:
            self.pixel_count = camera.pixel_count
            self.pixel_pitch = camera.pixel_pitch

        if self.logscale:
            self.ax2.set_yscale('log')
            self.ax3.set_yscale('log')
            self.ax2.set_ylim(1e-6, 1)
            self.ax3.set_ylim(1e-6, 1)
        else:
            self.ax2.set_yscale('linear')
            self.ax3.set_yscale('linear')
            self.ax2.set_ylim(-0.01, 1.01)
            self.ax3.set_ylim(-0.01, 1.01)

        self.ax1.set_ylim(-0.01, 1.01)
        self.ax1.set_xlim(-self.pixel_pitch * self.pixel_count / 2000, self.pixel_pitch * self.pixel_count / 2000)
        self.ax2.set_xlim(1e7, max(400, self.λ_min))
        self.ax4.set_xticks(np.arange(300, 1220, 2 if self.λ_0 < 600 else 5 if self.λ_0 < 900 else 10))

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def auto_scale(self, *args):
        ymax = min(1.2 * max(np.max(np.abs(self.line1.get_ydata())), 1e-5), 1)
        self.ax1.set_ylim(-0.01 * ymax, 1.01 * ymax)

        data = self.line2.get_ydata()
        ymax = max(np.max(data[len(data)//2:]), 1e-5)

        if self.logscale:
            ymin = max(np.min(data[len(data)//2:]), 1e-7)
            self.ax2.set_ylim(ymin, ymax**1.1 / ymin**0.1)
            self.ax3.set_ylim(ymin, ymax**1.1 / ymin**0.1)

        else:
            self.ax2.set_ylim(-0.012 * ymax, 1.2 * ymax)
            self.ax3.set_ylim(-0.012 * ymax, 1.2 * ymax)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)
        self.ax1.draw_artist(self.line1)
        self.ax2.draw_artist(self.line2)
        self.ax3.draw_artist(self.line3)
        self.blit(self.figure.bbox)

    def set_accum(self, accum):
        self.index = 0
        self.raw_data = np.zeros((accum, self.pixel_count))
        self.fft_data = np.zeros((accum, self.pixel_count*4))

    def fft(self, data):
        size = data.shape[1]
        data = np.pad(data, ((0, 0), (0, 7*size)))
        data = fft.fft(data)[:,1:1+4*size]
        return 7.5 * np.abs(data) / size

    def set_data(self, data):
        raw_data = np.sum(data, axis=0)
        fft_data = np.sum(self.fft(data), axis=0)

        if self.index >= len(self.raw_data):
            self.index = 0

        if len(raw_data) == len(self.raw_data[self.index]):
            self.raw_data[self.index] = raw_data
            self.fft_data[self.index] = fft_data
            self.index += 1

            y1 = np.average(self.raw_data, axis=0)
            y2 = np.average(self.fft_data, axis=0)
            x1 = (0.5 + np.arange(-len(y1)//2, len(y1)//2)) * self.pixel_pitch / 1000
            x2 = 1/np.linspace(1/(len(y2)*self.λ_min), 2/self.λ_min, 2*len(y2))[:len(y2)]
            x3 = self._raman(x2)

            self.line1.set_data(x1, y1)
            self.line2.set_data(x2, y2)
            self.line3.set_data(x3, y2)

            self.restore_region(self.background)
            self.ax1.draw_artist(self.line1)
            self.ax2.draw_artist(self.line2)
            self.ax3.draw_artist(self.line3)
            self.blit(self.figure.bbox)

    def get_data(self):
        x1 = self.line1.get_xdata()
        y1 = self.line1.get_ydata()
        x2 = self.line2.get_xdata()[::-1]
        y2 = self.line2.get_ydata()[::-1]
        return x1, y1, x2, y2


class App(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Multi-channel Fourier Transform Raman Spectrometer')
        self.protocol('WM_DELETE_WINDOW', self.quit)

        self.image = Image(Figure(), self)
        self.spectra = Spectra(Figure(), self)
        self.spectra.get_tk_widget().pack(side='top')
        self.ctemp = tk.IntVar(self, 0)
        self.stemp = tk.IntVar(self, 0)
        self.updater = Updater(self.ctemp, self.stemp)
        self.updater.start()
        self.controls = Frame(self)
        self.controls.pack(fill='x', side='bottom')
        self.camera_controls = Frame(self.controls)
        self.dummy_controls = Frame(self.controls)

        self.camera_type = tk.StringVar(self)
        self.accum = tk.IntVar(self, 1)
        self.gain = tk.DoubleVar(self, 1)
        self.exposure = tk.DoubleVar(self, 20)
        self.λ_min = tk.DoubleVar(self, 500)
        self.λ_0   = tk.DoubleVar(self, 532)
        self.dummy_signals = [tk.DoubleVar(self, x) for x in (0.01, 518, 1, 532, 0.1, 547, 0.4)]

        camera_type = OptionMenu(self.controls, self.camera_type, *self.updater.cameras.keys())
        accum = Entry(self.controls, textvariable=self.accum)
        exposure = Entry(self.controls, textvariable=self.exposure)
        gain = Entry(self.controls, textvariable=self.gain)
        λ_min = Entry(self.controls, textvariable=self.λ_min)
        λ_0 = Entry(self.controls, textvariable=self.λ_0)

        Label(self.controls, text=' Camera:').pack(side='left')
        camera_type.pack(side='left', pady=3)
        Label(self.controls, text=' Acc =').pack(side='left')
        accum.pack(side='left')
        Label(self.controls, text=' Gain =').pack(side='left')
        gain.pack(side='left')
        Label(self.controls, text=' Exp =').pack(side='left')
        exposure.pack(side='left')
        Label(self.controls, text=' λₘᵢₙ =').pack(side='left')
        λ_min.pack(side='left')
        Label(self.controls, text=' λ₀ =').pack(side='left')
        λ_0.pack(side='left')

        self.camera_type.trace('w', self.select_camera)

        self.buttons = []
        self.buttons.append(Button(self.controls, text='Toggle plot', command=self.toggle_plot))
        self.buttons.append(Button(self.controls, text='Auto scale', command=self.spectra.auto_scale))
        self.buttons.append(Button(self.controls, text='Log/Linear', command=self.toggle_logscale))
        self.buttons.append(Button(self.controls, text='Save as...', command=self.save_plot))
        self.buttons.append(Button(self.controls, text='Quit', command=self.quit))
        for button in reversed(self.buttons):
            button.pack(padx=2, pady=3, side='right')
        Label(self.controls, text='  ').pack(side='right')

        self.roitop = tk.IntVar(self, 0)
        self.roibtm = tk.IntVar(self, 2047)
        self.roibin = tk.IntVar(self, 1)
        Label(self.camera_controls, text=' Tₛₑₙ = [').pack(side='left')
        ctemp = Entry(self.camera_controls, textvariable=self.ctemp)
        ctemp.pack(side='left')
        Label(self.camera_controls, text='/').pack(side='left')
        stemp = Entry(self.camera_controls, textvariable=self.stemp)
        stemp.pack(side='left')
        Label(self.camera_controls, text='°C], ROI = [').pack(side='left')
        roitop = Entry(self.camera_controls, textvariable=self.roitop)
        roitop.pack(side='left')
        Label(self.camera_controls, text=',').pack(side='left')
        roibtm = Entry(self.camera_controls, textvariable=self.roibtm)
        roibtm.pack(side='left')
        Label(self.camera_controls, text=',').pack(side='left')
        roibin = Entry(self.camera_controls, textvariable=self.roibin)
        roibin.pack(side='left')
        Label(self.camera_controls, text='] ').pack(side='left')

        Label(self.dummy_controls, text='Dummy:').pack(side='left')
        dummy_signals = [Entry(self.dummy_controls, textvariable=x) for x in self.dummy_signals]
        for text, widget in zip(['A₁', 'λ₁', 'A₂', 'λ₂', 'A₃', 'λ₃', 'FWHM'], dummy_signals):
            Label(self.dummy_controls, text=f' {text} =').pack(side='left')
            widget.pack(side='left')

        for event in ['<Return>', '<FocusOut>']:
            accum.bind(event, self.set_accum)
            gain.bind(event, self.set_gain)
            exposure.bind(event, self.set_exposure)
            λ_min.bind(event, self.set_axes)
            λ_0.bind(event, self.set_axes)
            stemp.bind(event, self.set_temperature)
            roitop.bind(event, self.set_roi)
            roibtm.bind(event, self.set_roi)
            roibin.bind(event, self.set_roi)
            for widget in dummy_signals:
                widget.bind(event, self.set_dummy_signal)

        self.set_axes(self.λ_min.get(), self.λ_0.get())
        self.set_dummy_signal()
        self.bind('<Control-a>', self.spectra.auto_scale)
        self.bind('<Control-l>', self.toggle_logscale)
        self.bind('<Control-s>', self.save_plot)
        self.bind('<Control-q>', self.quit)

    def toggle_plot(self):
        if self.updater.handler != self.spectra.set_data:
            self.image.get_tk_widget().forget()
            self.spectra.get_tk_widget().pack(side='top')
            self.updater.handler = self.spectra.set_data
        else:
            self.spectra.get_tk_widget().forget()
            self.image.get_tk_widget().pack(side='top')
            self.updater.handler = self.image.set_data
        self.set_roi()

    def select_camera(self, *args):
        self.updater.paused = True
        camera = self.updater.cameras[self.camera_type.get()]
        self.title(self.title().split(' - ')[0] + ' - ' + str(camera))
        if camera.is_dummy:
            self.camera_controls.forget()
            self.dummy_controls.pack(side='right')
        elif camera.cam:
            self.dummy_controls.forget()
            self.camera_controls.pack(side='right')
        else:
            self.camera_controls.forget()
            self.dummy_controls.forget()

        self.image.get_tk_widget().forget()
        self.spectra.get_tk_widget().pack(side='top')
        self.updater.handler = self.spectra.set_data
        self.spectra.set_axes(self.λ_min.get(), self.λ_0.get(), camera)
        self.set_accum()
        self.updater.set_camera(camera, self.gain.get(), self.exposure.get())
        self.set_roi()
        self.updater.paused = False

    def set_accum(self, *args):
        try:
            accum = self.accum.get()
            camera = self.updater.cameras.get(self.camera_type.get())
            if camera != self.updater.camera or accum > 0 and accum != len(self.spectra.raw_data):
                if camera:
                    self.spectra.set_accum(accum)
        except tk.TclError: pass

    def set_gain(self, *args):
        try:
            gain = self.gain.get()
            if self.updater.camera and gain != self.updater.camera.gain:
                self.updater.camera.set_gain(self.gain.get())
        except tk.TclError: pass

    def set_exposure(self, *args):
        try:
            exposure = self.exposure.get()
            if self.updater.camera and exposure != self.updater.camera.exposure:
                self.updater.camera.set_exposure(self.exposure.get())
        except tk.TclError: pass

    def set_axes(self, event, *args):
        try:
            if self.λ_min.get() != self.spectra.λ_min or self.λ_0.get() != self.spectra.λ_0:
                if self.λ_min.get() > 0 and self.λ_0.get() > 0 and self.λ_min.get() < self.λ_0.get():
                    self.set_dummy_signal()
                    self.spectra.set_axes(self.λ_min.get(), self.λ_0.get())
        except tk.TclError: pass

    def set_temperature(self, *args):
        try:
            if self.updater.camera and self.updater.camera.get_temperature:
                self.updater.camera.set_temperature(self.stemp.get())
        except tk.TclError: pass

    def set_roi(self, *args):
        try:
            if self.updater.camera:
                roitop = self.roitop.get()
                roibtm = self.roibtm.get()
                roibin = self.roibin.get()
                if self.updater.camera.cam:
                    roitop, roibtm, roibin = self.updater.camera.set_roi(roitop, roibtm, roibin)
                    self.roitop.set(roitop)
                    self.roibtm.set(roibtm)
                    self.roibin.set(roibin)
                self.image.image.set_extent((0, self.updater.camera.pixel_count, roitop-0.5, roibtm+0.5))
        except tk.TclError: pass

    def set_dummy_signal(self, *args):
        try:
            λ_min = self.λ_min.get()
            s = [s.get() for s in self.dummy_signals]
            for camera in self.updater.cameras.values():
                if camera.is_dummy:
                    camera.set_dummy_signal(λ_min, (s[0], s[1]), (s[2], s[3]), (s[4], s[5]), fwhm=s[6])
        except tk.TclError: pass

    def toggle_logscale(self, *args):
        self.spectra.logscale = not self.spectra.logscale
        self.spectra.set_axes()

    def save_plot(self, *args):
        self.updater.paused = True

        if self.updater.camera:
            filetypes = [('All files: *.csv, *.png, *.svg', '*.csv *.png *.svg'),
                         ('Comma separated values files: *.csv', '*.csv'),
                         ('Portable network graphics files: *.png', '*.png'),
                         ('Scalable vector graphics files: *.svg', '*.svg')]
            filename = tk.filedialog.asksaveasfilename(filetypes=filetypes)

            if filename:
                filename = splitext(filename)
                if not filename[1] or filename[1] == '.csv':
                    with open(filename[0]+'.csv', 'w') as f:
                        f.write(f'λ (nm), intensity, x (mm), intensity\n')
                        x1, y1, x2, y2 = self.spectra.get_data()
                        for i in range(len(x1)):
                            f.write(f'{x2[i]},{y2[i]},{x1[i]},{y1[i]}\n')
                        for i in range(len(x1), len(x2)):
                            f.write(f'{x2[i]},{y2[i]}\n')

                if not filename[1] or filename[1] == '.png':
                    self.spectra.figure.savefig(filename[0]+'.png')

                if not filename[1] or filename[1] == '.svg':
                    self.spectra.figure.savefig(filename[0]+'.svg')

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
            else:
                self.updater.paused = False


if __name__ == '__main__':
    app = App()
    app.geometry("1600x993+310+0")
    app.mainloop()
