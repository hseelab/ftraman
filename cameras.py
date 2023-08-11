import numpy as np
from time import sleep
from threading import Lock
from usb.core import find


class DummyCamera(object):
    def __init__(self, pixel_count, pixel_pitch):
        self.d = pixel_pitch / 1000
        self.n = pixel_count
        
    def __str__(self):
        return f'<DummyCamera: {self.n} x {self.d} um>'

    def set_exposure_time(self, exposure_time):
        self.exposure_time = exposure_time

    def set_dummy_signal(self, *peaks, width=0.5):
        self._gamma = 2 * np.sqrt(np.log(2)) / (width * (self.d * self.n))
        self._peaks = [(a/2, 400*np.pi/(self.d*l)) for l, a in peaks]

    def get_line(self):
        sleep(0.001 * self.exposure_time)
        x = self.d * (np.arange(-self.n//2, self.n//2) + np.random.rand(1)[0])
        y = 0
        for a, k in self._peaks:
            y += a * (1 + np.cos(k*x))
        y *= np.exp(-(x * self._gamma)**2)
        return np.minimum(1, (self.exposure_time * y + np.random.rand(self.n) - 0.5) / 100)


class TCE1304U(object):
    def __init__(self):
        self.lock = Lock()
        self.dev = find(idVendor=0x04B4, idProduct=0x0328)
        if not self.dev:
            raise RuntimeError('Camera not found!')
        self.dev.set_configuration()
        self.d = 0.008
        self.n = 3648
        self.exposure_time = 0
        self.min_exposure_time = 0.1
        self.max_exposure_time = 1000
    
    def _write(self, cmd, *data):
        result = self.dev.write(0x01, bytes([cmd, len(data)] + list(data)))
        if result != len(data)+2:
            raise RuntimeError('Command write error!')
    
    def _read(self, size):
        result = self.dev.read(0x81, size+2)
        if result[0] != 1 or result[1] != size:
            raise RuntimeError('Command read error!')
        return result[2:]
        
    def __str__(self):
        with self.lock:
            self._write(0x21, 0)
            info = self._read(43).tobytes().decode()
            info = f'{info[1:11]}, Serial: {info[15:28]}, Date: {info[29:39]}'
            self._write(0x01, 2)
            return f'<Camera: {info}, Firmware: {".".join(map(str, self._read(3)))}>'
    
    def set_exposure_time(self, exposure_time):
        with self.lock:
            if exposure_time < 0.1: exposure_time = 0.1
            if exposure_time > 1000: exposure_time = 1000
            if exposure_time != self.exposure_time:
                self.exposure_time = exposure_time
                self._exposure_time = int(10 * exposure_time)
                self._write(0x31, self._exposure_time//0x100, self._exposure_time%0x100)
                self._write(0x30, 0)
        
    def get_line(self):
        with self.lock:
            self._write(0x34, 1)
            data = np.frombuffer(self.dev.read(0x82, 7680), '<u2')
            if data[3833] != self._exposure_time:
                raise RuntimeError('Data read error!')
            dark_current = np.average(data[16:29])
            return (data[32:3680] - dark_current)/ 65536


class SK2048U3HW(object):
    def __init__(self):
        raise RuntimeError('Camera not found!')
        self.d = 0.014
        self.n = 2048
