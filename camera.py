import numpy as np
import ctypes as c
from time import sleep
from threading import Lock
from usb.core import find
from ctypes import cdll, create_string_buffer, create_unicode_buffer, POINTER
from ctypes import c_wchar, c_wchar_p, c_bool, c_int, c_uint8, c_size_t, c_double

SKDLLPATH = 'C:/Program Files/Common Files/SK/SK91USB3-WIN/SK91USB3_x64u.dll'


class Camera(object):
    is_dummy = False
    camera_gain = 0
    exposure_time = 10

    def set_camera_gain(self, camera_gain):
        self.camera_gain = camera_gain

    def set_exposure_time(self, exposure_time):
        self.exposure_time = exposure_time

    def close_camera(self):
        pass


class DummyCam(Camera):
    '''
    Dummy Line Scan Camera
    '''
    def __init__(self, pixel_count, pixel_pitch):
        self.is_dummy = True
        self.pixel_count = pixel_count
        self.pixel_pitch = pixel_pitch
        
    def __str__(self):
        return f'<Camera: Dummy, Pixel: {self.pixel_count}, Pitch: {self.pixel_pitch}um>'

    def get_frame(self):
        sleep(0.001 * self.exposure_time)
        x = self.pixel_pitch * (np.arange(-self.pixel_count//2, self.pixel_count//2) + np.random.rand(1)[0])
        y = 0
        for a, k in self._peaks:
            y += a * (1 + np.cos(k*x))
        y *= np.exp(-(x * self._gamma)**2)
        return np.minimum(1, (self.camera_gain * y + np.random.rand(self.pixel_count) - 0.5) / 100)

    def set_dummy_signal(self, *peaks, width=0.5):
        self._gamma = 2 * np.sqrt(np.log(2)) / (width * (self.pixel_pitch * self.pixel_count))
        self._peaks = [(a/2, 500*np.pi/(self.pixel_pitch*l)) for l, a in peaks]


class SK2048U3(Camera):
    '''
    Schafter Kirchhoff USB 3.0 CMOS Line Scan Camera
    The camera driver software can be downloaded from
    https://www.sukhamburg.com/products/details/SKLineScan
    '''
    def __init__(self):
        self.lock = Lock()
        self._camera_id = 0
        self._channel = 0
        self._dll = cdll.LoadLibrary(SKDLLPATH)
        self._dll.SK_LOADDLL()
        if self._dll.SK_INITCAMERA(self._camera_id):
            raise RuntimeError('Camera not found!')

        self._dll.SK_GETPIXWIDTH.restype = c_double
        self._dll.SK_GETEXPOSURETIME.restype = c_double
        self._dll.SK_GETLINEFREQUENCY.restype = c_double
        self._dll.SK_GETMINLINEFREQUENCY.restype = c_double
        self._dll.SK_GETMAXLINEFREQUENCY.restype = c_double
        self.pixel_pitch = self._dll.SK_GETPIXWIDTH(self._camera_id)
        self.pixel_count = self._dll.SK_GETPIXELSPERLINE(self._camera_id)
        self.min_exposure_time = 1 / self._dll.SK_GETMAXLINEFREQUENCY(self._camera_id)
        self.max_exposure_time = 1 / self._dll.SK_GETMINLINEFREQUENCY(self._camera_id)

    def __str__(self):
        with self.lock:
            self._dll.SK_GETCAMTYPE.restype = POINTER(c_wchar)
            camera_type = c_wchar_p.from_buffer(self._dll.SK_GETCAMTYPE(self._camera_id))
            camera_sn = create_unicode_buffer(12)
            usb_version = create_unicode_buffer(12)
            self._dll.SK_GETSN(self._camera_id, camera_sn, 12)
            self._dll.SK_GETUSBVERSION(self._camera_id, usb_version, 12)
            return f'<Camera: {camera_type.value}, Serial: {camera_sn.value}, Interface: {usb_version.value}>'

    def set_camera_gain(self, camera_gain):
        with self.lock:
            if camera_gain != self.camera_gain:
                self.camera_gain = camera_gain
                if self._dll.SK_SETGAIN(self._camera_id, camera_gain-1, self._channel):
                    raise RuntimeError('Command write error! SK2048U3')

    def set_exposure_time(self, exposure_time):
        with self.lock:
            if exposure_time < self.min_exposure_time: exposure_time = self.min_exposure_time
            if exposure_time > self.max_exposure_time: exposure_time = self.max_exposure_time
            if exposure_time != self.exposure_time:
                self.exposure_time = exposure_time
                if self._dll.SK_SETEXPOSURETIME(self._camera_id, c_double(self.exposure_time)):
                    raise RuntimeError('Command write error! SK2048U3')

    def get_frame(self):
        with self.lock:
            data = np.zeros(2*self.pixel_count, dtype=np.uint8)
            data_p = data.ctypes.data_as(POINTER(c_uint8))
            result = self._dll.SK_GRAB(self._camera_id, data_p, c_size_t(1), c_size_t(1000), c_bool(0), 0, 0)
            if result != 15:
                raise RuntimeError(f'Data read error! SK2048U3')
            return data[0::2]/4096 + data[1::2]/16

    def close_camera(self):
        with self.lock:
            self._dll.SK_CLOSECAMERA(self._camera_id)


class TCE1304U(Camera):
    '''
    Mightex USB 2.0 CCD Line Scan Camera
    The libusb-win32 driver can be installed using Zadig,
    which can be downloaded from https://zadig.akeo.ie/
    '''
    def __init__(self):
        self.lock = Lock()
        self.pixel_pitch = 8
        self.pixel_count = 3648
        self.min_exposure_time = 0.1
        self.max_exposure_time = 1000
        self._dev = find(idVendor=0x04B4, idProduct=0x0328)
        if not self._dev:
            raise RuntimeError('Camera not found!')
        self._dev.set_configuration()
        
    def __str__(self):
        with self.lock:
            self._write(0x21, 0)
            info = self._read(43).tobytes().decode()
            return f'<Camera: {info[1:11]}, Serial: {info[15:28]}, Date: {info[29:39]}>'
    
    def _read(self, size):
        result = self._dev.read(0x81, size+2)
        if result[0] != 1 or result[1] != size:
            raise RuntimeError('Command read error! TCE1304U')
        return result[2:]
    
    def _write(self, cmd, *data):
        result = self._dev.write(0x01, bytes([cmd, len(data)] + list(data)))
        if result != len(data)+2:
            raise RuntimeError('Command write error! TCE1304U')
    
    def set_camera_gain(self, camera_gain):
        self.camera_gain = camera_gain

    def set_exposure_time(self, exposure_time):
        with self.lock:
            if exposure_time < self.min_exposure_time: exposure_time = self.min_exposure_time
            if exposure_time > self.max_exposure_time: exposure_time = self.max_exposure_time
            if exposure_time != self.exposure_time:
                self.exposure_time = exposure_time
                self._exposure_time = int(10 * exposure_time)
                self._write(0x31, self._exposure_time//0x100, self._exposure_time%0x100)
                self._write(0x30, 0)
        
    def get_frame(self):
        with self.lock:
            self._write(0x34, 1)
            data = np.frombuffer(self._dev.read(0x82, 7680), '<u2')
            if data[3833] != self._exposure_time:
                raise RuntimeError('Data read error! TCE1304U')
            dark_current = np.average(data[16:29])
            return (data[32:3680] - dark_current)/ 65536


if __name__ == '__main__':
    for Camera in [SK2048U3, TCE1304U]:
        cam = Camera()
        print('Camera detected:', cam)
        cam.set_exposure_time(0.1)
        data = cam.get_frame()
        cam.close_camera()
        print(f'pixel_count: {len(data)}')
