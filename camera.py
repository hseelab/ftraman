import numpy as np
import ctypes as c
from time import sleep
from threading import Lock
from usb.core import find
from ctypes import cdll, create_string_buffer, create_unicode_buffer, POINTER
from ctypes import c_wchar, c_wchar_p, c_bool, c_int, c_uint8, c_size_t, c_double
from pylablib.devices import Andor


class Camera(object):
    cam = None
    is_dummy = False
    gain = 0
    exposure = 10
    get_temperature = None

    def set_gain(self, gain):
        self.gain = gain

    def set_exposure(self, exposure):
        self.exposure = exposure

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
        return f'<Camera: Dummy, {self.pixel_count}px, {self.pixel_pitch}um>'

    def get_frame(self):
        sleep(0.001 * self.exposure)
        x = self.pixel_pitch * (np.random.rand(1)[0] + np.arange(-self.pixel_count//2, self.pixel_count//2))
        y = np.random.rand(self.pixel_count) - 0.5
        for a, k in self._peaks:
            y += self.gain * a * (1 + np.cos(k*x)) * np.exp(-x**2/self._sigma**2)
        return np.minimum(1, y/100).reshape(1, self.pixel_count)

    def set_dummy_signal(self, cutoff, *peaks, fwhm=0.5):
        self._sigma = fwhm * self.pixel_pitch * self.pixel_count / (2 * np.sqrt(np.log(2)))
        self._peaks = [(a / 2, cutoff * np.pi / (l * self.pixel_pitch)) for a, l in peaks]


class TCE1304(Camera):
    '''
    Mightex USB 2.0 CCD Line Scan Camera
    The libusb-win32 driver can be installed using Zadig,
    which can be downloaded from https://zadig.akeo.ie/
    '''
    def __init__(self):
        self.lock = Lock()
        self.pixel_pitch = 8
        self.pixel_count = 3648
        self.min_exposure = 0.1
        self.max_exposure = 1000
        self.dev = find(idVendor=0x04B4, idProduct=0x0328)
        if not self.dev:
            raise RuntimeError('Camera not found!')
        self.dev.set_configuration()

    def __str__(self):
        with self.lock:
            self._write(0x21, 0)
            info = self._read(43).tobytes().decode()
            return f'<Camera: {info[1:11]}, Serial: {info[15:28]}, Date: {info[29:39]}>'

    def _read(self, size):
        result = self.dev.read(0x81, size+2)
        if result[0] != 1 or result[1] != size:
            raise RuntimeError('Command read error! TCE1304U')
        return result[2:]

    def _write(self, cmd, *data):
        result = self.dev.write(0x01, bytes([cmd, len(data)] + list(data)))
        if result != len(data)+2:
            raise RuntimeError('Command write error! TCE1304U')

    def set_gain(self, gain):
        self.gain = gain

    def set_exposure(self, exposure):
        with self.lock:
            if exposure < self.min_exposure: exposure = self.min_exposure
            if exposure > self.max_exposure: exposure = self.max_exposure
            if exposure != self.exposure:
                self.exposure = exposure
                self._exposure = int(10 * exposure)
                self._write(0x31, self._exposure//0x100, self._exposure%0x100)
                self._write(0x30, 0)

    def get_frame(self):
        with self.lock:
            self._write(0x34, 1)
            data = np.frombuffer(self.dev.read(0x82, 7680), '<u2')
            if data[3833] != self._exposure:
                raise RuntimeError('Data read error! TCE1304U')
            dark_current = np.average(data[16:29])
            return ((data[32:3680] - dark_current)/ 65536).reshape(1, self.pixel_count)


class SK2048(Camera):
    '''
    Schafter Kirchhoff USB 3.0 CMOS Line Scan Camera
    The camera driver software can be downloaded from
    https://www.sukhamburg.com/products/details/SKLineScan
    '''
    def __init__(self):
        self.lock = Lock()
        self._camera_id = 0
        self._channel = 0
        self.dev = cdll.LoadLibrary('C:/Program Files/Common Files/SK/SK91USB3-WIN/SK91USB3_x64u.dll')
        self.dev.SK_LOADDLL()
        if self.dev.SK_INITCAMERA(self._camera_id):
            raise RuntimeError('Camera not found!')

        self.dev.SK_GETPIXWIDTH.restype = c_double
        self.dev.SK_GETEXPOSURETIME.restype = c_double
        self.dev.SK_GETLINEFREQUENCY.restype = c_double
        self.dev.SK_GETMINLINEFREQUENCY.restype = c_double
        self.dev.SK_GETMAXLINEFREQUENCY.restype = c_double
        self.pixel_pitch = self.dev.SK_GETPIXWIDTH(self._camera_id)
        self.pixel_count = self.dev.SK_GETPIXELSPERLINE(self._camera_id)
        self.min_exposure = 1 / self.dev.SK_GETMAXLINEFREQUENCY(self._camera_id)
        self.max_exposure = 1 / self.dev.SK_GETMINLINEFREQUENCY(self._camera_id)

    def __str__(self):
        with self.lock:
            self.dev.SK_GETCAMTYPE.restype = POINTER(c_wchar)
            camera_type = c_wchar_p.from_buffer(self.dev.SK_GETCAMTYPE(self._camera_id))
            camera_sn = create_unicode_buffer(12)
            usb_version = create_unicode_buffer(12)
            self.dev.SK_GETSN(self._camera_id, camera_sn, 12)
            self.dev.SK_GETUSBVERSION(self._camera_id, usb_version, 12)
            return f'<Camera: {camera_type.value}, Serial: {camera_sn.value}, Interface: {usb_version.value}>'

    def set_gain(self, gain):
        with self.lock:
            if gain != self.gain:
                self.gain = gain
                if self.dev.SK_SETGAIN(self._camera_id, int(102.3*gain), self._channel):
                    raise RuntimeError('Command write error! SK2048U3')

    def set_exposure(self, exposure):
        with self.lock:
            if exposure < self.min_exposure: exposure = self.min_exposure
            if exposure > self.max_exposure: exposure = self.max_exposure
            if exposure != self.exposure:
                self.exposure = exposure
                if self.dev.SK_SETEXPOSURETIME(self._camera_id, c_double(self.exposure)):
                    raise RuntimeError('Command write error! SK2048U3')

    def get_frame(self):
        with self.lock:
            data = np.zeros(2*self.pixel_count, dtype=np.uint8)
            data_p = data.ctypes.data_as(POINTER(c_uint8))
            result = self.dev.SK_GRAB(self._camera_id, data_p, c_size_t(1), c_size_t(1000), c_bool(0), 0, 0)
            if result != 15:
                raise RuntimeError(f'Data read error! SK2048U3')
            return (data[0::2]/4096 + data[1::2]/16).reshape(1, self.pixel_count)

    def close_camera(self):
        with self.lock:
            self.dev.SK_CLOSECAMERA(self._camera_id)


class DV420(Camera):
    '''
    iDus DV420A-OE Camera
    '''
    def __init__(self):
        self.lock = Lock()
        self.pixel_pitch = 26
        self.pixel_count = 1024
        self.cam = Andor.AndorSDK2Camera()
        self.cam.set_fan_mode('full')
        self.cam.set_temperature(-65)
        self.cam.setup_image_mode(0, 1024, 0, 255, 1, 1)
        self.cam.set_acquisition_mode('cont')
        self.cam.set_trigger_mode('software')
        self.cam.start_acquisition()

    def __str__(self):
        return f'<{self.cam.get_device_info()}>'

    def get_temperature(self):
        return (round(self.cam.get_temperature(), 1),
                round(self.cam.get_temperature_setpoint(), 1),
                self.cam.get_temperature_status())

    def set_temperature(self, temperature):
        self.cam.set_temperature(temperature)

    def set_gain(self, gain):
        with self.lock:
            self.cam.stop_acquisition()
            self.cam.set_amp_mode(0, 0, (gain-1)//2, (gain-1)%2)
            mode = self.cam.get_amp_mode()
            self.cam.start_acquisition()
            self.gain = gain
            self.dark = [0.0, 175.0, 240.0, 485.0, 745.0, 820.0][int(gain)]
            print(f'Horizontal Scan: {1000*mode.hsspeed_MHz:.0f} kHz')
            print(f'Preamp Gain: {mode.preamp_gain:.1f}')

    def set_exposure(self, exposure):
        with self.lock:
            self.exposure = exposure
            self.cam.stop_acquisition()
            self.cam.set_exposure(exposure/1000)
            self.cam.start_acquisition()
            print('Exposure Time:', exposure, 'ms')

    def set_roi(self, roitop, roibtm, roibin):
        with self.lock:
            roi = self.cam.setup_image_mode(0, 1024, roitop, roibtm, 1, roibin)
            return roi[2], roi[3], roi[5]

    def get_frame(self):
        with self.lock:
            self.cam.send_software_trigger()
            self.cam.wait_for_frame()
            image = self.cam.read_newest_image()
            return (image - self.dark) / 65535.0

    def close_camera(self):
        with self.lock:
            self.cam.stop_acquisition()
            self.cam.set_fan_mode('off')
            self.cam.close()


class ZL41W(Camera):
    '''
    Zyla ZL41WAVE sCMOS Camera
    '''
    def __init__(self):
        self.lock = Lock()
        self.pixel_pitch = 6.5
        self.pixel_count = 2048
        self.cam = Andor.AndorSDK3Camera()
        self.cam.set_cooler()
        self.cam.set_temperature(0)
        self.cam.set_attribute_value('CycleMode', 'Continuous')
        self.cam.set_attribute_value('TriggerMode', 'Software')
        self.cam.set_attribute_value('PixelEncoding', 'Mono16')
        self.cam.set_attribute_value('PixelReadoutRate', '270 MHz')
        self.cam.set_roi(0, 2048, 512, 1536, 1, 1)
        self.cam.start_acquisition()

    def __str__(self):
        return f'<Camera: ZL41WAVE, Serial: {self.cam.get_attribute_value("SerialNumber")}>'

    def get_temperature(self):
        return (round(self.cam.get_temperature(), 1),
                round(self.cam.get_temperature_setpoint(), 1))

    def set_temperature(self, temperature):
        self.cam.set_temperature(0)

    def set_gain(self, gain):
        with self.lock:
            self.cam.stop_acquisition()
            self.cam.set_attribute_value('PixelReadoutRate', '100 MHz' if gain > 1 else '270 MHz')
            mode = self.cam.get_attribute_value('PixelReadoutRate')
            self.cam.start_acquisition()
            self.gain = gain
            print('Pixel Readout Rate:', mode)

    def set_exposure(self, exposure):
        with self.lock:
            self.cam.stop_acquisition()
            self.cam.set_exposure(exposure/1000)
            self.cam.start_acquisition()
            self.exposure = exposure
            print('Exposure Time:', exposure, 'ms')

    def set_roi(self, roitop, roibtm, roibin):
        with self.lock:
            roi = self.cam.set_roi(0, 2048, roitop, roibtm, 1, roibin)
            return roi[2], roi[3], roi[5]

    def get_frame(self):
        with self.lock:
            self.cam.call_command('SoftwareTrigger')
            self.cam.wait_for_frame()
            acq = self.cam.read_newest_image()
            return (acq-100.0) / 65535.0

    def close_camera(self):
        with self.lock:
            self.cam.stop_acquisition()
            self.cam.close()


if __name__ == '__main__':
    cam = ZL41W()
    print(cam.get_frame())
