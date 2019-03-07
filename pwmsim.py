import numpy as np
import time
#import matplotlib.pyplot as plt


def get_pwm(signal, div=256):
    """ signal: signal
        div: PWM divisions/time resolution
    """
    pwm_on = np.around(signal * div)
    pwm_off = div - pwm_on
    pwm = b''.join([b'\x01' * int(pwm_on[i]) +
                    b'\x00' * int(pwm_off[i])
                    for i in range(len(signal))])
    return np.frombuffer(pwm, dtype=np.int8)

def get_time(start, ts, stop=None, num=None):
    if stop is not None:
        return np.arange(start, stop, ts)
    elif num is not None:
        return np.arange(num) * ts + start
    else:
        raise ValueError("stop and num are None")

def upsample(signal, up):
    assert isinstance(up, int), "up must be an integer"
    return signal.repeat(up)


def main():
    signal_freq = 1e3
    signal_period = 1/signal_freq

    sim_time = 10 * signal_period
    sample_freq = 1e6
    pwm_max = 256
    upsample_factor = 1

    fft_upsample_factor = 10

    sample_period = 1/sample_freq


    t = get_time(start=0, ts=sample_period, stop=sim_time)
    signal = (np.sin(2 * np.pi * signal_freq * t) + 1)/2

    signal = upsample(signal, up=upsample_factor)
    t = get_time(start=0, ts=sample_period/upsample_factor, stop=sim_time)

    pwm_out = get_pwm(signal, div=pwm_max)
    pwm_period = sample_period / upsample_factor
    pwm_resolution = pwm_period / pwm_max
    pwm_time = get_time(start=0, ts=pwm_resolution, num=len(pwm_out))

    pwm_out = upsample(pwm_out, fft_upsample_factor)
    pwm_spectrum = np.abs(np.fft.fft(pwm_out.astype(np.float)))
    pwm_f = np.arange(len(pwm_spectrum)) / (pwm_resolution / fft_upsample_factor)




if __name__ == '__main__':
    main()
