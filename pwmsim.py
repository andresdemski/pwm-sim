from scipy.signal import periodogram
import numpy as np
import time
import matplotlib.pyplot as plt

np.seterr(divide = 'ignore') 


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
    pwm_max = 64
    upsample_factor = 1

    sample_period = 1/sample_freq

    t = get_time(start=0, ts=sample_period, stop=sim_time)
    signal = (0.1* np.sin(2 * np.pi * signal_freq * t) + 1)/2

    signal = upsample(signal, up=upsample_factor)
    t = get_time(start=0, ts=sample_period/upsample_factor, stop=sim_time)

    pwm_out = get_pwm(signal, div=pwm_max)
    pwm_period = sample_period / upsample_factor
    pwm_freq = 1/pwm_period
    pwm_resolution = pwm_period / pwm_max
    pwm_time = get_time(start=0, ts=pwm_resolution, num=len(pwm_out))

    fs = 1/pwm_resolution
    pwm_f, pwm_spectrum = periodogram(pwm_out.astype(np.float), fs=fs)
    pwm_spectrum += np.ones(len(pwm_spectrum))*1e-10 # Saco los 0s que cagan el log10
    pwm_spectrum = 20 * np.log10(pwm_spectrum)


    print(vars())
    plt.figure(0)
    plt.subplot('511')
    plt.plot(t, signal)
    plt.subplot('512')
    plt.plot(pwm_time, pwm_out)
    plt.subplot('513')
    plt.plot(pwm_f, pwm_spectrum)
    f_res = pwm_f[1]
    for h in range(6):
        plt.subplot(5,3,10 + h)
        if h == 0:
            idx_start = int(0)
            idx_end = int(signal_freq / f_res * 10)
            ticks = pwm_f[[idx_start, idx_end]]
        else:
            idx_center = int(h * pwm_freq / f_res)
            idx_spam = int(signal_freq / f_res * 20)
            idx_start = int(idx_center - idx_spam/2)
            idx_end = int(idx_center + idx_spam/2)
            ticks = pwm_f[[idx_center, idx_start, idx_end]]
        plt.plot(pwm_f[idx_start:idx_end+1], pwm_spectrum[idx_start:idx_end+1])
        plt.xticks(ticks)

    plt.show()




if __name__ == '__main__':
    main()
