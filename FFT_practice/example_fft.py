import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # data generation 1
    Fs = 1000
    T = 1/Fs
    end_time = 1
    time = np.linspace(0, end_time, Fs)
    amp = [2, 1, 0.5, 0.2]
    freq = [10, 20, 30, 40]

    signal_1 = amp[0]*np.sin(freq[0]*2*np.pi*time)
    signal_2 = amp[1]*np.sin(freq[1]*2*np.pi*time)
    signal_3 = amp[2]*np.sin(freq[2]*2*np.pi*time)
    signal_4 = amp[3]*np.sin(freq[3]*2*np.pi*time)

    signal = signal_1 + signal_2 + signal_3 + signal_4

    plt.plot(time, signal)
    plt.savefig("given_periodic_curves_1.png")
    plt.clf()
    
    # data generation 2
    signal_list = []
    for i in range(len(amp)):
        signal_list.append(amp[i]*np.sin(freq[i]*2*np.pi*time))
    signal = sum(signal_list)
    
    plt.plot(time, signal)
    plt.savefig("given_periodic_curves_2.png")
    plt.clf()
    
    # fft
    s_fft = np.fft.fft(signal) # 추후 IFFT를 위해 abs를 취하지 않은 값을 저장한다.
    amplitude = abs(s_fft)*(2/len(s_fft)) # 2/len(s)을 곱해줘서 원래의 amp를 구한다.
    frequency = np.fft.fftfreq(len(s_fft), T)

    plt.xlim(0, 50)
    plt.stem(frequency, amplitude)
    plt.grid(True)
    plt.savefig("fft_result.png")
    plt.clf()
    
    # pick 
    fft_freq = frequency.copy()
    peak_index = amplitude[:int(len(amplitude)/2)].argsort()[-1]
    peak_freq = fft_freq[peak_index]
    
    # inverse
    fft_1x = s_fft.copy()
    fft_1x[fft_freq!=peak_freq] = 0
    filtered_data = 2*np.fft.ifft(fft_1x)
    cycle = round(Fs/peak_freq)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.title('1X sin wave')
    plt.plot(filtered_data[:400], marker='o', color='darkgreen', alpha=0.3)
    plt.subplot(2, 1, 2)
    plt.title('1-period graph')
    plt.plot(signal[:400], marker='o', color='lightgrey')
    plt.plot(signal[:cycle], color='indigo')
    plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    plt.savefig("ifft_result.png")
    plt.clf()
    
    
    
    
    