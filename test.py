import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import webrtcvad
import torchaudio

class csample:
    def __init__(self, path, vad_mode, chunk_num, vad):
        self.path = path
        self.vad = webrtcvad.Vad()
        #self.vad = vad
        self.vad.set_mode(vad_mode)
        self.chunk_num = chunk_num
        self.sample, self.fs = torchaudio.load(self.path)
    
    def wave_f32toint16(self, sample):
        sample = (sample * (2**16 / 2)).numpy().astype(np.int16)
        return sample

    def reset_vad(self):
        return 
    
    def sample_vad(self):
        vad_result = []
        sample = self.wave_f32toint16(self.sample)
        sample_chunk = np.reshape(sample[:1, :].T, (self.chunk_num, 320, 1))
        for chunk in sample_chunk: 
            # Scaling Chunk
            vad_result.append(self.vad.is_speech(chunk, self.fs))
        return vad_result
    
    def get_result(self):
        return self.wave_f32toint16(self.sample), self.sample_vad()
    
    
class Waveform_Aranger:
    def __init__(self, vad_mode=2, sample_rate=16000, chunk_num=50):
        self.vad = webrtcvad.Vad()
        self.vad_mode = vad_mode
        self.vad.set_mode(vad_mode)
        self.sample_rate = sample_rate
        self.chunk_num = chunk_num
        self.chunk_size = sample_rate // chunk_num
    
    def wave_f32toint16(self, sample):
        sample = (sample * (2**16 / 2)).numpy().astype(np.int16)
        return sample
    
    def get_result(self, path):
        temp = csample(path, self.vad_mode, self.chunk_num, self.vad)
        return temp.get_result()
    
    def find_startend_point(self, path):
        start = 0
        found_start = False
        end = 0
        found_end = False
        vad_result = []

        sample, fs = torchaudio.load(path)
        # float32 to int16
        sample = self.wave_f32toint16(sample)
        # 1 second to 20ms chunks
        sample_chunk = np.reshape(sample[:1, :].T, (self.chunk_num, 320, 1))

        for chunk in sample_chunk: 
            # Scaling Chunk
            vad_result.append(self.vad.is_speech(chunk, self.sample_rate))

        plt.plot(range(16000), sample[0])
        for idx, result in enumerate(vad_result):
            if result:
                if found_start == False:
                    first = idx
                    found_start = True
                plt.plot([idx * self.chunk_size, (idx+1) * self.chunk_size], [1, 1], color='red', linewidth=2)
            else:
                if found_start and found_end == False:
                    end = idx
                    found_end = True
                    
        if not found_start and not found_end:
            first = 0
            end = 49
        return first, end


annotations_file = 'data_all.csv'
file_labels = pd.read_csv(annotations_file)
wa = Waveform_Aranger(vad_mode=2)
sample_ph = file_labels.iloc[0, 2]
sample, vad_result = wa.get_result(sample_ph)

fig, ax = plt.subplots()
ax.set_title(sample_ph)
fig.subplots_adjust(bottom=0.2)
l, = ax.plot(range(16000), sample[0], lw=2)
k = [0] * 50
for idx, result in enumerate(vad_result):
    if result:
        k[idx], = ax.plot([idx * 320, (idx+1) * 320], [1, 1], color='red', linewidth=2)
        k[idx].set(antialiased=True, visible = True)
    else:
        k[idx], = ax.plot([idx * 320, (idx+1) * 320], [-1, -1], color='red', linewidth=2)
        k[idx].set(antialiased=True, visible = False)



class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        sample, vad_result = wa.get_result(file_labels.iloc[self.ind, 2])
        ax.set_title(file_labels.iloc[self.ind, 2])
        ydata = sample[0]
        l.set_ydata(ydata)
        for idx, result in enumerate(vad_result):
            if result:
                k[idx].set_ydata([1, 1])
                k[idx].set(antialiased=True, visible=True)
                ax.set_ylim(min(ydata), max(ydata))
                
            else:
                k[idx].set_ydata([-1, -1])
                k[idx].set(antialiased=True, visible=False)
                ax.set_ylim(min(ydata), max(ydata))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        sample, vad_result = wa.get_result(file_labels.iloc[self.ind, 2])
        ax.set_title(file_labels.iloc[self.ind, 2])
        ydata = sample[0]
        l.set_ydata(ydata)
        for idx, result in enumerate(vad_result):
            if result:
                k[idx].set_ydata([1, 1])
                k[idx].set(visible=True)
                ax.set_ylim(min(ydata), max(ydata))
            else:
                k[idx].set_ydata([-1, -1])
                k[idx].set(visible=False)
                ax.set_ylim(min(ydata), max(ydata))
        plt.draw()

callback = Index()
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()
