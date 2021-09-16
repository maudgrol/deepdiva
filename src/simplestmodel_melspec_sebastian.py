import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from os import listdir
import soundfile as sf

PATH =  "../small_data/"

def to_16_bit_PCM(filepath):
    print(listdir(filepath))
    for i, file in enumerate(listdir(f'{filepath}/audio')):
        print(i)
        data, samplerate = sf.read(f"{f'{filepath}/audio'}/{file}")
        sf.write(f'{filepath}/audio16bit/{file}', data, samplerate, subtype='PCM_16')


def wav_to_mel(sample):
    audio_binary = tf.io.read_file(sample)

    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)

    waveform = tfio.audio.resample(audio, 44100, 8000, name=None)[:, 0]

    audio = tf.cast(waveform,
                    tf.float32)  # / 32768.0 #normalizing the loudness of the audio the tf way... the number is the max number for this kind of datatype (could be another datatyp next time...).. is actually not necessary
    spectrogram = tfio.audio.spectrogram(
        audio, nfft=1024, window=1024, stride=64)

    spectrogram = tfio.audio.melscale(
        spectrogram, rate=44100, mels=64, fmin=0, fmax=2000)

    spectrogram /= tf.math.reduce_max(spectrogram)  # checks the max in all dimensions. #here we normalize the spectrogram
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # dd a dimension so we can do convolution later
    spectrogram = tf.image.resize(spectrogram, (64,64))  # image resize expects a picture (3 or more dimensions , it expects the first 2 dimensions to be height and width)
    spectrogram = tf.transpose(spectrogram, perm=(1, 0, 2))  # dimensions are numbered 0,1,2 ... and here we switch the first and second
    spectrogram = spectrogram[::-1, :, :]

    return spectrogram

#to_16_bit_PCM("../small_data/")
mel = wav_to_mel("../small_data/audio16bit/train_0_output_4.wav")
print(mel.shape)
plt.imshow(mel[:, :, 0])
plt.show()
plt.close()


wav_to_wav16binarytf

npythingy to_label ()

zip those two things

#give me only this
[i for i in ALLFILES if 'train_' in i]

#
file.split("output_", 1)[1] #split it once after output
file.split(".", 1)[0] #take the first [0] part of the split