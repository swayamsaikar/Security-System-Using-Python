# FACE DETECTION BY SWAYAM SAI KAR

import wave
import pyaudio
import cv2
import random
import winsound

# I am using haarcascade algorithim to detect faces

cameraObject = cv2.VideoCapture(0)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# https://stackoverflow.com/questions/892199/detect-record-audio-in-python
def takeSoundInput():

    randomRecordingName = random.randint(1, 100)

    # the file name output you want to record into
    filename = f"recorded{randomRecordingName}.wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 44100
    record_seconds = 10
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()


def main():
    while(True):
        randomImageName = random.randint(0, 1000)

        _, frame = cameraObject.read()

        grayScaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = trained_face_data.detectMultiScale(grayScaledImage)

        # print(face_coordinates)

        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            path = f"images/image{randomImageName}.png"
            cv2.imwrite(path, frame)
            winsound.PlaySound("alert.wav", winsound.SND_ASYNC)

        cv2.imshow("Swayam's Face Detector", frame)
        # You can see here the values
        # print((x, y), (x+w, y+h))

        key = cv2.waitKey(10)
        if key == 113:
            break


if __name__ == '__main__':
    takeSoundInput()
    main()


print("Code Completed")
