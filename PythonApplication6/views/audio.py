import speech_recognition as  sr

#     r.pause_threshold=1
#     r.adjust_for_ambient_noise(source, duration=1)# use the default microphone as the audio source

r = None
audio = None

def audio_file_to_text(filename):
    file = 'audio_file.wav'
    print(sr.__version__)
    try:
        r = sr.Recognizer()
        with sr.AudioFile(file) as source:
            print('FILEOPENED')
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            print(text)
    except Exception as ex:
        print(ex)

def start_recording():
    global r, audio
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening")
        audio = r.listen(source)

def stop_recording():
    global r, audio
    try:
        print("audio recorded")
        text = r.recognize_google(audio).lower()
        r = None
        audio = None
        print(text)
        return text
        
    except sr.UnknownValueError:
        return "could notunderstand audio"
