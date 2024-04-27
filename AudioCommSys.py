import speech_recognition as sr
from gtts import gTTS
import os
import platform

language = 'en'

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please say something....")
        audio = recognizer.listen(source, timeout=2)
        try:
            print("You said: \n" + recognizer.recognize_google(audio))
            return recognizer.recognize_google(audio)
        except Exception as e:
            print("Error: " + str(e))

def text_to_speech(text):
    output = gTTS(text=text, lang=language, slow=False)
    output_path = "output.mp3"
    output.save(output_path)

    if platform.system() == 'Windows':
        # Use os.system for Windows
        os.system(f'start {output_path}')
    else:
        # Use playsound for other platforms
        from playsound import playsound
        playsound(output_path)

def main():
    text_to_speech("Testing, Testing, Testing")

if __name__ == "__main__":
    main()
