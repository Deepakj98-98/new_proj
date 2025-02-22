import speech_recognition as sr
import pyttsx3
import keyboard
#from new_retrieve import *

# Initialize the recognizer 
r = sr.Recognizer() 
def SpeakText(command):
    
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    return command
    
    
# Loop infinitely for user to
# speak

while True:
    if keyboard.is_pressed("esc"):
        print("Esc key pressed. Exiting...")
        break  # Exit the loop    
    
    # Exception handling to handle
    # exceptions at the runtime
    try:
        
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("speak now")
            #listens for the user's input 
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            print(MyText)
            if MyText=="exit":
                break

            '''
            SpeakText(MyText)
            results = retrieve_from_qdrant(MyText)

            # Print results
            for result in results:
                print(f"ID: {result['id']}, Score: {result['score']}, Text: {result['payload'].get('text')}")
                '''


    except sr.RequestError as e:
        print("error")
        
    except sr.UnknownValueError:
        print("error")
