import gtts
from playsound import playsound

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os, ssl


print("")
recap = input("Do you want a recap of the previous project? Type 'yes' or 'no' :-").lower()

if(recap == "yes"):

    print("")
    print("Previously, you saw:-")
    print("")
    tts = gtts.gTTS("You're friend Erica had found some letters and messages in code and cryptics regarding computing and the modern day tech. You have helped her with these letters being a cryptography expert. And now comes the next part where you have to test the model out and show Erica how it works. Ready?")
    tts.save("hello.mp3")
    playsound("hello.mp3")
    print("")
    print("")
    recap = input("Do you want to check the code now or end the program. 'check' or 'end' :-").lower()

    if(recap == "check"):

        X = np.load('image.npz')['arr_0']
        Y = pd.read_csv("labels.csv")["labels"]
        #print(pd.Series(Y).value_counts())
        classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        nclasses = len(classes)

        x_train, x_test, y_train, y_test = ttsplit(X, Y, random_state = 0, train_size = 7500, test_size = 2500)

        scaled_xtr = x_train/255.0
        scaled_xts = x_test/255.0

        classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(scaled_xtr, y_train)

        y_pred = classifier.predict(scaled_xts)

        print("Accuracy is :", accuracy_score(y_test, y_pred))


        cap = cv2.VideoCapture(0)

        while True:

            ret,frame = cap.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            upper_left = (int(width/2-56), int(height/2-56))
            down_right = (int(width/2+56), int(height/2+56))
            cv2.rectangle(gray, upper_left, down_right, (0,255,0), 2)
            roi = gray[upper_left[1] : down_right[1], upper_left[0] : down_right[0]]

            img = Image.fromarray(roi)
            imgbw = img.convert('L')
            resized = imgbw.resize((22,30), Image.ANTIALIAS)

            invert = PIL.ImageOps.invert(resized)

            min_pix = np.percentile(invert, 20)
            scaled = np.clip(invert-min_pix, 0, 255)
            max_pix = np.max(invert)
            scaled = np.asarray(scaled)/max_pix

            test = np.array(scaled).reshape(1, 660)
            test_pred = classifier.predict(test)
            print("PREDICTED ALPHABET IS :", test_pred)
            
            cv2.imshow("Prediction", gray)
            
            if cv2.waitKey(1) and 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif(recap == "end"):
        print("Goodbye!")

elif(recap == "no"):

    print("")
    print("Well then. Let's go testing. Below given is the code with comments. Check it out and start away!")
    print("")

    X = np.load('image.npz')['arr_0']
    Y = pd.read_csv("labels.csv")["labels"]
    #print(pd.Series(Y).value_counts())
    classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    nclasses = len(classes)

    x_train, x_test, y_train, y_test = tts(X, Y, random_state = 0, train_size = 7500, test_size = 2500)

    scaled_xtr = x_train/255.0
    scaled_xts = x_test/255.0

    classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(scaled_xtr, y_train)

    y_pred = classifier.predict(scaled_xts)

    print("Accuracy is :", accuracy_score(y_test, y_pred))


    cap = cv2.VideoCapture(0)

    while True:

        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        down_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, down_right, (0,255,0), 2)
        roi = gray[upper_left[1] : down_right[1], upper_left[0] : down_right[0]]

        img = Image.fromarray(roi)
        imgbw = img.convert('L')
        resized = imgbw.resize((22,30), Image.ANTIALIAS)

        invert = PIL.ImageOps.invert(resized)

        min_pix = np.percentile(invert, 20)
        scaled = np.clip(invert-min_pix, 0, 255)
        max_pix = np.max(invert)
        scaled = np.asarray(scaled)/max_pix

        test = np.array(scaled).reshape(1, 660)
        test_pred = classifier.predict(test)
        print("PREDICTED ALPHABET IS :", test_pred)
        
        cv2.imshow("Prediction", gray)
        
        if cv2.waitKey(1) and 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


