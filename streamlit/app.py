import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import mediapipe as mp
import constants as c
from tensorflow.keras.models import load_model
import utils as util
import matplotlib.pyplot as plt
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('actions.h5')

def negate(x):
    return x * -1

def is_video(file):
    if file.type.split('/')[0] == 'video':
        return True
    else:
        return False


def main():
    st.header("Детекция движений на производстве")
    st.write("Примеры для тестирования [здесь](https://drive.google.com/drive/folders/1Dlm17a-9n4bn9UoC3iq6pghwDotzrqTx?usp=sharing)")
    st.markdown("""---""")
    file_object = st.file_uploader("Загрузите видео", type=["mp4"])

    if file_object is not None and is_video(file_object):
        class_list = []
        sequence = []
    
        destination = os.path.join("uploads", file_object.name)
        with open(destination, "wb") as f:
            f.write((file_object).getbuffer())

        cap = cv2.VideoCapture(destination)
        i = 0
        t = st.empty()
        FRAME_WINDOW = st.image([])
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                i += 1
                t.markdown('Обработка... {0}'.format(i))
                
                ret, frame = cap.read()

                try:
                    image, results = util.mediapipe_detection(frame, holistic)
                except:
                    break
               
                util.draw_styled_landmarks(mp_holistic, mp_drawing, image, results)
               
                keypoints = util.extract_keypoints(results)
                sequence.append(keypoints)
            
                sequence = sequence[negate(c.SEQUENCE_LEN):]

                if len(sequence) == c.SEQUENCE_LEN:        
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    class_list.append(c.CLASSES[np.argmax(res)])
                
                frameshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frameshow, width=350)
            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        df = pd.DataFrame({'classes': class_list})
        df = df.groupby('classes')["classes"].size().reset_index(name='count')
        total_count = df['count'].sum()
        
        t.empty()

        df.sort_values(by = 'count', inplace = True, ascending = False)
        for index, row in df.iterrows():

            st.write('Движение было распознано как {0} - {1}%'.format(row['classes'], round((row['count']/total_count)*100, 2) ))


if __name__ == "__main__":
    main()