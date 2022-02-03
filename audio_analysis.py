import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json


def classify_audio(audio_path):
    # loading json and creating model
    json_file = open('audio_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("audio_model.h5")
    print("Loaded SER model from disk")
    

    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    livedf2 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)

    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    liveabc = str(liveabc[0])

    emotions_dict = {'0': 'aggressive', '1': 'confident', '2': 'hesitant', '3': 'friendly', '4': 'disappointed',
                     '5': 'aggressive', '6': 'confident', '7': 'hesitant', '8': 'friendly', '9': 'disappointed'}
    emotion = emotions_dict[liveabc]
    return emotion



if __name__ == "__main__":
    print(classify_audio('audio.mp3'))