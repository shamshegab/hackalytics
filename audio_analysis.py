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
    

    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    livedf = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    livedf = pd.DataFrame(data=livedf)
    livedf = livedf.stack().to_frame().T
    twodim = np.expand_dims(livedf, axis=2)

    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
    emotions = {'aggressive': round(livepreds[0][0] + livepreds[0][5], 2),
                'confident': round(livepreds[0][1] + livepreds[0][6], 2),
                'hesitant': round(livepreds[0][2] + livepreds[0][7], 2),
                'friendly': round(livepreds[0][3] + livepreds[0][8], 2),
                'disappointed': round(livepreds[0][4] + livepreds[0][9], 2)}
    return emotions



if __name__ == "__main__":
    print(classify_audio('audio.mp3'))