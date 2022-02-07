import text2emotion as te
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def get_emotions(text):
    emotions_result={}
    result = te.get_emotion(text)
#     {'Happy': 0.33, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.67}
# {'friendly': 0.33, 'aggressive': 0.0, 'Surprise': 0.0, 'dissapointed': 0.0, 'hessitant': 0.67}
    emotions_result['friendly']=result['Happy']
    emotions_result['aggressive']=result['Angry']
    emotions_result['suprised']=result['Surprise']
    emotions_result['disappointed']=result['Sad']
    emotions_result['hesitant']=result['Fear']
    return emotions_result
