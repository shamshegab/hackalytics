from fer import FER
from fer import Video


def FacialEmotionDetection(videoName):
    videofile = videoName
# Face detection
    detector = FER(mtcnn=True)
# Video predictions
    video = Video(videofile)
# Output list of dictionaries
    raw_data = video.analyze(detector, display=False)
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)
# Plot emotions
#   fig = df.plot(figsize=(20, 16), fontsize=26).get_figure()
    return df


if __name__ == "__main__":
    FacialEmotionDetection("test.mp4")