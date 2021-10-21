from flask import Flask, render_template, url_for, request, redirect , flash
import moviepy.editor as mp
import os
from google.cloud import speech
from google.protobuf.json_format import MessageToJson
from load_model import load_model
import json
#import tensorflow_hub as hub
import numpy as np
import imageio
#from fer import FER
#from fer import Video
from google.cloud import storage
from datetime import datetime


app = Flask(__name__)
#text2emotion = load_model()

module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
#model = hub.load(module_url)

@app.route('/' , methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		video = request.form['video']
		video2 = request.form['video2']
		name = request.form['name']
		email = request.form['email']

		if not video:
			flash('please upload your answer first')
		if not video2:
			flash('please upload your answer first')
		else:
			#store video to GCS
			video_list = [video,video2]
			create_output_file()
			question_num=1
			for video in video_list:
				upload_video_to_gcs(video,email,question_num)
				##call emotion detection code --Mahmoud
				#FER = FacialEmotionDetection(video)
				#print(FER)
				video_emotions = ( 1 , 2 )

				#get transript from the video
				recognized_text = get_transcript(video)
				

				##detect transcript to emotions  --Shams
				#result = text2emotion(recognized_text)
				#emotion_str = str(result)
				#print(emotion_str)
				text_emotions = ( 3 , 4 )

				##call semantic analysis code --Abbas
				#ans = {
				#	"Model_Answer": "Machine Learning Problems can be Supervised or Unsupervised",
				#	"Applicant_Answer": recognized_text
				#}
				#rate = sentence_similarity(ans)
				#print(rate)
				rate =  0.9

				append_to_output_file(email,name,question_num,rate,video_emotions,text_emotions)
				question_num += 1

			##upload results to GCS
			upload_result_to_gcs(email)
			return redirect(url_for('response'))
	return render_template('index.html')

@app.route('/review_response')
def response():
	return render_template('review_response.html')

def create_output_file():
	file_header = "date,email,name,question_number,ve_happy,ve_sad,te_neutral,te_admiration"
	#create file and write header to it
	f = open("candidate_analysis.csv", "w")
	f.write("".join((file_header,"\n")) )
	f.close()
	return 1

def append_to_output_file(email,name,question_num,rate,video_emotions,text_emotions):
	f = open("candidate_analysis.csv", "a")
	now = datetime.now()
	date = now.strftime("%d/%m/%Y %H:%M:%S")
	question_num_str = str(question_num)
	rate_str = str(rate)
	video_emotions_str = str(video_emotions)
	video_emotions_str = video_emotions_str.strip("()")
	text_emotions_str = str(text_emotions)
	text_emotions_str = text_emotions_str.strip("()")
	record = "".join((date , "," , email , "," , name , "," , question_num_str , "," , rate_str , "," , video_emotions_str , "," , text_emotions_str,"\n" ))
	f.write(record)
	f.close()
	return 1

#get GCS bucket object 
def upload_video_to_gcs(video,applicant_mail,question_num):
	# Setting credentials using the downloaded JSON file
	client = storage.Client.from_service_account_json(json_credentials_path='speech_to_text_credentials.json')
	# Creating bucket object
	bucket = client.get_bucket('hackalytics')
	# Name of the destination file in the bucket
	gcs_file_name = "".join(("applicants/",applicant_mail,"/Q",str(question_num)))
	print(gcs_file_name)
	object_name_in_gcs_bucket = bucket.blob(gcs_file_name)
	object_name_in_gcs_bucket.upload_from_filename(video)
	return 1


#get GCS bucket object 
def upload_result_to_gcs(applicant_mail):
	# Setting credentials using the downloaded JSON file
	client = storage.Client.from_service_account_json(json_credentials_path='speech_to_text_credentials.json')
	# Creating bucket object
	bucket = client.get_bucket('hackalytics')
	# Name of the destination file in the bucket
	gcs_file_name = "".join(("applicants/",applicant_mail,"/candidate_analysis.csv"))
	print(gcs_file_name)
	object_name_in_gcs_bucket = bucket.blob(gcs_file_name)
	object_name_in_gcs_bucket.upload_from_filename("candidate_analysis.csv")
	return 1
		

#get transript from the video
def get_transcript(video):
	##convert video to audio  --Yara
	VideoToAudio(video,"michael2.mp3")
	##call speech to text  --Yara
	operation = AudioToText("michael2.mp3")

	##transform text to get transcript 
	recognized_text = ''
	for i in range(len(operation.results)):    
		recognized_text += operation.results[i].alternatives[0].transcript
	print(recognized_text)
	return recognized_text

def VideoToAudio (VideoPath ,AudioPath ):
	# Insert Local Video File Path
	clip = mp.VideoFileClip(VideoPath)
	clip.audio.write_audiofile(AudioPath)
	return 1

def AudioToText (VideoPath):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS']='speech_to_text_credentials.json'
	speech_client = speech.SpeechClient()
	media_file_name_mp3 = VideoPath
	with open(media_file_name_mp3, 'rb') as f1:
		byte_data_mp3 = f1.read()
	audio_mp3 = speech.RecognitionAudio(content=byte_data_mp3)

	config_mp3 = speech.RecognitionConfig(
		sample_rate_hertz=48000,
		enable_automatic_punctuation=True,
		language_code='en-US'
	)
	response_standard_mp3 = speech_client.recognize(
		config=config_mp3,
		audio=audio_mp3
	)
	return response_standard_mp3

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


def sentence_similarity(answers):
	assert len(answers) == 2
	assert "Model_Answer" in answers
	assert "Applicant_Answer" in answers

	answers_list = [a for a in answers.values()]
	embeddings = model(answers_list)
	similarity = np.inner(embeddings, embeddings)
	return similarity[0][1]


if __name__ == "__main__":
	app.run(debug=True)