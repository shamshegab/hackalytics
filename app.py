from flask import Flask, render_template, url_for, request, redirect , flash
import moviepy.editor as mp
import os
from google.cloud import speech
from video_analysis import classify_video
from audio_analysis import classify_audio
from google.cloud import storage
from datetime import datetime


app = Flask(__name__)
temp_folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) , 'temp_folder' )


@app.route('/' , methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		video = request.files['video']
		video2 = request.files['video2']
		video3 = request.files['video3']
		video4 = request.files['video4']
		name = request.form['name']
		email = request.form['email']
		if not video:
			flash('please upload your answer first')
		if not video2:
			flash('please upload your answer first')
		else:
			create_temp_dir()
			create_output_files()

			video_list = [video,video2,video3,video4]
			
			question_num=1
			for i in range(len(video_list)):
				video_path = get_video_path(video_list[i],question_num)
				print(video_path)

				#store video to GCS
				upload_video_to_gcs(video_path,email,question_num)

				#get transript from the video
				recognized_text = get_transcript(video_path)
				print(recognized_text)
				
				
				print("Begin Audio analysis")
				audio_pred_results = classify_audio(os.path.join( temp_folder_path , "video_audio.mp3" ))
				print("Audio analysis results:")
				print(audio_pred_results)

				print("Begin Video analysis")
				video_pred_results = classify_video(video_path,100)
				print("Video analysis results:")
				print(video_pred_results)
				
				append_to_output_files(email,name,question_num)
				question_num += 1
				remove_temp_video(video_path)

			##upload results to GCS
			upload_result_to_gcs(email)
			return redirect(url_for('response'))
	return render_template('index.html')

@app.route('/review_response')
def response():
	return render_template('review_response.html')


def create_temp_dir():
	isExist = os.path.exists(temp_folder_path)
	if not isExist:
		# Create a new directory because it does not exist 
		os.mkdir(temp_folder_path, 0o777)
		print("The temp directory is created!")
	return 1

def remove_temp_video(video_path):
	os.remove(video_path)
	return 1

def get_video_path(video,question_num):
	videofilename = "".join((str(question_num),"-",video.filename))
	video.save(os.path.join( temp_folder_path , videofilename))
	video_path = os.path.join( temp_folder_path , videofilename )
	return video_path

def create_output_files():
	file1_header = "date,email,name,question_number"
	#create file and write header to it
	f = open(os.path.join( temp_folder_path , "candidate_analysis.csv" ), "w")
	f.write("".join((file1_header,"\n")) )
	f.close()
	return 1

def append_to_output_files(email,name,question_num):
	f = open(os.path.join( temp_folder_path , "candidate_analysis.csv" ), "a")
	now = datetime.now()
	date = now.strftime("%d/%m/%Y %H:%M:%S")
	question_num_str = str(question_num)
	#video_emotions_str = str(video_emotions)
	#video_emotions_str = video_emotions_str.strip("()")
	#text_emotions_str = str(text_emotions)
	#text_emotions_str = text_emotions_str.strip("()")
	record = "".join((date , "," , email , "," , name , "," , question_num_str ,"\n" ))
	f.write(record)
	f.close()

	return 1

#get GCS bucket object 
def upload_video_to_gcs(video,applicant_mail,question_num):
	# Setting credentials using the downloaded JSON file
	client = storage.Client.from_service_account_json(json_credentials_path='hackathon-sa-credentials.json')
	# Creating bucket object
	bucket = client.get_bucket('hiring-application-bucket')
	# Name of the destination file in the bucket
	gcs_file_name = "".join(("applicants/videos/",applicant_mail,"_Q",str(question_num)))
	object_name_in_gcs_bucket = bucket.blob(gcs_file_name)
	object_name_in_gcs_bucket.upload_from_filename(video)
	return 1


#get GCS bucket object 
def upload_result_to_gcs(applicant_mail):
	# Setting credentials using the downloaded JSON file
	client = storage.Client.from_service_account_json(json_credentials_path='hackathon-sa-credentials.json')
	# Creating bucket object
	bucket = client.get_bucket('hiring-application-bucket')
	# Name of the destination file in the bucket
	gcs_file_name = "".join(("applicants/results/",applicant_mail,"_candidate_analysis.csv"))
	object_name_in_gcs_bucket = bucket.blob(gcs_file_name)
	object_name_in_gcs_bucket.upload_from_filename(os.path.join( temp_folder_path , "candidate_analysis.csv" ))

	return 1
		

#get transript from the video
def get_transcript(video):
	##convert video to audio  --Yara
	VideoToAudio(video, os.path.join( temp_folder_path , "video_audio.mp3" ))

	##call speech to text  --Yara
	operation = AudioToText(os.path.join( temp_folder_path , "video_audio.mp3" ))

	##transform text to get transcript 
	recognized_text = ''
	for i in range(len(operation.results)):    
		recognized_text += operation.results[i].alternatives[0].transcript
	return recognized_text

def VideoToAudio (VideoPath ,AudioPath ):
	# Insert Local Video File Path
	clip = mp.VideoFileClip(VideoPath)
	clip.audio.write_audiofile(AudioPath)
	clip.reader.close()
	clip.close()
	del clip.reader
	del clip
	return 1

def AudioToText (AudioPath):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS']='hackathon-sa-credentials.json'
	speech_client = speech.SpeechClient()
	with open(AudioPath, 'rb') as f1:
		byte_data_mp3 = f1.read()
		f1.close()
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

	

if __name__ == "__main__":
	app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",8080) ) )
