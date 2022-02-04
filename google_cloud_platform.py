from google.cloud import storage
from google.cloud import speech
import moviepy.editor as mp
import os



temp_folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) , 'temp_folder' )

#get GCS bucket object 
def upload_video_to_gcs(vacancy,video,applicant_mail,question_num):
	# Setting credentials using the downloaded JSON file
	client = storage.Client.from_service_account_json(json_credentials_path='hackathon-sa-credentials.json')
	# Creating bucket object
	bucket = client.get_bucket('hiring-application-bucket')
	# Name of the destination file in the bucket
	gcs_file_name = "".join(("applicants/videos/",applicant_mail,"_Q",str(question_num),".mp4"))
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

	