import os
from datetime import datetime

temp_folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) , 'temp_folder' )


def create_temp_dir():
	isExist = os.path.exists(temp_folder_path)
	if not isExist:
		# Create a new directory because it does not exist 
		os.mkdir(temp_folder_path, 0o777)
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
	file1_header = "name,job,Question,emotion,'emotion source',value,'emotionl intelligence','English Fluency','Total rank'"
	#create file and write header to it
	f = open(os.path.join( temp_folder_path , "candidate_analysis.csv" ), "w")
	f.write("".join((file1_header,"\n")) )
	f.close()
	return 1


def append_to_output_files(vacancy, email, name, question_num, source, emotion, value, emotional_intelligence, fluency_score, final_score):
	f = open(os.path.join(temp_folder_path , "candidate_analysis.csv" ), "a")
	now = datetime.now()
	date = now.strftime("%d/%m/%Y %H:%M:%S")
	question_num_str = str(question_num)
	value_str = str(value)
	record = "".join((email, ",", name, ",", vacancy, ",", date, ",", question_num_str, ",", source, ",", emotion, ",",
	 				value_str, ",", str(emotional_intelligence), ",", str(fluency_score), ",", str(final_score), "\n"))
	f.write(record)
	f.close()
	return 1


def append_output(vacancy, email, name, question_num, fluency_score, audio_pred_results, video_pred_results, text_emotions):

	emotion_score_mapping = {'aggressive': 0.0,
							'rude': 0.0,
							'disappointed': 0.3,
							'hesitant': 0.5,
							'suprised': 0.6,
							'neutral': 0.7,
							'confident': 1.0,
							'friendly': 1.0}

	emotional_intelligence = 0
	for em, em_score in emotion_score_mapping.items():
		if (em in audio_pred_results) and (em in video_pred_results) and (em in text_emotions):
			normalised_emotion = (audio_pred_results[em] + video_pred_results[em] + text_emotions[em])/3
		elif em in audio_pred_results:
			normalised_emotion = audio_pred_results[em]/3
		elif (em in video_pred_results) and (em in text_emotions):
			normalised_emotion = (video_pred_results[em] + text_emotions[em])/3
		elif em in video_pred_results:
			normalised_emotion = video_pred_results[em]/3

		emotional_intelligence = emotional_intelligence + (normalised_emotion * em_score)
	
	fluency_score = float(fluency_score)/100
	final_score = (emotional_intelligence + fluency_score)/2

	for emotion, value in audio_pred_results.items():
		append_to_output_files(vacancy, email, name, question_num, 'Audio', emotion, value,
												 emotional_intelligence, fluency_score, final_score)
	
	for emotion, value in video_pred_results.items():
		append_to_output_files(vacancy, email, name, question_num, 'Video', emotion, value,
												 emotional_intelligence, fluency_score, final_score)

	for emotion, value in text_emotions.items():
		append_to_output_files(vacancy, email, name, question_num, 'Text', emotion, value,
												 emotional_intelligence, fluency_score, final_score)
