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
	file1_header = "vacancy,date,email,name,question_number,v_aggressive,v_rude,v_hessitant,v_friendly,v_dissapointed,v_suprised,v_neutral"
	#create file and write header to it
	f = open(os.path.join( temp_folder_path , "candidate_analysis.csv" ), "w")
	f.write("".join((file1_header,"\n")) )
	f.close()
	return 1


def append_to_output_files(vacancy,email,name,question_num,video_pred_results):
	f = open(os.path.join( temp_folder_path , "candidate_analysis.csv" ), "a")
	now = datetime.now()
	date = now.strftime("%d/%m/%Y %H:%M:%S")
	question_num_str = str(question_num)
	video_pred_results_str = str(video_pred_results)
	video_pred_results_str = text_emotions_str.strip("{}")
	record = "".join((vacancy, ",", date , "," , email , "," , name , "," , question_num_str, "," ,video_pred_results_str,"\n" ))
	f.write(record)
	f.close()

	return 1
