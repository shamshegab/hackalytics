from flask import Flask, render_template, url_for, request, redirect , flash
import os
import working_directory
import google_cloud_platform
#from video_analysis import classify_video
#from audio_analysis import classify_audio
#from english_fluency import fluency_detector

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
		vacancy = request.form['vacancy']

		if not video:
			flash('please upload your answer first')
		if not video2:
			flash('please upload your answer first')
		else:
			working_directory.create_temp_dir()
			print("The temp directory is created!")
			working_directory.create_output_files()

			video_list = [video,video2,video3,video4]
			
			question_num=1
			for i in range(len(video_list)):
				video_path = working_directory.get_video_path(video_list[i],question_num)
				print(video_path)	

				audio_path = os.path.join(temp_folder_path , "video_audio.mp3")
				print(audio_path)				

				#store video to GCS
				#google_cloud_platform.upload_video_to_gcs(vacancy,video_path,email,question_num)

				#get transript from the video
				recognized_text = google_cloud_platform.get_transcript(video_path)
				print(recognized_text)
				
				
				print("Begin Audio analysis")
				audio_pred_results = classify_audio(audio_path)
				print("Audio analysis results:")
				print(audio_pred_results)

				for emotion, value in audio_pred_results.items():
					append_to_output_files(vacancy, email, name, question_num, 'Audio', emotion, value)

          
        		print("Begin Fluency analysis")
				fluency_score = fluency_detector(os.path.join( temp_folder_path , "video_audio.mp3" ))
				print("Fluency Score: ", fluency_score)

        
				print("Begin Video analysis")
				video_pred_results = classify_video(video_path,100)
				print("Video analysis results:")
				print(video_pred_results)
				
				for emotion, value in video_pred_results.items():
					append_to_output_files(vacancy, email, name, question_num, 'Video', emotion, value)
				
				question_num += 1
				working_directory.remove_temp_video(video_path)

			##upload results to GCS
			google_cloud_platform.upload_result_to_gcs(email)
			return redirect(url_for('response'))
	return render_template('index.html')

@app.route('/review_response')
def response():
	return render_template('review_response.html')


if __name__ == "__main__":
	app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",8080) ) )
