<<<<<<< HEAD
import os



import requests



import tempfile



import logging



from celery import Celery



from redis_utils import publish_update 







# Import your modules



import video_analyzer



import llm_analyzer







# --- FIX HERE: Import the correct function name ---



from utils import format_final_json







# Define Queue



celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')



celery_app.conf.update(



    task_serializer='json', result_serializer='json', 



    accept_content=['json'], task_default_queue='r6_queue'



)







@celery_app.task(bind=True)



def run_analysis_task(self, video_url):



    job_id = self.request.id



    tmp_path = None



    



    try:



        # Try streaming directly from URL first (fastest option)

        publish_update(job_id, "processing", detail="Analyzing video from stream...")

        

        try:

            # Attempt direct URL streaming

            metrics = video_analyzer.analyze_video(video_url)

            

        except Exception as stream_error:

            # Fallback: Download if streaming fails

            logger.warning(f"Streaming failed: {stream_error}. Falling back to download.")

            publish_update(job_id, "processing", detail="Downloading R6 Replay...")

            

            r = requests.get(video_url, stream=True)

            f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            tmp_path = f.name

            for chunk in r.iter_content(8192): f.write(chunk)

            f.close()

            

            publish_update(job_id, "processing", detail="Scanning for Operators & Objectives...")

            metrics = video_analyzer.analyze_video(tmp_path)







        # 3. LLM Analysis



        publish_update(job_id, "processing", detail="Analyzing Tactical Performance...")



        llm_out = llm_analyzer.generate_insights(metrics)







        # 4. Format



        # --- FIX HERE: Call the correct function ---



        final_json = format_final_json(video_url, metrics, llm_out)



        



        publish_update(job_id, "complete", result=final_json)



        return final_json







    except Exception as e:



        publish_update(job_id, "error", error=str(e))



        raise e



    finally:



        if tmp_path and os.path.exists(tmp_path):



=======
import os



import requests



import tempfile



import logging



from celery import Celery



from redis_utils import publish_update 







# Import your modules



import video_analyzer



import llm_analyzer







# --- FIX HERE: Import the correct function name ---



from utils import format_final_json







# Define Queue



celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')



celery_app.conf.update(



    task_serializer='json', result_serializer='json', 



    accept_content=['json'], task_default_queue='r6_queue'



)







@celery_app.task(bind=True)



def run_analysis_task(self, video_url):



    job_id = self.request.id



    tmp_path = None



    



    try:



        # Try streaming directly from URL first (fastest option)

        publish_update(job_id, "processing", detail="Analyzing video from stream...")

        

        try:

            # Attempt direct URL streaming

            metrics = video_analyzer.analyze_video(video_url)

            

        except Exception as stream_error:

            # Fallback: Download if streaming fails

            logger.warning(f"Streaming failed: {stream_error}. Falling back to download.")

            publish_update(job_id, "processing", detail="Downloading R6 Replay...")

            

            r = requests.get(video_url, stream=True)

            f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            tmp_path = f.name

            for chunk in r.iter_content(8192): f.write(chunk)

            f.close()

            

            publish_update(job_id, "processing", detail="Scanning for Operators & Objectives...")

            metrics = video_analyzer.analyze_video(tmp_path)







        # 3. LLM Analysis



        publish_update(job_id, "processing", detail="Analyzing Tactical Performance...")



        llm_out = llm_analyzer.generate_insights(metrics)







        # 4. Format



        # --- FIX HERE: Call the correct function ---



        final_json = format_final_json(video_url, metrics, llm_out)



        



        publish_update(job_id, "complete", result=final_json)



        return final_json







    except Exception as e:



        publish_update(job_id, "error", error=str(e))



        raise e



    finally:



        if tmp_path and os.path.exists(tmp_path):



>>>>>>> 1208943 (add)
            os.remove(tmp_path)