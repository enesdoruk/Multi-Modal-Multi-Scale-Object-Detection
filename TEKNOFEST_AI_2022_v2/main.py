from turtle import st
import warnings
warnings.filterwarnings("ignore")

import time 
import logging
from datetime import datetime
from pathlib import Path

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
   
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("------------------------------------zekAi------------------------------------")
    print("zekAi AI Model Started...")
   
    team_name = "zekai"
    password = "gd6g78Vh"
    evaluation_server_url = "http://10.10.10.10/"


    configure_logger(team_name)
    
    detection_model = ObjectDetectionModel(evaluation_server_url)

    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    frames_json = server.get_frames()

    images_folder = "./images/"
    result_folder = './result/'
    Path(images_folder).mkdir(parents=True, exist_ok=True)
   
    for frame in frames_json:
        start = time.time()

        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])
        Path(images_folder + frame['video_name']).mkdir(parents=True, exist_ok=True)
        Path(result_folder + frame['video_name']).mkdir(parents=True, exist_ok=True)
        predictions = detection_model.process(predictions, evaluation_server_url, frame)
        result = server.send_prediction(predictions)

        stop = time.time()
        print("Total time per frame = ", stop - start)

        #if stop - start < 0.6:
        #    time.sleep(0.30)
    
    
if __name__ == '__main__':
    run()
