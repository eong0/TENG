from datatime import timedelta
import cv2
import numpy as np
import os

SAVING_FRAMES_PER_SECOND = 10
# 1초당 10 frames 를 저장한다는 뜻!

def format_timedelta(td):
  '''timedelta 객체를 간지나게 사용하기 위한 함수'''
  result = str(id)
  try :
    result, ms = results.split(",")
  except ValueError : #예외가 발생할 때 실행하는 코드
     return (result + ".00").replace(":","-")
     ms = int(ms)
     ms = round(ms/1e4)
     return f"{result}.{ms:02}".replace(":","-")
    
def get_saving_frames_durations(cap,saving_fps):
  ''' 
