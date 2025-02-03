from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import AutoTokenizer
import cv2
import numpy as np
import torch

class MELDDataset(Dataset):
    def __init__(self,csv_path,vdo_dir):
        self.data = pd.read_csv(csv_path)
        self.vdo_dir = vdo_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_map = {
            'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6    
        }
        
        self.sentiment_map={
            'neagtive':0,
            'neutral':1,
            'positive':2
        }
    def __load_video_frames(self,vdo_path):
        cap = cv2.VideoCapture(vdo_path)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Unable to open video: {vdo_path}")
            
            ret,frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"video not found: {vdo_path}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            while len(frames)<30 and cap.isOpened():
                ret , frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame,(224,224))
                frame = frame / 255.0 
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"video error: {str(e)}")
        finally:
            cap.release()
        if(len(frames)==0):
            raise ValueError(f"no frames could be extracted")
        
        #pad or truncate frames
        if len(frames)<30:
            frames += [np.zeros_like(frames[0])] * (30-len(frames))
        else:
            frames= frames[:30]
       
        #Before permute: [frames,height,width,channels]
        #after permute: [frames,channels,height,width]
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vdo_fname = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""
        path = os.path.join(self.vdo_dir,vdo_fname)
        vdo_path_exists = os.path.exists(path)
        
        if vdo_path_exists==False:
            raise FileNotFoundError(f"No video found for file name: {path}")
        text_inputs = self.tokenizer(row['Utterance'],
                                     padding='max_length',
                                     truncation=True,
                                     max_length=128,
                                     return_tensors='pt')
        
        video_frames = self.__load_video_frames(path)
        print(video_frames)
        
        
if __name__ == "__main__":
    meld = MELDDataset('D:\sentiment aanalysis\dataset\dev\dev_sent_emo.csv',
                       'D:\sentiment aanalysis\dataset\dev\dev_splits_complete')
    print(meld[0])                                                                         