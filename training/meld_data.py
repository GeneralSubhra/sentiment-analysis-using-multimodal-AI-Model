from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import AutoTokenizer

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
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vdo_fname = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""
        path = os.path.join(self.vdo_dir,vdo_fname)
        vdo_path = os.path.exists(path)
        
        if vdo_path==False:
            raise FileNotFoundError(f"No video found for file name: {path}")
        print("file found")
        
if __name__ == "__main__":
    meld = MELDDataset('D:\sentiment aanalysis\dataset\dev\dev_sent_emo.csv',
                       'D:\sentiment aanalysis\dataset\dev\dev_splits_complete')
    print(meld[1])                                                                         