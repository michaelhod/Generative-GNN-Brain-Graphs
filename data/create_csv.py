import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MatrixVectorizer import MatrixVectorizer
import pandas as pd
import numpy as np



class CreateCSV():
    def __init__(self, model: ):
        self.model = model
        self.mv = MatrixVectorizer()


    @staticmethod
    def generate_pd(self, lr, padding = False, padding_val=26):
        preds_list = []
        print("Generating predictions...")
        with torch.no_grad(): 
            for i, lr in enumerate(lr):
                all_zeros_lr = not np.any(lr)
                if not all_zeros_lr:
                    lr = torch.from_numpy(lr).type(torch.FloatTensor)
                    
                    final_preds, _, _, _ = self.model(lr, self.model., hr_dim)
                    numpy_preds = final_preds.detach().numpy()
                    if padding:
                        numpy_preds = unpad(numpy_preds, padding_val = 26)
                    preds_list.append(numpy_preds)
                    

                if (i+1) % 10 == 0:
                    print(f"Processed {i+1}/{len(v_lr_test)} samples")

        


        vectorized_preds = np.zeros((len(preds_list), 35778))
        for i, pred in enumerate(preds_list):
            vectorized_preds[i] = self.mv.vectorize(pred, include_diagonal=False)
            melted_preds = vectorized_preds.flatten()
        print(f"Total-predictions: {len(melted_preds)}")

        df = pd.DataFrame({
            'ID': range(1, len(melted_preds) + 1),
            'Predicted': melted_preds
        })

        return df
    
    @staticmethod
    def save_csv(self, df):
        df.to_csv('submission.csv', index=False)
        print(f"Submission saved with {len(df)} entries")
