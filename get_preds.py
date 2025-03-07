import sys
sys.path.append("agsr_net") 
import torch
import numpy as np
import pandas as pd
from agsr_net.layers import *
from agsr_net.model import AGSRNet  
from agsr_net.preprocessing import unpad
from MatrixVectorizer import MatrixVectorizer


lr_dim, hr_dim = 160, 268
class Args:
    def __init__(self):
        self.epochs = 200  
        self.lr = 0.0001   
        self.lmbda = 0.1 
        self.lr_dim = 160     
        self.hr_dim = 320  
        self.hidden_dim = 320    
        self.padding = 26 
        self.mean_dense = 0.0    
        self.std_dense = 0.01      
        self.mean_gaussian = 0.0  
        self.std_gaussian = 0.1

ks = [0.9, 0.7, 0.6, 0.5]
args = Args() 

model = AGSRNet(ks, args)

try:
    state_dict = torch.load('best_model.pt')
    model.load_state_dict(state_dict)
    model.eval()  
    print("Best model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

LR_TEST_DATA_PATH = "data/lr_test.csv"

df_lr_test = pd.read_csv(LR_TEST_DATA_PATH)

v_lr_test = np.zeros((len(df_lr_test), 160, 160))

mv = MatrixVectorizer()


for i, row in enumerate(df_lr_test.values):
    v_lr_test[i] = mv.anti_vectorize(row, 160)


preds_list = []
print("Generating predictions...")
with torch.no_grad(): 
    for i, lr in enumerate(v_lr_test):
        all_zeros_lr = not np.any(lr)
        if not all_zeros_lr:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            
            final_preds, _, _, _ = model(lr, lr_dim, hr_dim)
            final_preds = np.array(final_preds)
            
            numpy_preds = final_preds.detach().numpy()
            unpadded_preds = unpad(numpy_preds, args.padding)
            preds_list.append(unpadded_preds)
            

        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(v_lr_test)} samples")

preds_list = np.array(preds_list)
melted_preds = preds_list.flatten()

print(f"Total predictions: {len(melted_preds)}")

submission_df = pd.DataFrame({
    'ID': range(1, len(melted_preds) + 1),
    'Predicted': melted_preds
})

submission_df.to_csv('submission.csv', index=False)
print(f"Submission saved with {len(submission_df)} entries")