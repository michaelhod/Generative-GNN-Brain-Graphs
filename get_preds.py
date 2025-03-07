import torch
import numpy as np
import pandas as pd
from agsr_net.model import AGSRNet  
from MatrixVectorizer import MatrixVectorizer

lr_dim, hr_dim = 160, 268

try:
    model = torch.load('best_model.pt')
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
            preds_list.append(final_preds.detach().numpy())
        
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