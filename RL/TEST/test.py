import pandas as pd

default_df = pd.read_csv('./data/default_result.csv')
sft_df = pd.read_csv('./data/sft_result.csv')

default_mean = default_df['EleutherAI/polyglot-ko-1.3b_bert_score_F1'].mean()
sft_mean = sft_df['bradmin/sft_bert_score_F1'].mean()

print(default_mean)
print(sft_mean)


# import torch
#
# print("Torch version:{}".format(torch.__version__))
# print("cuda version: {}".format(torch.version.cuda))
# print("cudnn version:{}".format(torch.backends.cudnn.version()))