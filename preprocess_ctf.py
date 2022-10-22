import pickle
from utils.data_class import Counterfactual
'''
Robustness to Spurious Correlations in Text Classification via Automatically Generated Counterfactuals 
from: Zhao Wang and Aron Culotta
ct_text_all_causal: 通过论文所提方案（从 all vocabulary words）监测到的因果词生成的反事实样本
ct_text_amt：通过亚马逊众包生成的反事实样本
ct_text_bad: top words里除了因果词以外的部分
ct_text_causal:通过人为标注的因果词生成的反事实样本
ct_text_identified_causal: 通过论文所提方案（从 top words）监测到的因果词生成的反事实样本
ct_text_top:通过逻辑回归选出的单词生成的反事实样本
'''

def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data


# imdb_l = load_pkl(r"data/IMDB-L/ori/ds_imdb_para.pkl")
# imdb_s = load_pkl(r"data/IMDB-S/ori/ds_imdb_sent.pkl")
# kindle = load_pkl(r"data/KINDLE/ori/ds_kindle.pkl")
# ami_ds = load_pkl("/data/AMI/ori/ds_ami.pkl")

