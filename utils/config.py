from os.path import dirname, realpath, join
from pathlib import Path


class Config:

    #这些变量是在所有实例之间共享的变量，类变量
    #data


    # data_dir = "../data/AMI EVALITA 2018"
    # train_fp = r"data/AMI EVALITA 2018/en_training_anon.tsv"
    # test_fp = r"data/AMI EVALITA 2018/en_testing_labeled_anon.tsv"
    # test_fair_fp = r"data/unitended bias in AMI/synthetic_test_set.tsv"

    # cleaned_train_fp = r"data/AMI EVALITA 2018/my_train.tsv"
    # cleaned_test_fp = r"data/AMI EVALITA 2018/my_test.tsv"
    # cleaned_test_fair_fp = r"data/AMI EVALITA 2018/my_test_fair.tsv"

    DATA_DIC = {"AMI": {"train": r"data/AMI EVALITA 2018/my_train.tsv",
                        "test": r"data/AMI EVALITA 2018/my_test.tsv",
                        "unbiased": r"data/AMI EVALITA 2018/my_test_fair.tsv"}
                }

    HAS_LABELS_TEST = {"AMI": {"test": True, "unbiased": True}
                      }

    NUM_LABELS = {"AMI":2}


    VEC_DIR = r"data/AMI EVALITA 2018/vector"
    CKPT_DIR = ""


    res_path = "../res/res.csv" #所有模型所有数据集的结果
    # vectorizer_path = f"{data_dir}/tokenizer/" #只有特征学习的方法需要用到这里面的数据



    # def __init__(self, model_name=None, lr=None,seed=None, log_name=None): #创建了这个类的实例时就会调用该方法
    #      #self代表类的实例而非类，在定义方法的时候必须要有
    #
    #     # self.saved_model_path = f"saved_model/{model_name}_{self.vec_type}.pkl"
    #     # self.saved_topk_path = f"saved_model/{model_name}_{self.vec_type}_top_k.csv"
    #     self.lr = lr
    #     self.model_name = model_name
    #     self.seed = seed
    #     self.log_name = log_name



