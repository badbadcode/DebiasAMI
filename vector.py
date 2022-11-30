'''
将训练集、测试集、公平测试集的向量以不同的编码形式保存下来
'''
import torch
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from tqdm import tqdm
from utils.config import Config
from utils.funcs import check_dir
import pickle
def load_pkl(path):
    pickle_file = open(path,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data


def SaveOnehotVec(data_name):
    # cleaned_fp = cleaned_fp_dic["train"]
    # data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
    # x_train = data_train['cleaned_text'].values.astype('U')

    # data_name = "IMDB-S"
    ds = load_pkl(Config.DATA_DIC[data_name])
    x_train = list(ds.train['text'].values.astype('U'))

    vectorizer = CountVectorizer(binary=True,min_df=7)
    vectorizer.fit(x_train)
    vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/onehot_vectorizer.pkl"
    check_dir(vectorizer_path)
    with open(vectorizer_path, 'wb') as fw:
        pickle.dump(vectorizer, fw)

    for split in ["train", "test", "unbiased"]:

        # cleaned_fp = cleaned_fp_dic[split]
        # data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
        # x_train = data_train['cleaned_text'].values.astype('U')
        if split=="train":
            sentences = list(ds.train['text'].values.astype('U'))
        elif split=="test":
            sentences = list(ds.test['text'].values.astype('U'))
        elif split=="unbiased":
            if data_name == "AMI":
                sentences = list(ds.unbiased['text'].values.astype('U'))
            elif data_name == "IMDB-S":
                sentences = list(ds.test_ct['text'].values.astype('U'))
            else:
                sentences = list(ds.test['ct_text_amt'].values.astype('U'))

        vec_sens = vectorizer.transform(sentences)
        vec_sens = torch.tensor(np.asarray(vec_sens.todense()))
        vec_fp = f"{Config.DATA_DIR[data_name]}/vector/onehot/{split}.pt"
        check_dir(vec_fp)
        torch.save(vec_sens,vec_fp)


def SaveTfidfVec(data_name):
    # data_train = pd.read_csv(cleaned_fp_dic["train"], sep="\t", header=0)
    # x_train = data_train['cleaned_text'].values.astype('U')
    # data_name = "IMDB-S"
    ds = load_pkl(Config.DATA_DIC[data_name])
    x_train = list(ds.train['text'].values.astype('U'))

    vectorizer = TfidfVectorizer(
        max_features=800000,
        token_pattern=r"(?u)\b\w+\b",
        min_df=7,
        # max_df=0.1,
        analyzer='word',
        # ngram_range=(1, 5)
    )
    vectorizer.fit(x_train)
    vectorizer_path = f"{Config.DATA_DIR[data_name]}/tokenizer/tfidf_vectorizer.pkl"
    check_dir(vectorizer_path)
    with open(vectorizer_path, 'wb') as fw:
        pickle.dump(vectorizer, fw)

    for split in ["train","test","unbiased"]:
        # cleaned_fp = cleaned_fp_dic[split]
        # data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)
        # x_train = data_train['cleaned_text'].values.astype('U')
        if split=="train":
            sentences = list(ds.train['text'].values.astype('U'))
        elif split=="test":
            sentences = list(ds.test['text'].values.astype('U'))
        elif split=="unbiased":
            if data_name == "AMI":
                sentences = list(ds.unbiased['text'].values.astype('U'))
            elif data_name =="IMDB-S":
                sentences = list(ds.test_ct['text'].values.astype('U'))
            else:
                sentences = list(ds.test['ct_text_amt'].values.astype('U'))

        vec_sens = vectorizer.transform(sentences)
        vec_sens = torch.tensor(np.asarray(vec_sens.todense()))
        vec_fp = f"{Config.DATA_DIR[data_name]}/vector/tfidf/{split}.pt"
        check_dir(vec_fp)
        torch.save(vec_sens,vec_fp)


def SaveBertVec(split):

    cleaned_fp = cleaned_fp_dic[split]
    data_train = pd.read_csv(cleaned_fp, sep="\t", header=0)

    x_train = data_train['cleaned_text'].values.astype('U').tolist()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    bert = bert.to(device)

    vec_train = []

    for text in tqdm(x_train):
        encoded_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=80)
        input_ids = encoded_inputs["input_ids"]
        attention_masks = encoded_inputs["attention_mask"]
        input_ids = torch.tensor(input_ids).to(device)
        attention_masks = torch.tensor(attention_masks).to(device)

        output = bert(input_ids.unsqueeze(0),
                          token_type_ids=None,
                          attention_mask=attention_masks.unsqueeze(0), output_hidden_states=True)

        z0 = output.hidden_states[-1][:, 0, :]
        # print(z0.size()) #[1,768]
        # print(z0.squeeze(0).size())  #[768]
        vec_train.append(z0.squeeze(0).detach().cpu().numpy())

    vec_train = torch.tensor(vec_train)
    vec_fp = f"{Config.VEC_DIR}/bert/{split}.pt"
    check_dir(vec_fp)
    torch.save(vec_train,vec_fp)



# cleaned_fp_dic = {"train": Config.cleaned_train_fp,
#             "test": Config.cleaned_test_fp,
#            "fair": Config.cleaned_test_fair_fp}


data_name = "IMDB-L"
SaveOnehotVec(data_name) #3782,min_df=5:1168,6:1001,7:882
SaveTfidfVec(data_name) #3808,min_df=5:1186,6:1018


# for split in ["train","test","fair"]:
#     SaveBertVec(split) #768

# ty = "onehot"
# train = torch.load(f"{Config.VEC_DIR}/{ty}/train.pt")
# test = torch.load(f"{Config.VEC_DIR}/{ty}/test.pt")
# fair = torch.load( f"{Config.vec_dir}/{ty}/fair.pt")





