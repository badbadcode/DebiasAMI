import joblib
from my_utils import *
from utils.config import Config
import pickle
import pandas as pd
model_name="LR"

def train(model_name):
    config = Config(model_name)
    x_train, x_dev, y_train, y_dev = load_split_train()
    vectorizer = fitted_vectorizer(vec_type=config.vec_type, x_train=x_train)
    with open(config.feature_path, 'wb') as fw:
        pickle.dump(vectorizer, fw)
    vec_train, vec_dev = get_vec_train(vectorizer, x_train, x_dev)
    model = getModel(model_name)
    model.fit(vec_train, y_train)
    top_k_words = TopK(model,20,vectorizer)
    top_k_df = pd.DataFrame({"top_k_words":top_k_words})
    top_k_df.to_csv(config.saved_topk_path, index=True, sep='\t', header=True)
    print("top_k_words:",top_k_words)
    joblib.dump(model, config.saved_model_path)
    y_dev_pred_word = model.predict(vec_dev)
    print("dev", accuracy_score(y_dev, y_dev_pred_word))


train("LR")



