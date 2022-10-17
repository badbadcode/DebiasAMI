'''
test
'''
from my_utils import *
from utils.config import Config
import joblib
import pickle

def test(model_name):

    config = Config(model_name)
    vectorizer = pickle.load(open(config.feature_path, "rb"))
    x_test, y_test, x_test_fair, y_test_fair = load_test()
    vec_test,vec_test_fair = get_vec_test(vectorizer,x_test,x_test_fair)
    model = joblib.load(filename=config.saved_model_path)
    y_test_pred_word = model.predict(vec_test)
    y_test_fair_pred_word = model.predict(vec_test_fair)
    print("test",accuracy_score(y_test, y_test_pred_word))
    print("fair test",accuracy_score(y_test_fair, y_test_fair_pred_word))

test("LR")