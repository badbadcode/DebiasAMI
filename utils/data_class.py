

class Counterfactual:
    def __init__(self, df_train, df_test,df_unbiased, moniker):
        # display(df_train.head(1))
        self.moniker = moniker
        self.train = df_train
        self.test = df_test
        self.unbiased = df_unbiased

