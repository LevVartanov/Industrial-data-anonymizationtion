import sklearn.preprocessing
import tabgan as tg
import numpy as np
import sklearn
from _ctgan.synthesizer import _CTGANSynthesizer as CTGAN


class Anonymizer():
    def __init__(self, model, model_params) -> None:
        self.model = model
        self.model_params = model_params
    
    def fit(X, y, epochs=100):
        pass


class TabularAnonymizer(Anonymizer):
    def __init__(self, encoder, encoder_params, model='tabgan', model_params={'gen_x_times': 1.1,
            'cat_cols': None,
            'bot_filter_quantile': 0.001,
            'top_filter_quantile': 0.999,
            'is_post_process':  True,
            'use_adversarial': True,
            'adversarial_model_params':{
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 150,
                "learning_rate": 0.02,
                "random_state": 42,
            },
            'pregeneration_frac': 2,
            'only_generated_data': False,
            'gen_params': {"batch_size": 50, 'patience': 25, "epochs": 50, "llm": "distilgpt2"}}) -> None:
        super().__init__(model, model_params)
        self.encoder = encoder
        self.encoder_params = encoder_params
        if self.model == 'tabgan':
            self.generator = CTGAN(batch_size=self.model_params['gen_params']["batch_size"], patience=self.model_params['gen_params']["patience"])

    def encode(self, X):
        if self.encoder == 'ordinal':
            enc = sklearn.preprocessing.OrdinalEncoder(categories=self.encoder_params['categories']).set_output(transform="pandas")  
            return enc.fit_transform(X[self.encoder_params['features']]) 

    def fit(self, X, y):
        X_ = X.copy()
        # y_ = y.copy()
        if self.model == 'tabgan':
            if y is not None:
                X_["TEMP_TARGET"] = y
            # print(X_)
            self.generator.fit(X_, [], epochs=self.model_params['gen_params']["epochs"])
            self.df_dtypes = X_.dtypes.values
            # else:
            #     ctgan.fit(train_df, self.cat_cols, epochs=self.gen_params["epochs"])

    def generate(self):
        if self.model == 'tabgan':
            generated_df = self.generator.sample(50)
            # print(generated_df.shape)
            for i in range(len(generated_df.columns)):
                generated_df[generated_df.columns[i]] = generated_df[
                    generated_df.columns[i]
                ].astype(self.df_dtypes[i])
            X = generated_df.drop('TEMP_TARGET', axis=1)
            y = generated_df['TEMP_TARGET']
            if self.model_params['is_post_process']:
                X, y = self.postprocess_data(X, y, X)
            if self.model_params['use_adversarial']:
                # print('ququ')
                return self.adversarial_filtering(X.copy(), y, X.copy())
            else:
                return X, y
    
    def adversarial_filtering(self, train_df, target, test_df):
        # if test_df is None:
        #     logging.info("Skipping adversarial filtering, because test_df is None.")
        #     return train_df, target
        ad_model = tg.adversarial_model.AdversarialModel(
            model_params=self.model_params['adversarial_model_params']
        )
        # self._validate_data(train_df, target, test_df)
        train_df['TEMP_TARGET'] = target
        ad_model.adversarial_test(test_df, train_df.drop('TEMP_TARGET', axis=1))

        train_df["test_similarity"] = ad_model.trained_model.predict(
            train_df.drop('TEMP_TARGET', axis=1)
        )
        train_df.sort_values("test_similarity", ascending=False, inplace=True)
        # train_df = train_df.head(self.get_generated_shape(train_df) * train_df.shape[0])
        del ad_model

        return (
            train_df.drop(["test_similarity", 'TEMP_TARGET'], axis=1).reset_index(
                drop=True
            ),
            train_df['TEMP_TARGET'].reset_index(drop=True),
        )
    
    def postprocess_data(self, train_df, target, test_df):
        # if not self.is_post_process or test_df is None:
        #     # logging.info("Skipping postprocessing")
        #     return train_df, target

        # self._validate_data(train_df, target, test_df)
        train_df['TEMP_TARGET'] = target

        # Filter numerical columns
        for col in test_df.columns:
            # if self.cat_cols is None or col not in self.cat_cols:
                min_val = test_df[col].quantile(self.model_params['bot_filter_quantile'])
                max_val = test_df[col].quantile(self.model_params['top_filter_quantile'])
                train_df = train_df[(train_df[col].isna()) | ((train_df[col] >= min_val) & (train_df[col] <= max_val))]

                # if train_df.shape[0] < 10:
                #     raise ValueError(f"Too few samples (<10) after filtering column {col}. "
                #                      f"Test data may be skewed. Filter range: [{min_val}, {max_val}]")

        # Filter categorical columns
        # if self.cat_cols:
        #     for col in self.cat_cols:
        #         train_df = train_df[train_df[col].isin(test_df[col].unique())]
        #         if train_df.shape[0] < 10:
        #             raise ValueError(f"Too few samples (<10) after filtering categorical column {col}")

        # logging.info(
        #     f"Generated shapes after postprocessing: {train_df.drop(self.TEMP_TARGET, axis=1).shape} plus target")

        result_df = train_df.reset_index(drop=True)
        return (
            result_df.drop('TEMP_TARGET', axis=1),
            result_df['TEMP_TARGET']
        )


class EconomicTabularAnonymizer(TabularAnonymizer):
    def __init__(self, encoder='ordinal', encoder_params={'categories': [np.array(['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Preschool', 'HS-grad', 'Prof-school', 'Some-college', 'Bachelors',
        'Masters', 'Doctorate', 'Assoc-voc',
        'Assoc-acdm'],
       dtype=object),
 np.array(['Never-married', 'Widowed', 'Divorced', 'Separated',
        'Married-spouse-absent', 'Married-AF-spouse', 'Married-civ-spouse'],
       dtype=object),
 np.array(['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair',
        'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
        'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
        'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support',
        'Transport-moving'], dtype=object),
 np.array(['Not-in-family','Unmarried',  'Other-relative',
        'Wife', 'Husband', 'Own-child'], dtype=object),
 np.array(['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other',
        'White'], dtype=object),
 np.array(['Female', 'Male'], dtype=object),
 np.array(['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
        'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
        'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',
        'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',
        'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
        'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',
        'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',
        'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',
        'Yugoslavia'], dtype=object)], 'features': ['education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']}, model='tabgan', model_params={'gen_x_times': 1.1,
            'cat_cols': None,
            'bot_filter_quantile': 0.001,
            'top_filter_quantile': 0.999,
            'is_post_process':  True,
            'use_adversarial': True,
            'adversarial_model_params':{
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 150,
                "learning_rate": 0.02,
                "random_state": 42,
            },
            'pregeneration_frac': 2,
            'only_generated_data': False,
            'gen_params': {"batch_size": 50, 'patience': 25, "epochs": 50, "llm": "distilgpt2"}}) -> None:
        super().__init__(encoder, encoder_params, model, model_params)
    
