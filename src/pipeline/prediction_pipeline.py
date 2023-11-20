import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 no_of_dependents:int,
                 income_annum:float,
                 loan_amount:float,
                 loan_term:int,
                 cibil_score:int,
                 residential_assets_value:float,
                 commercial_assets_value:float,
                 luxury_assets_value:float,
                 bank_asset_value:float,
                 education:str,
                 self_employed:str):
        
        self.no_of_dependents=no_of_dependents
        self.income_annum=income_annum
        self.loan_amount=loan_amount
        self.loan_term=loan_term
        self.cibil_score=cibil_score
        self.residential_assets_value=residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value
        self.education = education
        self.self_employed = self_employed

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'no_of_dependents':[self.no_of_dependents],
                'income_annum':[self.income_annum],
                'loan_amount':[self.loan_amount],
                'loan_term':[self.loan_term],
                'cibil_score':[self.cibil_score],
                'residential_assets_value':[self.residential_assets_value],
                'commercial_assets_value':[self.commercial_assets_value],
                'luxury_assets_value':[self.luxury_assets_value],
                'bank_asset_value':[self.bank_asset_value],
                'education':[self.education],
                'self_employed':[self.self_employed]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)