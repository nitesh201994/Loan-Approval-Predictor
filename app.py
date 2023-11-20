from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application




@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            no_of_dependents=int(request.form.get('no_of_dependents')),
            income_annum = float(request.form.get('income_annum')),
            loan_amount = int(request.form.get('loan_amount')),
            loan_term = int(request.form.get('loan_term')),
            cibil_score = int(request.form.get('cibil_score')),
            residential_assets_value = float(request.form.get('residential_assets_value')),
            commercial_assets_value = int(request.form.get('commercial_assets_value')),
            luxury_assets_value = int(request.form.get('luxury_assets_value')),
            bank_asset_value = int(request.form.get('bank_asset_value')),
            education= request.form.get('education'),
            self_employed= request.form.get('self_employed'),
        )

       

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        
        if pred==1:
            pred_f='Loan Application : Approved '
        elif pred==0:
            pred_f='Loan Application : Rejected '
        else:
            pred_f='error'

        return render_template('result.html',final_result=results)

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)