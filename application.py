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
            CreditScore=int(request.form.get('CreditScore')),
            Age = int(request.form.get('Age')),
            Tenure = int(request.form.get('Tenure')),
            Balance = float(request.form.get('Balance')),
            NumOfProducts = int(request.form.get('NumOfProducts')),
            HasCrCard = int(request.form.get('HasCrCard')),
            IsActiveMember = int(request.form.get('IsActiveMember')),
            EstimatedSalary = int(request.form.get('EstimatedSalary')),
            Geography = request.form.get('Geography'),
            Gender= request.form.get('Gender'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        if pred==1:
            pred_f='churn  '
        elif pred==0:
            pred_f='not churn '
        else:
            pred_f='error'

        results=pred_f

        return render_template('result.html',final_result=results)

if __name__=='__main__':
    app.run(host='0.0.0.0')