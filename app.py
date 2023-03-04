import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import json
import numpy as np
import pandas as pd

# Starting app 
app=Flask(__name__)
# Loading the Model
logmodel=pickle.load(open('logmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    new_data=pd.DataFrame(data=new_data,index=None,columns=None)
    output=logmodel.predict(new_data)
    # json dumps converts the python object to json object (it cannot convert int to json object tahts why we have converted it into the string)
    ans=json.dumps(str(output[0]))
    print(ans)
    return jsonify(ans)

if __name__=='__main__':
    app.run(debug=True)
