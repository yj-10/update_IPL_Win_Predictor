from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sklearn
import pickle

# importing model
model = pickle.load(open('model_pipe.pkl','rb'))


# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    try:
        batting_team = request.form.get('Batting_Team')
        bowling_team = request.form.get('Bowling_Team')
        selected_city = request.form.get('City')
        target = request.form['Target']
        score = request.form['Score']
        overs = request.form['Over_Completed']
        wickets = request.form['Wickets_Out']

        # print(batting_team)

        runs_left = int(target) - int(score)
        balls_left = 120 - (int(overs)*6)
        wickets = 10 - int(wickets)
        crr = int(score)/int(overs)
        rrr = (runs_left*6)/balls_left

        input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        result1 = batting_team + "- " + str(round(win*100)) + "%"
        result2 = bowling_team + "- " + str(round(loss*100)) + "%"

        return render_template('index.html',result1 = result1, result2 =result2)
    except Exception as e:
        print("Exception Occured",e)

# python main
if __name__ == "__main__":
    app.run(debug=True)