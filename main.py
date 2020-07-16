from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        th = int(myDict['th'])
        animl = int(myDict['animl'])
        maf = int(myDict['maf'])
        sneez = int(myDict['sneez'])
        cough = int(myDict['cough'])
        go = int(myDict['go'])
        inputFeatures = [fever, age, pain, runnyNose, diffBreath, th, animl, maf, sneez, cough, go]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template("show.html", inf=round(infProb*100))
    return render_template("index.html")
    
    

if __name__ == "__main__":
    app.run(debug=True)