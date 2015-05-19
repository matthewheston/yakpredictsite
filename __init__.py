from flask import Flask, render_template, request, Response
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    yak_string = request.form['theyak']
    score = model.predict([yak_string])
    return Response(str(score[0]))

if __name__ == '__main__':
    app.run(debug=True)
