from flask import Flask, render_template, request, Response
from sklearn.externals import joblib
import os

app = Flask(__name__)
my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir, 'model.pkl')

model = joblib.load(pickle_file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yakpredict')
def yakpredict():
    return render_template('predict.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/predict/', methods=['POST'])
def predict():
    yak_string = request.form['theyak']
    score = model.predict([yak_string])
    return Response(str(score[0]))

if __name__ == '__main__':
    app.run(debug=True)
