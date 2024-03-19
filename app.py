from flask import Flask, render_template, request
from model import run_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/disease_diagnosis', methods=['GET', 'POST'])
def disease_diagnosis():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        symptoms_list = symptoms.split('\n')

        diseases = run_model(symptoms_list)

        return render_template('disease_diagnosis.html', diseases=diseases)
    else:
        return render_template('disease_diagnosis.html')

if __name__ == '__main__':
    app.run(debug=True)