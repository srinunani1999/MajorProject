from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename 
from views.audio import start_recording, stop_recording
from views.nltk import predict_text
from views.sentiment_svc import train_model
from views.sentiment_nb import print_stop

app = Flask(__name__)

# @app.route('/store-audio/', methods= ['POST', 'GET'])
# def store_audio():
#     if request.method == 'POST':
#         audio = request.files['audiofile']
#         audio.save('./views/' + secure_filename(audio.filename))
#         audio_file_to_text(secure_filename(audio.filename))
#         return "upload successfull"
#     else:
#         return redirect(url_for('index'))

@app.route('/predict', methods= ["POST", "GET"])
def prediction():
    if request.method == "POST":
        text = request.form['spoken_text']
        data, result = predict_text(text)
        print(data)
        print(result)
        return render_template('predict.html', data=data, result=result)
    pass

@app.route('/recording', methods= ["POST", "GET"])
def recording():
    if request.method == 'POST':
        name = request.form['record_state']
        if name == "Start_Recording":
            start_recording()
            return render_template('index.html', button_name="Stop_Recording")
        elif name == "Stop_Recording":
            text = stop_recording()
            return render_template('result.html', audio_text=text)
    return redirect(url_for('index'))

@app.route('/feedback')
def index():
    return render_template('index.html', button_name="Start_Recording")
 
if __name__ == "__main__":
    # train_model()
    print_stop()
    app.run() 
   