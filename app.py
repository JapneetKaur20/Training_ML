#importing libraries
from flask import Flask,render_template,request
import joblib

app = Flask(__name__)

# specifying the route to homepage
@app.route("/")
def homepage():
    return render_template('homepage.html')

# specifying route to predict page
@app.route('/predict', methods = ['POST'])
def predict():
    # loading the save model and vectorizer using joblib
    bnb = joblib.load('Model.pkl')
    tf = joblib.load('Vector.pkl')

    # vectorizing message and predicting it
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect =  tf.transform(data).toarray()
        my_pred = bnb.predict(vect)
    
    return render_template("predict.html", prediction = my_pred)

if __name__ == '__main__':
	app.run(debug=True)
