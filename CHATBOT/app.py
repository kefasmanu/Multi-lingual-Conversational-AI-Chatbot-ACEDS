from flask import Flask, render_template, request, jsonify
from flask_cors import CORS #allowing cross ressource sharing
from flask_ngrok import run_with_ngrok

from chat import process_message
app = Flask(__name__)
#CORS(app)
run_with_ngrok(app)


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    lang = request.get_json().get("lang")
    
    #print(lang)
    #print(text)
    #TODO: Check if the text is valid
    #response = get_response(text)
    response=process_message(text, lang)

    # response=""
    # if lang == "fr":
    #     response = get_response_fr(text)
    # else:
    #      response = get_response_eng(text)

    message = {"answer": response}
    return jsonify(message)


#Starting the app
if __name__ =="__main__":
   #app.run(debug=True) 
   app.run() 

#ngrok config add-authtoken 2J3NC1RchsN8uuW4oEYrD1daHsJ_3YoL3Fp7SK3R8oN8dUCqZ
#ngrok config add-authtoken 2J3NC1RchsN8uuW4oEYrD1daHsJ_3YoL3Fp7SK3R8oN8dUCqZ
