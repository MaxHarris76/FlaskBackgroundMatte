from flask import Flask, render_template, request
from flask.templating import render_template_string





app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/submit", methods = ['POST'])
def submit():
    # HTML -> .py
    if request.method == "POST":
        name = request.form["username"]

    # .py -> HMTL
    return render_template("submit.html", n= name)

if __name__ =="__main__":
    app.run(debug=True)