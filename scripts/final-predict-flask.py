import csv
import pandas as pd
import numpy as np
import json
from flask import Flask, jsonify, render_template
app = Flask(__name__)


@app.route("/")
def default():

	df = pd.read_csv("../data/predicted-coords.csv", header=None)

	lat = df[0][0]
	lng = df[0][1]
	data = [[lat, lng]]

	return render_template('final-predict-view.html', data=data)



if __name__ == '__main__':
	app.run(debug=True)