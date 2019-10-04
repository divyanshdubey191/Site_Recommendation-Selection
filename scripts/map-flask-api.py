import csv
import pandas as pd
import numpy as np
import json
from flask import Flask, jsonify, render_template
app = Flask(__name__)


@app.route("/")
def default():

	df = pd.read_csv("../data/google-maps-data.csv")

	lat = df['0'].values.tolist()
	lng = df['1'].values.tolist()
	label = df['2'].values.tolist()
	data = []
	for i in range(len(lat)):
		data.append([lat[i], lng[i], label[i]])

	return render_template('plot_suggested.html', data=map(json.dumps, data))



if __name__ == '__main__':
	app.run(debug=True)