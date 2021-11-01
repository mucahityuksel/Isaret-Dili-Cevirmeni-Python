import flask
from flask import request, jsonify, Response
import cv2
import numpy as np
import math
import itertools
import multiprocessing



app = flask.Flask(__name__)
app.config["DEBUG"] = True



@app.route("/test", methods=["GET"])
def test():
    f = open("kelimeler.txt", "r+")
    a = f.readline()
    return ''.join(x[0] for x in itertools.groupby(a))


@app.route("/video", methods=["POST"])
def videoUpload():
    print(request.files["file"])
    file = request.files["file"]
    file.save("dosya.mp4")
    import bitirme
    for a in ("bitirme"):
        p = multiprocessing.Process(target=lambda: __import__(a))
        p.start()
    return request("video upload")



app.run()
