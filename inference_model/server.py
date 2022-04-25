import os

import sys

import base64

import numpy as np

import importlib

from paddle_serving_app import ImageReader

from multiprocessing import freeze_support

from paddle_serving_server.web_service import WebService

class ImageService(WebService):

def preprocess(self, feed={}, fetch=[]):

reader = ImageReader(image_shape=[3, 300, 300],

image_mean=[0.5, 0.5, 0.5],

image_std=[0.5, 0.5, 0.5])

feed_batch = []

for ins in feed:

if "image" not in ins:

raise ("feed data error!")

sample = base64.b64decode(ins["image"])

img = reader.process_image(sample)

feed_batch.append({"image": img})

return feed_batch, fetch

image_service = ImageService(name="image")

image_service.load_model_config("./serving_server/")

image_service.prepare_server(

workdir="./work", port=int(8090), device="cpu")

image_service.run_server()

image_service.run_flask()