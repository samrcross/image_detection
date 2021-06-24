import tensorflow.keras
from PIL import Image, ImageOps
from detecto import core
import numpy as np
import os
import json

class PowderClassifier:

    def __init__(self, working_dir_path = "C:/Users/sam.cross/PycharmProjects/image_recognition/"):

        self.working_dir_path = working_dir_path
        self.model_path = self.working_dir_path + "keras_model.h5"
        self.cdmodel_path = self.working_dir_path + "model.pth"
        self.training_image_folder = self.working_dir_path + "training/images/"
        self.training_labels =  self.working_dir_path + "training/labels.json"  # File containing labels for training

    # Uses information stored in training/images/ and training/labels.json to train the model
    # After training, updates model files in self.model_path and self.cdmodel_path
    def train_model(self):

        #
        pass


    # Uses model to classify all images in folder and return results
    def classify_images(self, image_folder, savepath=""):

        # Dictionary to store results
        # Keys are strings containing sample_ID or filename
        # Values are classifications returned by algorithm
        # e.g: {"A0190-1.png": 2, "A0190-2.png": 3, ...}
        classification_results = {}

        # Loads trained models
        model = tensorflow.keras.models.load_model(self.model_path, compile=False)
        cdmodel = core.Model.load(self.cdmodel_path, ['sample'])

        print("Classifying images in %s" % image_folder)

        counter = 0
        for filename in os.listdir(image_folder):
            if (filename.endswith(".png")):
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

                # crucible detection
                img = Image.open(image_folder + filename)

                print("Analyzing image at", image_folder + filename)

                pred = cdmodel.predict(img)
                lbl, box, score = pred
                temp = box.numpy()[0]
                box = (round(temp[0]), round(temp[1]), round(temp[2]), round(temp[3]))
                image = img.crop(box)  # passing this to powder model

                # powder stuff
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)

                # display image (to user) that model took as input
                # image.show()
                # run image through the model
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                result = self.classify(prediction, filename)

                # Adds result to dictionary
                classification_results[filename] = result

                counter += 1
                print(counter)

        # If savepath is specified, saves to json
        if savepath:
            with open(savepath, 'w') as fw:
                json.dump(classification_results, fw)

        return classification_results


    # Converts prediction vector to classification
    def classify(self, prediction, filename):

        print("Classifying %s, prediction:" % filename, prediction)

        x = filename.split(".png")
        if (prediction[0][0] > 0.8):
            print(x[0], "volume close to ZERO")
            return 0

        elif (prediction[0][1] > 0.8):
            print(x[0], "volume is ONE")
            return 1

        elif (prediction[0][2] > 0.8):
            print(x[0], "volume is TWO")
            return 2

        elif (prediction[0][3] > 0.8):
            print(x[0], "volume is THREE")
            return 3

        elif (prediction[0][4] > 0.8):
            print(x[0], "volume is FOUR")
            return 4

        else:
            idx = np.where(prediction[0] == np.amax(prediction[0]))
            prediction[0][idx[0][0]] += 0.8
            print("Unclear but most likely ", end='')
            return self.classify(prediction, filename)