# Example script for running code in image_classifier
from image_classifier import PowderClassifier

working_dir_path = "C:/Users/sam.cross/PycharmProjects/image_recognition/"
test_images_folder = working_dir_path + "test_images/"
json_path = working_dir_path + "classification_results.json"

classifier = PowderClassifier()
results = classifier.classify_images(test_images_folder, savepath=json_path)
print(results)