import cv2
import os
import glob


def imread_img(path_image):

	return cv2.imread(path_image)

def label_image(path_to_data):

	list_label = os.listdir(path_to_data)

	label2id = {}
	for id_, value in enumerate(list_label):
		label2id[value] = id_

	data_train = []
	data_label = []

	for label in list_label:
		for img in glob.glob(path_to_data+"/" + label + '/*'):
			image = imread_img(img)
			label_image = label2id[label]

			data_train.append(image)
			data_label.append(label_image)

	return data_train, data_label, label2id

# data_train, data_label, label2id = label_image('data/')
#
# print(data_train[0])
# cv2.imwrite('test.jpg', data_train[0])
# print(data_label[0])
# print(label2id)




