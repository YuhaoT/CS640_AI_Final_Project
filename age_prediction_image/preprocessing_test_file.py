from imutils import paths
import face_recognition
import cv2
import pandas as pd

# Load the image path
image_folder = "../resource/filtered_pics"
label_location = "../resource/labeled_users.csv"
target_folder = "../resource/cropped_profile"
image_paths = list(paths.list_images(image_folder))
face_detection_model_type = "CNN" # either "CNN" or "HOG"

label_csv = pd.read_csv(label_location)

for (i, imagePath) in enumerate(image_paths):
	# The file_name here is also the id of the owner of the profile, when output the resized and cropped ones, also
	# need to add the labels to its name
	file_name = imagePath.split("/")[-1]
	# read the age and race labels out
	labels = label_csv.loc[label_csv['user_id'] == int(file_name.split('.')[0])]
	if labels.shape[0] == 0:
		# The owner of the profile doesn't exist
		continue
	if pd.isnull(labels.iloc[0]['race']):
		continue
	race = int(labels.iloc[0]['race'])
	if pd.isnull(labels.iloc[0]['year_born']):
		continue
	age = int(2021 - int(labels.iloc[0]['year_born']))

	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image.
	boxes = face_recognition.face_locations(rgb,
				model = 'CNN')
	if len(boxes) != 1:
		if len(boxes) == 0:
			continue
		# In our dataset, only one box is expected, so if there are more than one face, we use the largest one
		largest_index = -1
		largest_area = -1
		for (j, (top, right, bottom, left)) in enumerate(boxes):
			area = (right - left) * (bottom - top)
			if area > largest_area:
				largest_area = area
				largest_index = j
		top, right, bottom, left = boxes[largest_index]
	else:
		top, right, bottom, left = boxes[0]

	croppedImg = image[top: bottom+1, left: right + 1]
	croppedImg_stretched = cv2.resize(croppedImg, (200, 200))
	cv2.imwrite(target_folder + "/" + str(age) + "_" + str(race) + "_" + file_name, croppedImg_stretched)
	# compute the facial embedding for the face
