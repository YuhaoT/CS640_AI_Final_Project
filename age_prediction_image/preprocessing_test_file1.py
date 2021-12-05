from imutils import paths
import face_recognition
import cv2
import os
import pandas as pd

# Load the image path
original_image_folder = "../resource/profile1/profile_pics/"
label_location = "../resource/profile1/labeled_users.csv"
cropped_image_folder = "../resource/profile1/cropped_profile1/"
face_detection_model_type = "CNN" # either "CNN" or "HOG"

label_csv = pd.read_csv(label_location, lineterminator="\n")

# we need to iterate all user info in the label file, since some users may not have the profile image.
cnt1 = 0
cnt2 = 0
cnt3 = 0
for index, row in label_csv.iterrows():
	cnt3 += 1
	try:
		picture_id = int(row[0])
	except ValueError:
		cnt1 += 1
		continue
	if pd.isnull(row["human.labeled.age"]):
		cnt2 += 1
		continue
	user_age = int(row["human.labeled.age"])
	image_path = original_image_folder + str(picture_id) + ".jpeg"
	if not os.path.isfile(image_path):
		continue
	image = cv2.imread(image_path)
	#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image.
	boxes = face_recognition.face_locations(image, model="CNN")
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
	cv2.imwrite(cropped_image_folder + str(user_age) + "_" + str(picture_id) + ".jpeg", croppedImg_stretched)