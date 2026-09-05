import face_recognition # Main library. Allows us to map facial images.
import cv2 # Allows us to manipulate images and use a camera for input
import numpy as np # Used for array manipulation
import os # Used to help with path linking
from datetime import datetime # Gives us the current time when someone arrives.

################################################################################

# Here we create the path to the images of our 'known faces'.
# We then init our arrays to load data from the images into.

path = "imagesBasic"
images = [] # Our images will be appended to this array for future use.
classNames = [] # The names (fileNames) will be appended here for future use.
myList = os.listdir(path) # os.listdir creates the link to our images.

################################################################################

# We want to load in each name and image to our arrays so we have the data we
# are searching for.

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') # Identifying the current image to load
    images.append(curImg) # We add the current image to our array.
    # As the stem is our person name, we split the filename and append the name
    # to the end of the list.
    classNames.append(os.path.splitext(cl)[0])

################################################################################

# We want to record the facial encodings of all the images. We use this later to
# compare the encoding of the known people to the current detected face.

def findEncodings(images):
    # We want to store a list of encodings of all of our known people.
    encodeList = [] 
    # We want to iterate through our list of images.
    for img in images:
        # For each image, we want to convert to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # We then want to encode the face and store the data points to encode.
        encode = face_recognition.face_encodings(img)[0]
        # We want to append encode to the array so we can use to compare later.
        encodeList.append(encode)
    # We return the encodedList to use later.
    return encodeList

################################################################################

# We want to update our attendance.csv to register the attendance of a person
# in the list of known people. We update the attendance list by writing the
# persons name with the time that person was registered.

def markAttendance(name):
    # We open the attendance.csv as f in read and write mode.
    with open('attendance.csv', 'r+') as f:
        # We create read the csv into myDataList
        myDataList = f.readlines()
        # We want to read the current registered names into NameList
        nameList = []
        for line in myDataList:
            # We parse each line so we can compare the names in .csv to nameList
            entry = line.split(',')
            # We then append the name into the nameList.
            nameList.append(entry[0])
        # If the is not already registered, we want to add it to the registered
        # list with the name and time.
        if name not in nameList:
            # We get the current time at scanning.
            now = datetime.now()
            # We convert the current time into a string.
            dtString = now.strftime('%H:%M%S')
            # We write the name and time back into attendance.csv
            f.writelines(f'\n{name},{dtString}')
        
################################################################################

# We call findEncodings() to get our encodings of the current known people.
encodeListKnown = findEncodings(images)

# We set and start out capture source. (1 is my laptop's camera)
cap = cv2.VideoCapture(1)

################################################################################

# Until we are done, we want to compare the current known persons encodings to 
# any face in view of a camera. To end this loop, use ctrl + c.

while True: 
    # We read our capture source andunpack cap.read to get the status and 
    # a copy of the image.
    success, img = cap.read()
    # We update the image size to 1/4 of the size to decrease processing times.
    imgSmall = cv2.resize(img, (0,0),None, 0.25, 0.25)
    # We update the current capture's color to RGB
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    # We want to find the current face locations in the current image, and then
    # we then want to get the encodings of the faces in the current image.
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)
    
    # We then want to compare the encodings from the current frame to the 
    # the encodings of our current known people.
    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, 
                                                      encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)
        
        # If we have a match, we want to display the capture, rectangle and 
        # the name. Finally, we want to call markAttendance() so we can 
        # update the attendance.csv.
        if matches[matchIndex]:
            # We call the name from the index of our matched person
            name = classNames[matchIndex]
            y1,x2,y2,x1 = faceLoc 
            # * 4 as we set to 0.25 earier. Ensures the rectangle covers the
            # entire detected face.
            y1,x2,y2,x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # We create the outside rectangle, and an inside rectangle. This
            # is so we have a nicer border area to display the name.
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), 
                          cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 3)
            
            markAttendance(name)
            
            # We show our webcam output.
            cv2.imshow('Webcam', img)
            cv2.waitKey(1)
            