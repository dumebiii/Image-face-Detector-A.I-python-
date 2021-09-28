import cv2

# load some trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontal_default.xml')

# Choose and import a photo we will be using for training our model
img = cv2.imread("elom.png")

# turn image to grayscale cause computer only understands images in white and black no colour
grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# try and detect image
face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

# # Shows co-ordinates of the image added in the code
print(face_coordinates)


# Drawing a rectangle from co-ordinates gotten from print('face_co-ordinates')
# And when want to write this , we write it this way
# (img(image_var), (x1,y1(first)), (x1+x,y1+y(second)), (0,255,0)-colour(green), 2(thickness of rectangle))
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,  255, 0), 3)

 # We try and display our image
 #cv2.imshow('Clever programmer face detector', img)
cv2.imshow('go', img)
cv2.waitKey()
