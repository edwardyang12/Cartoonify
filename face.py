import cv2

# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
def face_detect(img, face_cascade):

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    newfaces = []
    for (x, y, w, h) in faces:
        w_s = int(w*1.38)
        h_s = int(h*1.38)
        y_s = 2*(h_s-h)//3
        x_s = (w_s-w)//2
        cv2.rectangle(img, (x-x_s, y-y_s), (x-x_s+w_s, y-y_s+h_s), (255, 0, 0), 2)
        newfaces.append((x-x_s, y-y_s,w_s, h_s))
    return img, newfaces

if __name__ == '__main__':
    img = cv2.imread('00000.png')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = face_detect(img, face_cascade)
    cv2.imshow('img', img)
    cv2.waitKey()
