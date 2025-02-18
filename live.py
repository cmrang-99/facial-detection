import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if not result:
        break  # terminate the loop if the frame is not read successfully

    # apply the function we created to the video frame
    faces = detect_bounding_box(
        video_frame
    )  

    # display the processed frame in a window
    cv2.imshow(
        "Admission System", video_frame
    ) 

    # Check every 1 ms if the user pressed 'q' or close the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # If the window is closed, break the loop
    if cv2.getWindowProperty("Admission System", cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()