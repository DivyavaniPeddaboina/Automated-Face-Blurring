import cv2

# Load the Haar Cascade face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
video_capture = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while video_capture.isOpened():
    # Capture frame-by-frame
    success, frame = video_capture.read()
    if not success:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(35, 35))

    # Apply blur to detected faces
    for (x, y, w, h) in detected_faces:
        face_area = frame[y:y+h, x:x+w]
        blur_kernel = (w//9*2+1, h//9*2+1)  # Ensure odd kernel size
        blurred_face = cv2.GaussianBlur(face_area, blur_kernel, 25)
        frame[y:y+h, x:x+w] = blurred_face

    # Display the frame with blurred faces
    cv2.imshow('Live Face Blur', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
