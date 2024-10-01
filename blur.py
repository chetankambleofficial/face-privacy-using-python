apture = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for selected effect and the frame queue
selected_effect = 'blur'
frame_queue = queue.Queue(maxsize=10)

# Function to apply pixelation effect
def apply_pixelation(face_region, pixel_size=10):
    h, w, _ = face_region.shape
    small = cv2.resize(face_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated
