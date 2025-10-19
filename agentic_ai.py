from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import pyttsx3
import imutils
import dlib
import cv2
import threading
import requests
import geocoder
import csv
import time

# =============================
# 🔊 Initialize Audio Components
# =============================
mixer.init()
mixer.music.load("music.wav")

engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

# =============================
# 🤖 Telegram Bot Setup
# =============================
BOT_TOKEN = "8433067790:AAF6yq2vS_my-ImDHhcZdEm55sAEeVC_zMY"
CHAT_ID = "-4814851926"

# =============================
# 📍 Location Function
# =============================
def get_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            lat, lon = g.latlng
            return f"https://www.google.com/maps?q={lat},{lon}"
        else:
            return "Location unavailable"
    except Exception as e:
        return f"Error getting location: {e}"

# =============================
# 🚨 Alert Functions
# =============================
def speak_alert():
    def run_voice():
        engine.say("Driver, it seems you are not alright. Wake up!")
        engine.runAndWait()
    threading.Thread(target=run_voice, daemon=True).start()

def send_telegram_alert(message):
    def send_msg():
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✅ Telegram alert sent!")
            else:
                print(f"❌ Telegram error: {response.text}")
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")
    threading.Thread(target=send_msg, daemon=True).start()

# =============================
# 👁️ Eye Aspect Ratio Function
# =============================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# =============================
# 🤖 Agentic AI Class
# =============================
class DriverAgent:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def log_event(self, ear, flag):
        """Log drowsiness events with timestamp"""
        with open("drowsiness_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), ear, flag])

    def escalate(self):
        """Send repeated alert if driver is still drowsy"""
        location = get_location()
        message = f"⚠️ ESCALATION: Driver still drowsy!\nLocation: {location}"
        send_telegram_alert(message)
        print("⚠️ Escalation alert sent!")

    def handle_drowsiness(self, ear, flag):
        """Decide actions based on EAR and flag count"""
        self.log_event(ear, flag)
        if flag >= 2 * frame_check:  # escalate if drowsy for 2x threshold
            self.escalate()
        else:
            location = get_location()
            message = f"⚠️ ALERT: Driver seems drowsy!\nLocation: {location}"
            send_telegram_alert(message)

# =============================
# 📷 Drowsiness Detection Setup
# =============================
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
alert_active = False
msg_sent = False

agent = DriverAgent(BOT_TOKEN, CHAT_ID)

# =============================
# 🔁 Main Loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not mixer.music.get_busy():
                    mixer.music.play()

                if not alert_active:
                    speak_alert()
                    alert_active = True

                if not msg_sent:
                    agent.handle_drowsiness(ear, flag)
                    msg_sent = True
        else:
            flag = 0
            alert_active = False
            msg_sent = False
            mixer.music.stop()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# =============================
# 🧹 Cleanup
# =============================
cv2.destroyAllWindows()
cap.release()
mixer.quit()
