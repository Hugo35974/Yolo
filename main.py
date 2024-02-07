import cv2
import torch

model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur lors de la lecture de l'image de la caméra.")
        break
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(im)
    predictions = results.xyxy[0].numpy()
    for box in predictions:
        x_min, y_min, x_max, y_max = box[0:4].astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
