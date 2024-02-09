import cv2
import torch

# Charger YOLOv5 depuis torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Ouvrir la vidéo
video_path = 'sample-5s.mp4'  # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

while True:
    # Lire l'image de la vidéo
    ret, frame = cap.read()

    # Vérifier si la lecture de l'image a réussi
    if not ret:
        print("Fin de la vidéo.")
        break

    # Convertir l'image OpenCV (BGR) en format RGB
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer l'inférence avec YOLOv5
    results = model(im)

    # Récupérer les prédictions
    predictions = results.xyxy[0].numpy()

    # Dessiner les bounding boxes sur l'image
    for box in predictions:
        x_min, y_min, x_max, y_max = box[0:4].astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow('Video', frame)
    results.print()
    results.save()
    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
