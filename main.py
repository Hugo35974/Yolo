import cv2
import torch

# Charger YOLOv5 depuis torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ouvrir la vidéo
video_path = "sample-5s.mp4"  # Remplacez par le chemin de votre vidéo
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

    # Dessiner les bounding boxes sur l'image avec les labels
    for box in predictions:
        x_min, y_min, x_max, y_max = box[0:4].astype(int)
        label = model.names[int(box[5])]  # Récupérer le label à partir de l'indice de classe
        confidence = box[4]  # Confidence de la détection
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Afficher l'image
    cv2.imshow('Video', frame)
    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
