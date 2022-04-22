import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False) as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
           break

        # Modo espejo para un video en directo
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pose.process para iniciar el procesado de la pose
        procesado_pose = pose.process(frame_rgb)

        # Coordenadas de los puntos claves para la sentadilla
        if procesado_pose.pose_landmarks is not None:
            # Puntos de la cadera [24]
            x1 = int(procesado_pose.pose_landmarks.landmark[24].x * width)
            y1 = int(procesado_pose.pose_landmarks.landmark[24].y * height)

            # Puntos de la rodilla [26]
            x2 = int(procesado_pose.pose_landmarks.landmark[26].x * width)
            y2 = int(procesado_pose.pose_landmarks.landmark[26].y * height)

            # Puntos del tobillo [28]
            x3 = int(procesado_pose.pose_landmarks.landmark[28].x * width)
            y3 = int(procesado_pose.pose_landmarks.landmark[28].y * height)

            # Obtenemos los tres puntos que conformaría el triángulo
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            # Calculo de los lados del triángulo dado sus puntos.
            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            # Calcular el Ángulo. (Arcoseno) (Radianes)
            # degress -> Para convertir radianes a grados.
            angulo = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))


            # Visualización
            # IMG Auxiliar - (Espectro negro)
            tracking_pose = np.zeros(frame.shape, np.uint8)
            cv2.line(tracking_pose, (x1, y1), (x2, y2), (255, 255, 0), 20)
            cv2.line(tracking_pose, (x2, y2), (x3, y3), (255, 255, 0), 20)
            cv2.line(tracking_pose, (x1, y1), (x3, y3), (255, 0, 0), 5)

            # Dibujar contorno del triángulo - Pasamos los tres puntos del triángulo
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])

            # Con fillPoly podremos aportarle relleno al contorno de un polígono
            # fillPoly(img, pts, color) {
                # img = imagen a tratar |
                # pts = Array de polígonos, donde cada polígono es un array de puntos |
                # color = Color del relleno.
            cv2.fillPoly(tracking_pose, pts=[contours], color=(128, 0, 250))

            # Creamos una nueva visualización, añadiendo la imagen principal con la auxiliar transparentadaç
            # haciendo que se superpongan.
            resultado_tracking = cv2.addWeighted(frame, 1, tracking_pose, 0.8, 0)

            # Círculos para marcar los puntos de referencia
            cv2.circle(resultado_tracking, (x1, y1), 6, (0, 255, 0), 4)
            cv2.circle(resultado_tracking, (x2, y2), 6, (255, 0, 0), 4)
            cv2.circle(resultado_tracking, (x3, y3), 6, (0, 0, 255), 4)

            # Visualizar el ángulo
            cv2.putText(resultado_tracking, str(int(angulo)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)

            cv2.imshow("Tracking", resultado_tracking)
        cv2.imshow("Entrada", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()