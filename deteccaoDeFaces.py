import cv2 


imagem = cv2.imread('people1.jpg')

imagem = cv2.resize(imagem, (800, 600))

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
('mat', imagem_cinza)

