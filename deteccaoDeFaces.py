import cv2 


imagem = cv2.imread('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\img\\people1.jpg')

imagem = cv2.resize(imagem, (800, 600))

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
('mat', imagem_cinza)

detector_facial = cv2.CascadeClassifier('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\cascade\\haarcascade_frontalface_default.xml')


deteccoes = detector_facial.detectMultiScale(imagem_cinza)

for x, y, w, h in deteccoes:
    # print(x, y, w, h)
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 5)

cv2.imshow('mat', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()


