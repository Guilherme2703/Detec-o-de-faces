import cv2 

imagem = cv2.imread('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\img\\clock.jpg')

imagem = cv2.resize(imagem, (600, 700))

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
('mat', imagem_cinza)

detector_facial = cv2.CascadeClassifier('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\cascade\\haarcascade_frontalface_default.xml')
detector_olhos = cv2.CascadeClassifier('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\cascade\\haarcascade_frontalface_default.xml')
detector_relogios = cv2.CascadeClassifier('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\cascade\\clocks.xml')
detector_carros = cv2.CascadeClassifier('C:\\Users\\Guilherme L Freitas\\OneDrive\\Desktop\\projetos\\deteccaoDeFaces\\cascade\\')
# detector de faces
# deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.01, minNeighbors=4,
#                                              minSize=(32,32), maxSize=(100,100))

# deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.01)
# for x, y, w, h in deteccoes:
#     # print(x, y, w, h)
#     cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 5)

# # detecto de olhos
# deteccoes_olhos = detector_olhos.detectMultiScale(imagem_cinza, scaleFactor=1.01, minNeighbors=1, maxSize=(90, 90), minSize=(80, 80))
# for (x, y, w, h) in deteccoes_olhos:
#     # print(w,h)
#     cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,0,255), 2)

# detecto de relogios 
# deteccoes_relogio = detector_relogios.detectMultiScale(imagem_cinza, scaleFactor=1.5, minNeighbors=1, minSize=(150,150))
# for (x, y, w, h) in deteccoes_relogio:
#     print(w, h)
#     cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 200), 2)


cv2.imshow('mat', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()


