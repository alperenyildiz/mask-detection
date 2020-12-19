import cv2
import numpy as np



whT = 320
cap = cv2.VideoCapture(0)
confThreshold=0.5#maskeyi tanımak için geçerli olan treshold
nmsThreshold=0.3 #Bounding box eşik değeri

classFile="maske.names"
classNames=[]

with open(classFile,"rt") as f:
    classNames=f.read().rstrip("\n").split("\n")


#Ağırlık ve yapılandırıcı dosyalarımızın yolunu ve türünü belirliyoruz
modelConfiguration="maske.cfg"
modelWeights="maske_final.weights"

#YOLO modelimiz ile haberleşiyoruz.
model=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
#Opencv arka uçta cpu ile kullanmak için:
model.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(detectionLayers,img):
    hT,wT,cT=img.shape
    bbox=[] #sınırlayıcı kutularımızı tutan liste
    classIds=[] #sınıflarımızı tutan id listesi
    confs=[] #Bulunan nesnelerin güven değerini tutan listemiz


    for detectionLayer in detectionLayers:
        for objectDetection in detectionLayer:
            scores=objectDetection[5:]  #ilk 5 değer bounding box ile ilgili.
            classId=np.argmax(scores)  #skorlar içerisinden en büyüğünün bulunduğu indis bilgisi alıyoruz.
            confidence=scores[classId] #En büyük güven değeri
            if confidence>confThreshold: #Nesnelerin kabul edilebilmesi için belirlediğimiz eşik değerinin kontrolü
                w,h=int(objectDetection[2]*wT),int(objectDetection[3]*hT)
                x,y=int((objectDetection[0]*wT)-w/2),int((objectDetection[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                #Nesnenin etrafına bounding box çizilebilmesi için x ve y noktaları belirleniyor.



    #Non maxiumum suppression ;
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3] #Güven skoru yüksek olan boxların noktaları alınıyor

        if classNames[classIds[i]].upper() == "NO-MASK":
            g,b,r = 0,0,255
        else:
            g,b,r = 0,255,0

        cv2.rectangle(img,(x,y),(x+w,y+w),(g,b,r),3)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (g,b,r), 2)



while True:
    success,img=cap.read()
    #Darknet/YOLO görüntüleri blob formatında istediğinden blob formatına çeviriyoruz.(Binary Large Object)
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False) #[img,scale factor,boyut,]
    model.setInput(blob)

    layerNames=model.getLayerNames()
    #YOLO'da bulunan 3 adet çıkış katmanını alıyoruz.  yolo_82, yolo_94, yolo_106
    outputLayers=[layerNames[i[0]-1] for i in model.getUnconnectedOutLayers()]
    detectionLayers=model.forward(outputLayers) #Sinir ağı çıkışlarımızın döndüğü değerler matrisi


    findObject(detectionLayers,img)
    #cv2.namedWindow("Mask Detection", cv2.WINDOW_NORMAL)
    #cv2.flip(img,1)
    cv2.imshow("Mask Detection",img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

    cv2.waitKey(50)
