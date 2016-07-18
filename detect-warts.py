from wart_detection import find_warts 
import cv2

regions = find_warts("images/wart-on-skin3.png")

img = cv2.imread("images/wart-on-skin3.png")

for c in regions:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("img", img)

k = cv2.waitKey()
cv2.destroyAllWindows()

quit()

