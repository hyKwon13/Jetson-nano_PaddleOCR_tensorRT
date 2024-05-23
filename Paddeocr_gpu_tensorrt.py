import cv2
from paddleocr import PaddleOCR

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt, rectangles
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
        bottom_right_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bottom_right_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        rectangles.append((top_left_pt, bottom_right_pt))

def perform_ocr(frame, top_left_pt, bottom_right_pt, ocr):
    if top_left_pt == (-1, -1) or bottom_right_pt == (-1, -1):
        return
    roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
    if roi.size == 0:
        return
    result = ocr.ocr(roi, cls=False)
    result = result[0]
    if result is not None:
        for line in result:
            print(line[1][0])
            cv2.putText(frame, line[1][0], (top_left_pt[0], bottom_right_pt[1]+21), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.namedWindow('paddleocr')
cv2.setMouseCallback('paddleocr', draw_rectangle)
cap = cv2.VideoCapture(0)

drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
rectangles = [] 

ocr = PaddleOCR(
    use_gpu=True,
    use_tensorrt=True,
    use_angle_cls=False,
    lang='en',
    show_log=False,
)


            
for _ in range(30):
    _, _ = cap.read()

while True:
    ret, frame = cap.read()

    for rect in rectangles:
        cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)
    
    if drawing and top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
    if not drawing and top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
        perform_ocr(frame, top_left_pt, bottom_right_pt, ocr)
    cv2.imshow('paddleocr', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
