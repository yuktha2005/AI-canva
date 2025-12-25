import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Canvas
canvas = None
prev_x, prev_y = 0, 0

# Colors (BGR)
colors = [
     (0, 0, 0) ,
     (255, 0, 0),    
     (0, 255, 255),  
     (255, 0, 255), 
     (0, 255, 0)   
          
]
color_names = ["ERASER", "BLUE", "GREEN", "YELLOW","PURPLE"]
current_color = colors[0]
current_idx = 0

def fingers_up(hand):
    fingers = []
    fingers.append(hand.landmark[8].y < hand.landmark[6].y)   
    fingers.append(hand.landmark[12].y < hand.landmark[10].y)  
    return fingers

def draw_palette(img):
    h, w, _ = img.shape
    box_w = w // len(colors)
    toolbar_y1 = 10
    toolbar_y2 = 70
    for i, col in enumerate(colors):
        x1 = i * box_w
        x2 = (i + 1) * box_w
        # filled color box
        cv2.rectangle(img, (x1, toolbar_y1), (x2, toolbar_y2), col, -1)

        # contrasting label color
        brightness = int(col[0]) + int(col[1]) + int(col[2])
        text_color = (0, 0, 0) if brightness > 382 else (255, 255, 255)

        # label
        cv2.putText(img, color_names[i], (x1 + 10, toolbar_y2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # border for each box
        cv2.rectangle(img, (x1, toolbar_y1), (x2, toolbar_y2), (50, 50, 50), 2)

    # highlight current selection
    try:
        sel_x1 = current_idx * box_w
        sel_x2 = (current_idx + 1) * box_w
        cv2.rectangle(img, (sel_x1 + 4, toolbar_y1 + 4), (sel_x2 - 4, toolbar_y2 - 4), (255, 255, 255), 3)

        # draw a small thumb marker centered on the selected color box
        center_x = (sel_x1 + sel_x2) // 2
        center_y = (toolbar_y1 + toolbar_y2) // 2
        cv2.circle(img, (center_x, center_y), 12, (255, 255, 255), 2)
        cv2.circle(img, (center_x, center_y), 8, colors[current_idx], -1)
    except Exception:
        pass

def draw_toolbar(img):
    h, w, _ = img.shape
    overlay = img.copy()
    alpha = 0.55
    # translucent toolbar background
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # title
    cv2.putText(img, "AI Canvas", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.9, (240, 240, 240), 2)

    # current color indicator box
    cv2.rectangle(img, (w - 140, 12), (w - 40, 78), (60, 60, 60), -1)
    cv2.rectangle(img, (w - 136, 16), (w - 44, 74), current_color, -1)
    cv2.putText(img, "Current", (w - 130, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    draw_toolbar(frame)
    draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "NONE"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index_up, middle_up = fingers_up(hand)

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # thumb tip position
            thumb_x = int(hand.landmark[4].x * w)
            thumb_y = int(hand.landmark[4].y * h)

            # Selection using index+middle OR thumb extended into toolbar area
            thumb_extended_up = hand.landmark[4].y < hand.landmark[3].y

            if (index_up and middle_up) or (thumb_extended_up and thumb_y < 90):
                mode = "SELECT"
                prev_x, prev_y = 0, 0

                # choose which pointer to use for selection (index if available, else thumb)
                sel_x = x if (index_up and middle_up) else thumb_x
                sel_y = y if (index_up and middle_up) else thumb_y

                if sel_y < 90:
                    box_w = w // len(colors)
                    idx = sel_x // box_w
                    if 0 <= idx < len(colors):
                        current_color = colors[idx]
                        current_idx = idx

                cv2.circle(frame, (sel_x, sel_y), 15, current_color, cv2.FILLED)

            # Draw / Erase mode
            elif index_up and not middle_up:
                mode = "DRAW"
                cv2.circle(frame, (x, y), 10, current_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                thickness = 40 if current_color == (0, 0, 0) else 8
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, f"Mode: {mode}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame,
                "Index: Draw | Index+Middle: Select Color | C: Clear | Q: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("AI Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



