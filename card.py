import glob
import cv2
import threading
import time

sift = cv2.SIFT()

ratio = 1.54

card = None
roi = None
active = True

types_of_cards = [('as', 11), ('krol', 4), ('dama', 3),
                  ('walet', 2), ('10', 10), ('9', 0)]
colors = ['kier', 'karo', 'trefl', 'pik']
card_to_value = {'talia/%s_%s.jpg' % (t[0], c): t for c in colors
                                                  for t in types_of_cards}


def window_size(w):
    h = ratio * w
    w1 = 640
    w2 = 480
    return w2/2-h/2, w2/2+h/2, w1/2-w/2, w1/2+w/2

def get_patterns():
    return [(i, preprocess(cv2.imread(i))) for i in glob.glob('talia/*.jpg')]

def compute_keypoints(patterns):
    return [(name, mat, sift.detectAndCompute(mat, None)) for name, mat in patterns]

def preprocess(mat):
    mat = cv2.GaussianBlur(mat, (7, 7), 1.5)
    return mat

def match(frame, kp):
    frame = preprocess(frame)
    #cv2.imshow('roi', frame)
    kp2, des2 = sift.detectAndCompute(frame, None)

    bf = cv2.BFMatcher()
    maximum = None
    for i in kp:
        # p1 = cv2.drawKeypoints(frame, kp2)
        # p2 = cv2.drawKeypoints(i[1], i[2][0])
        # cv2.imshow('frejm', p1)
        # cv2.imshow('paterno', p2)
        # cv2.waitKey(1)
        matches_pre = bf.knnMatch(i[2][1], des2, k=2)
        matches = []
        for m, n in matches_pre:
            if m.distance < 0.7 * n.distance:
                matches.append(n)

        if maximum is None or maximum[1] < len(matches)/float(len(i[2][1]) + len(kp2)):
            maximum = (i[0], len(matches)/float(len(i[2][1]) + len(kp2)), matches)

    print 'The card is ', maximum[0], 'with cer =', maximum[1]
    if maximum[1] < 0.05:
        return None
    else:
        return maximum[0]

def threaded_matching(kp):
    global card
    while threaded_matching.active:
        if roi is not None:
            card = match(roi, kp)
            time.sleep(1)


def main():
    global roi
    w = 180
    cv2.namedWindow('Card Recognizer')
    print 'Loading patterns'
    patterns = get_patterns()
    print 'Loaded %d patterns' % len(patterns)

    pk = compute_keypoints(patterns)

    threaded_matching.active = True

    t = threading.Thread(args=(pk,), target=threaded_matching)

    t.start()
    cap = cv2.VideoCapture(0)

    result = 0

    try:
        while True:
            a, b, c, d = window_size(w)
            val, frame = cap.read()
            roi = frame[a:b, c:d]

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            gray[a:b, c:d] = roi


            if card is not None:
                info = card_to_value[card]
                cv2.putText(gray, "Czy to %s (%d)?" % info, 
                            (0, 430), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255))
                cv2.putText(gray, "Enter by zatwierdzic", (320, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

            cv2.putText(gray, "Wynik: %d" % result, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

            cv2.putText(gray, "R - reset / Q - wyjscie", (380, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

            cv2.imshow('Card Recognizer', gray)


            char = cv2.waitKey(1) & 0xFF
            if char == ord('a'):
                w += 10
            elif char == ord('z'):
                w -= 10
            elif char == ord('r'):
                result = 0
            elif char == ord('\n'):
                result += card_to_value[card][1]
            elif char == ord('q'):
                break

    except Exception as e:
        print e
        pass
    finally:
        print 'Application is going down'
        threaded_matching.active = False
        t.join()
        cap.release()

if __name__ == '__main__':
    main()
