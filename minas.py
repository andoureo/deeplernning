import cv2
import numpy as np

# フレーム差分の計算
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    return diff

def main():
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    # フレームを3枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]*1.2), int(frame.shape[0]*1.2)))

        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=10)
        mask = cv2.resize(mask, (int(mask.shape[1]*1.2), int(mask.shape[0]*1.2)))

        # 結果を表示
        cv2.imshow("Frame2", frame)
        cv2.imshow("Mask", mask)

        # 3枚のフレームを更新
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        key = cv2.waitKey(10)
        if key == 27:                                                       # escキーを押したら終了
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
