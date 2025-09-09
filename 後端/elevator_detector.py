import os
import cv2
from ultralytics import YOLO
import numpy as np

# ---------------- 設定 ---------------- #
MODEL_PATH = "yolov8s.pt"      # 模型
INPUT_TYPE = "video"           # "image" 或 "video"
IMAGE_FOLDER = "images/"       # 如果是圖片模式 → 資料夾路徑
VIDEO_PATH = "videos/984.mp4"  # 如果是影片模式 → 影片路徑
OUTPUT_FOLDER = "output/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 設定畫面滿員的面積比例 (50%)
CAPACITY_THRESHOLD = 0.5

# ---------------- 初始化模型 ---------------- #
model = YOLO(MODEL_PATH)

# 計算所有框的聯集面積
def calculate_union_area(boxes, height, width):
    union_mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        union_mask[y1:y2, x1:x2] = 1
    return np.sum(union_mask)

# 標記結果
def display_result(frame, results):
    boxes = results[0].boxes.xyxy
    height, width, _ = frame.shape

    # 計算總物體聯集面積
    union_area = calculate_union_area(boxes, height, width)
    frame_area = width * height
    area_ratio = union_area / frame_area

    # 框框顏色
    box_color = (0, 255, 0) if area_ratio < CAPACITY_THRESHOLD else (0, 0, 255)

    # 繪製框框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 顯示總面積比例
    text = f"Area ratio: {area_ratio:.2f} {'(FULL!)' if area_ratio >= CAPACITY_THRESHOLD else ''}"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# ---------------- 圖片模式 ---------------- #
def process_images():
    print("📷 處理圖片中...")
    for img_name in os.listdir(IMAGE_FOLDER):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            image = cv2.imread(img_path)

            # YOLO 偵測
            results = model(image)
            image = display_result(image, results)

            # 儲存結果
            save_path = os.path.join(OUTPUT_FOLDER, img_name)
            cv2.imwrite(save_path, image)
            print(f"✅ 儲存偵測結果到 {save_path}")
    print("🎉 圖片處理完成！")

# ---------------- 影片模式 ---------------- #
# ---------------- 影片模式 ---------------- #
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_path}")
        return

    # 影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 取得原始檔名（不含副檔名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_analyzed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = display_result(frame, results)
        out.write(frame)

        # 即時顯示（可按 q 離開）
        cv2.imshow("Analyzing Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📹 已處理 {frame_count} 幀...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎉 影片處理完成，輸出到 {output_path}")


# ---------------- 主程式 ---------------- #
if __name__ == "__main__":
    if INPUT_TYPE == "image":
        process_images()
    elif INPUT_TYPE == "video":
        process_video(VIDEO_PATH)
    else:
        print("❌ INPUT_TYPE 必須是 'image' 或 'video'")
