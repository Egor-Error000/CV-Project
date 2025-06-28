from detector import load_model, detect_people
from tracker import create_tracker
from utils import setup_video_io, draw_tracks
import cv2
from tqdm import tqdm
import os

def main():
    video_path = 'data/crowd.mp4'
    output_path = 'results/output.mp4'

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео {video_path} не найдено")

    print("[INFO] Загружаем модель YOLOv8x...")
    model = load_model("yolov8x.pt")

    print("[INFO] Инициализируем трекер DeepSort...")
    tracker = create_tracker()

    cap, writer, fps, total_frames = setup_video_io(video_path, output_path)
    print(f"[INFO] Видео: {int(cap.get(3))}x{int(cap.get(4))}, {fps:.2f} FPS, {total_frames} кадров")

    pbar = tqdm(total=total_frames, desc="Inference + Tracking")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(model, frame)
        tracks = tracker.update_tracks(detections, frame=frame)
        frame = draw_tracks(frame, tracks)

        writer.write(frame)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"[INFO] Готово! Результат сохранён в {output_path}")

if __name__ == "__main__":
    main()

