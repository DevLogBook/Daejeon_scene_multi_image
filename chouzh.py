import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, interval=30):
    """
    从视频中每隔 n 帧抽取一张图片
    :param video_path: 视频文件路径
    :param output_dir: 保存图片的目录
    :param interval: 抽帧间隔（例如：30表示每30帧抽一张）
    """
    # 转换为路径对象并创建输出文件夹
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频总帧数: {total_frames}, FPS: {fps:.2f}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检查是否到达抽帧间隔
        if frame_count % interval == 0:
            # 构造文件名（例如：video_name_frame_0000.jpg）
            save_path = output_dir / f"{video_path.stem}_frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(save_path), frame)
            saved_count += 1
            
            # 打印进度
            if saved_count % 10 == 0:
                print(f"已处理帧: {frame_count}/{total_frames}, 已保存: {saved_count}张")

        frame_count += 1

    cap.release()
    print(f"\n任务完成！共抽取并保存了 {saved_count} 张图片到: {output_dir}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你自己的路径
    MY_VIDEO = r"D:\PostGraduate\数据集\自采数据集\小麦26-3-11\4c1999ac828d00fead685cc045e37d33.mp4"
    SAVE_PATH = "data/extracted_frames"
    
    # 比如你的视频是30帧/秒，设置 interval=30 就是每秒抽一张
    extract_frames(MY_VIDEO, SAVE_PATH, interval=15)
    
