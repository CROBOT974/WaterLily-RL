import imageio.v2 as imageio
import os

def create_GIF(frames_dir, output_gif):



    # 读取所有图片文件并排序（假设文件名中有数字）
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    # 读取图像帧
    frames = [imageio.imread(os.path.join(frames_dir, f)) for f in frame_files]

    # 保存为 GIF
    imageio.mimsave(output_gif, frames, fps=100)

if __name__ == "__main__":
    frames_dir = "../images"
    output_gif = "../viv_simulation_2.gif"

    create_GIF(frames_dir, output_gif)