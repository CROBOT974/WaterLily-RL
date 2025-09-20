import imageio.v2 as imageio
import os

def create_GIF(frames_dir, output_gif, step=10, fps=100):
    # 读取所有图片文件并排序（假设文件名中有数字）
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    # 抽帧（比如每 10 张取 1 张）
    sampled_files = frame_files[::step]

    # 读取图像帧
    frames = [imageio.imread(os.path.join(frames_dir, f)) for f in sampled_files]

    # 保存为 GIF
    imageio.mimsave(output_gif, frames, fps=fps, loop=0)

if __name__ == "__main__":
    frames_dir = "images"
    output_gif = "viv_simulation.gif"

    # 每 10 张取 1 张，fps=20
    create_GIF(frames_dir, output_gif, step=10, fps=20)
