from PIL import Image
import os

folder_path = "/home/weilong/Code/Python/face_recognition/venv/dataset/me"
for filename in os.listdir(folder_path):
    if filename.endswith(".webp"):
        im = Image.open(os.path.join(folder_path, filename))
        im = im.convert("RGB")
        im.save(os.path.join(folder_path, filename.replace(".webp", ".jpg")), "JPEG")


def rename_photos(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 对图片文件按顺序重命名
    count = 1
    for old_name in image_files:
        extension = os.path.splitext(old_name)[1]  # 获取文件扩展名
        new_name = str(count) + extension
        os.rename(os.path.join(folder_path, old_name), os.path.join(folder_path, new_name))
        count += 1

# 要处理的文件夹路径
folder_path = '/home/weilong/Code/Python/face_recognition/venv/dataset/me'  # 替换为你的文件夹路径
rename_photos(folder_path)
