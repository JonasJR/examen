import sys, os, random, time, psutil, wmctrl, subprocess
from PIL import Image

path = raw_input("\nDirectory: ")
while True:
    f = random.choice(os.listdir(path))
    img = Image.open(path+"/"+f)
    img.show()
    subprocess.call(['wmctrl', '-a', '"Terminal"''])
    rank = raw_input("rank: ")
    if rank == "exit":
        break
    os.rename(path+"/"+f, path+"/"+rank+"/"+f)
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()
