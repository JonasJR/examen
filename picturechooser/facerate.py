import sys, os, random, time, psutil, subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = raw_input("\nDirectory: ")
while True:
    f = random.choice(os.listdir(path))
    img = mpimg.imread(path+"/"+f)
    plt.ion()
    plt.imshow(img)
    plt.show(block=False)
    subprocess.call(['wmctrl', '-a', '"Terminal"'])
    rank = raw_input("rank: ")
    plt.clf()
    if rank == "exit":
        break
    os.rename(path+"/"+f, path+"/"+rank+"/"+f)
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()
plt.close()
