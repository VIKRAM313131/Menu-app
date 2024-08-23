import os


for i in range(0, 10):
    os.rename(f"data/day{i+1}" , f"data/topics {i+1}")
