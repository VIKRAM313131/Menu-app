import os
lists = os.listdir("data")
print(lists)

for list in lists:
    print(list)
    print(os.listdir(f"data/{list}"))