import os

import matplotlib.pyplot as plt

log_files = [
    "D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\old\\log.txt",
    "D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\log.txt"
]

for path in log_files:
    with open(path, "r") as file:
        lines = file.readlines()

    losses = []
    for line in lines:
        pos = line.find("Train Loss: ")

        if pos < 0:
            continue

        loss = float(line[pos + 12:pos + 12 + 6])
        losses.append(loss)

    line, = plt.plot(losses, label=os.path.basename(os.path.dirname(path)))

plt.yscale("log")
plt.ylim(0, 0.08)
plt.legend()
plt.show()
