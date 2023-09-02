import os

import matplotlib.pyplot as plt

log_files = [
    ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\old\\log.txt", 0),
    ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s64\\log.txt", 0),
    ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s128\\log.txt", 1600),
    ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s256\\log.txt", 3200),
]

for path, x_offset in log_files:
    with open(path, "r") as file:
        lines = file.readlines()

    losses = []
    for line in lines:
        pos = line.find("Train Loss: ")

        if pos < 0:
            continue

        loss = float(line[pos + 12:pos + 12 + 6])
        losses.append(loss)

    x = range(x_offset, x_offset + len(losses))
    line, = plt.plot(x, losses, label=os.path.basename(os.path.dirname(path)))

plt.yscale("log")
plt.ylim(0, 0.08)
plt.legend()
plt.show()
