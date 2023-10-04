import os

import matplotlib.pyplot as plt

# log_files = [
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\old\\log.txt", 0),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s64\\log.txt", 0),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s128\\log.txt", 1600),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new\\s512\\log.txt", 3200),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\log.txt", 0),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\log.txt", 1600),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\clean\\log.txt", 4000),
#     ("D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\cosine\\log.txt", 4000),
#     (
#         "D:\\Osu! Dingen\\Beatmap ML Datasets\\results\\new2\\r\\cosine\\l1\\log.txt",
#         4600,
#     ),
# ]
log_files = [
    ("..\\results\\037-DiT-B\\log.txt", 0),
    ("..\\results\\038-DiT-B\\log.txt", 0),
    ("..\\results\\039-DiT-B\\log.txt", 0),
    ("..\\results\\044-DiT-B\\log.txt", 0),
    ("..\\results\\048-DiT-B\\log.txt", 5750),
]

for path, x_offset in log_files:
    with open(path) as file:
        lines = file.readlines()

    losses = []
    for line in lines:
        pos = line.find("Train Loss: ")

        if pos < 0:
            continue

        loss = float(line[pos + 12 : pos + 12 + 6])
        losses.append(loss)

    x = range(x_offset, x_offset + len(losses))
    (line,) = plt.plot(x, losses, label=os.path.basename(os.path.dirname(path)))

plt.yscale("log")
# plt.ylim(0, 0.08)
plt.legend()
plt.show()
