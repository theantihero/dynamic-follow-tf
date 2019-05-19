import json
import ast
import os
import matplotlib.pyplot as plt
os.chdir("C:/Git/dynamic-follow-tf/data/15-min")
with open("df-data", "r") as f:
    d_data = f.read().split("\n")

driving_data = []
for i in d_data:
    if i != "":
        line = ast.literal_eval(i)
        if line[0] < -0.22352 or sum(line) == 0 or (sum(line[:3]) == 0 and line[5]+line[6] == 0):
            continue
        line[0] = max(line[0], 0)
        line[2] = max(line[2], 0)
        line[3] = max(line[3], 0)
        line[-1] = line[-1] / 4047.0
        driving_data.append(line)

print(len(driving_data))

save_data=True

if save_data:
    x_train = [i[:5] for i in driving_data]
    y_train = [i[5] - i[6] for i in driving_data]
    with open("x_train", "w") as f:
        json.dump(x_train, f)
    
    with open("y_train", "w") as f:
        json.dump(y_train, f)



'''x = [i for i in range(len(driving_data))]
y = [i[6] for i in driving_data]
plt.plot(x, y)
plt.show()'''