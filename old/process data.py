import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random

os.chdir("C:/Git/dynamic-follow-tf/data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\df"
d_data = []
gm_counter = 0
other_counter = 0
for folder in os.listdir(data_dir):
    if any([sup_car in folder for sup_car in ["CHEVROLET VOLT PREMIER 2017", "HOLDEN ASTRA RS-V BK 2017", "gbergman"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > 40000: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                for line in df:
                    if line != "" and "[" in line and "]" in line and len(line) >= 40:
                        gm_counter+=1
                        d_data.append(ast.literal_eval(line))
    # the following should improve performance for deciding when and how much to apply gas (but might reduce braking performance)
    '''elif any([sup_car in folder for sup_car in ["TOYOTA COROLLA 2017", "TOYOTA PRIUS 2017", "TOYOTA RAV4 HYBRID 2017", "TOYOTA RAV4 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > 40000: #if bigger than 40kb
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                for line in df:
                    if line != "" and "[" in line and "]" in line and len(line) >= 40:
                        line = ast.literal_eval(line)
                        line[6] = 0.0  # don't include brake pressure
                        d_data.append(line)
                        other_counter+=1
                        #if line[6] == 0.0 or line[5] > 0.0:  # for cars without brake sensor (like toyotas), only include lines with no brake. brake pressure is too inaccurate
                            #line[6] = 0.0  # don't include brake pressure
                            #other_counter+=1
                            #d_data.append(line)  # need to experiment with including braking samples, but setting brake to 0 so the model will coast instead of not knowing what to do'''
                    
driving_data = []
for line in d_data:  # do filtering
    if line[0] < -0.22352 or sum(line) == 0: #or (sum(line[:3]) == 0):
        continue
    if line[4] > 5 or line[4] < -5: # filter out crazy acceleration
        continue
    #line[0] = max(line[0], 0)
    #line[2] = max(line[2], 0)
    #line[3] = max(line[3], 0)
    
    #line[-1] = line[-1] / 4047.0  # only for corolla
    #line = [line[0], line[1], (line[2]-line[0]), line[3], line[4], line[5], line[6], line[7]] # this makes v_lead, v_rel instead
    driving_data.append(line)
random.shuffle(driving_data)

split_leads=False
if split_leads:
    split_data = [[]]
    counter = 0
    for idx, i in enumerate(driving_data):
        if idx != 0:
            if abs(i[3] - driving_data[idx-1][3]) > 2.5: # if new lead, split
                counter+=1
                split_data.append([])
        split_data[counter].append(i)
    #print([len(i) for i in split_data])
    
    new_split_data = []
    for i in split_data:
        if len(i) >= 20: # remove small amounts of data
            new_split_data.append(i)
    
    x_train = []
    y_train = []
    n = 20
    # the following further splits the data into 20 sample sections for training
    for lead in new_split_data:
        '''if lead[0][0] > 4.4704:
            continue'''
        if len(lead) == n: # don't do processing if list is just 20
            x_train.append([i[:5] for i in lead])
            y_train.append(lead[-1][5] - lead[-1][6]) # append last gas/brake values
            continue
        
        lead_split = [lead[i * n:(i + 1) * n] for i in range((len(lead) + n - 1) // n)]
        if lead_split[-1] != 20: # if last section isn't 20 samples, remove
            del lead_split[-1]
        lead_split_x = [[x[:5] for x in i] for i in lead_split] # remove gas and brake from xtrain
        lead_split_y = [(i[-1][5] - i[-1][6]) for i in lead_split] #only (last) gas/brake
        for lead in lead_split_x:
            x_train.append(lead)
        y_train.extend(lead_split_y)
    
    for i in x_train:
        for x in i:
            if len(x) != 5:
                print("uh oh")
        if len(i) != 20:
            print("Bad") # check for wrong sizes in array
    
    save_data = True
    if save_data:
        with open("LSTM/x_train-gbergman", "w") as f:
            json.dump(x_train, f)
        with open("LSTM/y_train-gbergman", "w") as f:
            json.dump(y_train, f)

even_out=True
if even_out:  # makes number of gas/brake/nothing samples equal to min num of samples
    gas = [i for i in driving_data if i[5] - i[6] > 0]
    nothing = [i for i in driving_data if i[5] - i[6] == 0]
    brake = [i for i in driving_data if i[5] - i[6] < 0]
    to_remove_gas = len(gas) - min(len(gas), len(nothing), len(brake)) if len(gas) != min(len(gas), len(nothing), len(brake)) else 0
    to_remove_nothing = len(nothing) - min(len(gas), len(nothing), len(brake)) if len(nothing) != min(len(gas), len(nothing), len(brake)) else 0
    to_remove_brake = len(brake) - min(len(gas), len(nothing), len(brake)) if len(brake) != min(len(gas), len(nothing), len(brake)) else 0
    del gas[:to_remove_gas]
    del nothing[:to_remove_nothing]
    del brake[:to_remove_brake]
    driving_data = gas + nothing + brake
    
    

print(len(driving_data))
print()
y_train = [i[5] - i[6] for i in driving_data]
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))
print("\nSamples from GM: {}, samples from other cars: {}".format(gm_counter, other_counter))

save_data = True
if save_data:
    save_dir="gm-only"
    x_train = [i[:5] for i in driving_data]
    with open(save_dir+"/x_train", "w") as f:
        json.dump(x_train, f)
    
    with open(save_dir+"/y_train", "w") as f:
        json.dump(y_train, f)
    print("Saved data!")

'''driving_data = [i for idx, i in enumerate(driving_data) if 20000 < idx < 29000]
x = [i for i in range(len(driving_data))]
y = [i[0] for i in driving_data]
plt.plot(x, y)
plt.show()'''