import os

for i in range (30):
    if i%4 == 0: task = 'python FedAVG.py MobileNet'

    elif i%4 == 1: task = 'python FedAVG.py ResNet18'

    elif i%4 == 2: task = 'python FedAVG.py ResNet50'

    elif i%4 == 3: task = 'python FedAVG.py ResNet101'

    os.system(task)