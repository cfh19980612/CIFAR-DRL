import os

for i in range (30):
    if i%4 == 0: task = 'python FedAVG.py --b 128 --e 1 --net "MobileNet" '

    elif i%4 == 1: task = 'python FedAVG.py --b 128 --e 1 --net "ResNet18" '

    elif i%4 == 2: task = 'python FedAVG.py --b 128 --e 1 --net "ResNet50"'

    elif i%4 == 3: task = 'python FedAVG.py --b 128 --e 1 --net "ResNet101"'

    os.system(task)