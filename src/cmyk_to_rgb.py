<<<<<<< HEAD
# DID NOT END UP USING


from PIL import Image
import os
import numpy as np

for i, letter in enumerate(['apple','banana','mixed','orange']):
        directory = f'../data/fruit-images-for-object-detection/train_zip/train/{letter}/'
        files = os.listdir(directory)
        label = np.array([0]*10)
        label[i] = 1
        for file in files:
            print(f'Looking at {file}')
            try:
                image = Image.open(directory+file)
                if image.mode == 'CMYK':
                    image = image.convert('RGB').save(directory+file)
                    print(f'Converted {image.mode} {directory+file}')
            except:
                continue
=======
# DID NOT END UP USING


from PIL import Image
import os
import numpy as np

for i, letter in enumerate(['apple','banana','mixed','orange']):
        directory = f'../data/fruit-images-for-object-detection/train_zip/train/{letter}/'
        files = os.listdir(directory)
        label = np.array([0]*10)
        label[i] = 1
        for file in files:
            print(f'Looking at {file}')
            try:
                image = Image.open(directory+file)
                if image.mode == 'CMYK':
                    image = image.convert('RGB').save(directory+file)
                    print(f'Converted {image.mode} {directory+file}')
            except:
                continue
>>>>>>> 1e4c7b82003fce8e238b4a943ea5b4479e518f0c
