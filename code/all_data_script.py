import os

CATS = ['chair', 'table', 'lamp', 'knife', 'vase', 'storagefurniture']

cats = os.listdir('../hier')

for i, cat in enumerate(cats):
    if cat not in CATS:
        continue
    print(f"On {cat} ({i}/{len(cats)}")
    os.system(f'python3 make_dataset.py ../data/{cat} {cat}')
