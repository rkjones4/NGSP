import os

SKIP = []

cats = os.listdir('../hier')

for i, cat in enumerate(cats):
    if cat in SKIP:
        continue
    print(f"On {cat} ({i}/{len(cats)})")
    os.system(f'python3 make_splits.py ../data/{cat} {cat} split')
