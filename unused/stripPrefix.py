import os
import sys

root_dir = sys.argv[1]

mapping = []
for file in os.listdir(root_dir):
    left = file.index('_')+1
    new_file = file[left:]

    mapping.append([ file, new_file ])

print(mapping)

if input('y/N') != 'Y':
    print('Cancelled')
    sys.exit(0)

for file, new_file in mapping:
    os.rename(file, new_file)

print('Done')
