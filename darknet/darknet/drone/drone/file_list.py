import os

def search(path, extension):
    files = os.listdir(path)
    
    file_list = list()
    for file in files:
        ext = os.path.splitext(file)[1]

        if ext == extension:
            file_list.append(file)
    
    return file_list

file_list = search('./', '.jpg')
f = open('drone.txt', mode='w')
for file in file_list:
    file_name = file
    f.write('drone/drone/' + file_name + '\n')
f.close()
    
