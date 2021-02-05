import os

def search(path, extension):
    files = os.listdir(path)
    
    file_list = list()
    for file in files:
        ext = os.path.splitext(file)[1]

        if ext == extension:
            file_list.append(file)
    
    return file_list

f = open('frame727.txt', mode='r')
text = f.readline()
f.close()

print(text)

frame = search('./', '.jpg')
for file in frame[1:]:
    file_name = os.path.splitext(file)[0]
    
    f = open('./' + file_name + '.txt', mode='w')
    f.write(text)
    f.close()
    