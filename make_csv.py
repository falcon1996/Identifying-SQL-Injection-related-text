import glob
import csv

data = []

def file_changer(filename):
    with open(filename, 'r+') as txt_file:
        
        content = txt_file.read()
        
        content = content.replace(',', '')
        content = content.replace('.', '')
        content = content.replace('\n', ' ')
        data.append(content)
        print('Added value to list!')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

def pos_write_to_csv(list_of_text):
    with open('./path-to-file/pos.csv', 'a+') as csvfile:
        for domain in list_of_text:
            print(domain)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            csvfile.write(domain + '\n')


def neg_write_to_csv(list_of_text):
    with open('./path-to-file/neg.csv', 'a+') as csvfile:
        for domain in list_of_text:
            print(domain)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            csvfile.write(domain + '\n')



for i in range(0,10065):
    i = str(i)
    print('Writing positive csv for :'+i)
    data.clear()
    for file in glob.glob('./path-to-file/pos/'+i+'.txt'):
        file_changer(file)

    pos_write_to_csv(data)

for i in range(0,10215):
    i = str(i)
    print('Writing negative csv for :'+i)
    data.clear()
    for file in glob.glob('./path-to-file/neg/'+i+'.txt'):
        file_changer(file)

    neg_write_to_csv(data)