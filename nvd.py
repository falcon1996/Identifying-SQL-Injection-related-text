import json

# For vulnerability specific data
year_list = ['2002','2003','2004','2005','2006','2007','2008','2009', '2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']

for year in year_list:

    injection_texts = []
    non_injection_texts = []

    with open('./path-to-file/nvd/nvdcve-1.1-'+year+'.json') as f:
        data = json.loads(f.read())

        for item in data['CVE_Items']:

            if('sql injection' in item['cve']['description']['description_data'][0]['value'].lower()):
                injection_texts.append(item['cve']['description']['description_data'][0]['value'])
            
            else:
                non_injection_texts.append(item['cve']['description']['description_data'][0]['value'])

    f.close()

    num=0
    for text in injection_texts:
        print('Writing SQL injection positive data for '+ year + '!')
        f = open("./path-to-file/pos-"+year+"/cv_"+year+'_'+str(num)+".txt", "a")
        f.write(text)
        f.close()
        num+=1
    print('Total injection cyber related entries: ',num)

    num=0
    for text in non_injection_texts:
        print('Writing SQL injection negative data for '+ year + '!')
        f = open("./path-to-file/neg/neg-"+year+"/cv_"+year+'_'+str(num)+".txt", "a")
        f.write(text)
        f.close()
        num+=1
    print('Total non injection cyber related entries: ',num)