import re
import csv
import glob
import json
from xml.etree import ElementTree as ET

def readData(filename):
    data = {
        "SampleType": [],
        "SampleName": [],
        "problem": [],
        "drug": [],
        "treatment": [],
        "test": [],
    }
    tree = ET.parse(filename)

    fulltext = ''
    for a in tree.iter():
        #print a.__dict__
        if 'Sofa' in a.tag:
            fulltext = a.attrib['sofaString']
            substring = re.search(r'Sample Type.*\nSample Name.*', fulltext.encode('ascii', 'ignore')).group(0)
            data['SampleType'].append(re.search(r': .*', substring).group(0).replace(': ', '').strip('\r'))
            data['SampleName'].append(re.search(r': .*$', substring).group(0).replace(': ', '').strip('\r'))

    for a in tree.iter():
        if a.attrib.get('semanticTag') in ['problem', 'treatment', 'test']:
            text = fulltext[int(a.attrib.get('begin')): int(a.attrib.get('end'))]
            desc = json.loads(a.attrib.get('attribute', '{}')).get('umlsCuiDesc', '')
            #print text + "|" + desc
            if desc != '':
                data[a.attrib['semanticTag']].append(desc.encode('ascii', 'ignore')) # if desc != '' else str(text.encode('ascii', 'ignore')))
        elif a.attrib.get('semanticTag') in ['drug']:
            text = fulltext[int(a.attrib.get('begin')): int(a.attrib.get('end'))]
            desc = json.loads(a.attrib.get('attribute', '{}')).get('umlsCuiDesc', '')
            #print text + "|" + desc
            #if desc != '':
            data[a.attrib['semanticTag']].append(text.encode('ascii', 'ignore')) # if desc != '' else str(text.encode('ascii', 'ignore')))

    for key, value in data.iteritems():
        data[key] = ", ".join(value)

    return data


if __name__ == '__main__':
    path = '../clampoutput/*.xmi'
    csvfile = '../clampdata_1.csv'
    parsedData = []
    files = glob.glob(path)
    for file in files:
        try:
            parsedData.append(readData(file))
        except Exception as e:
            print file, e

    with open(csvfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['SampleType','SampleName','problem', 'drug', 'treatment', 'test'])
        for i in parsedData:
            writer.writerow([i['SampleType'], i['SampleName'], i['problem'], i['drug'], i['treatment'], i['test']])

