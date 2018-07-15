import re
import pandas
from xml.etree import ElementTree as ET

filename = './medtagger.xml'
csvfile = './_medtagger.csv'
semGroups = {
    "ANAT": "Anatomy",
    "CHEM": "Chemicals & Drugs",
    "DISO": "Disorders",
    "PHYS": "Physiology",
    "PROC": "Procedures"
}
data = {
    "Sample Type": [],
    "Sample Name": [],
    "Anatomy": [],
    "Chemicals & Drugs": [],
    "Disorders": [],
    "Physiology": [],
    "Procedures": []
}

tree = ET.parse(filename)

for a in tree.iter():
    if 'Sofa' in a.tag:
        full_file = a.attrib['sofaString']
        substring = re.search(r'Sample Type.*\nSample Name.*', full_file.encode('ascii', 'ignore')).group(0)
        data['Sample Type'].append(re.search(r': .*', substring).group(0).replace(': ', ''))
        data['Sample Name'].append(re.search(r': .*$', substring).group(0).replace(': ', ''))
    elif 'ConceptMention' in a.tag and a.attrib['semGroup'] in semGroups.keys():
        data[semGroups[a.attrib['semGroup']]].append(a.attrib['normTarget'])

df = pandas.DataFrame.from_dict(data, orient='index')
df.to_csv(csvfile, sep=',', encoding='utf-8')
