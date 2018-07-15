import glob
import csv
import pandas

path = '../mtsamplesdata/*.txt'
files = glob.glob(path)
arr = []
for file in files:
    with open(file, 'r') as f:
      isContent = False
      sample = []
      con = []
      for line in f:
        if 'Sample Type' in line:
          sample.append(line.split(':')[1].strip())
        if 'Sample Name:' in line:
          sample.append(line.split(':')[1].strip())
        content = []
        if "Description" in line:
            isContent = True
        if isContent and ("Keywords:" not in line or "Sponsored Content" not in line):
            content.append(line)
        if "Keywords:" in line or "Sponsored Content" in line:
            isContent = False
            break
        else:
            if len(content) > 0:
              output = ''.join(str(r).strip('\n') for v in content for r in v)
              con.append(output)
      if isContent:
        break
      if len(sample) > 0:
        arr.append(sample)
      if len(con) > 0:
        con1 = ' '.join(con)
        con1 = con1.replace('Description: ', '', 1)
        sample.append(con1)

#print(arr)

with open("../txtTocsv_output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(['Sample Type','Sample Name','Description'])
    writer.writerows(arr)
