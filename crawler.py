from bs4 import BeautifulSoup

import urllib3
from selenium import webdriver
import time
baseurl="http://www.mtsamples.com"


def save(x):
    print x
    temp=str(x[x.index("="):])
    temp=temp.replace("%","")
    temp=temp.replace('/',"")
    temp=temp.replace(' ',"")
    driver = webdriver.Chrome("../chromedriver")
    driver.get(x)
    '''WebDriverWait(driver, 100).until( lambda driver: driver.find_element_by_id('sampletext'))'''
    results = driver.find_element_by_id('sampletext').text
    temp1=""
    temp1=temp1+results.encode('utf-8').strip()

    '''print(results)'''

    filename="../mtsamplesdata/"+temp+".txt"

    f=open(filename,'w')
    f.write(temp1)
    f.close()
    driver.quit()



def getdesclinks(desc):
    print desc
    try:
        http=urllib3.PoolManager()
        links=[]
        page=http.request("GET",desc)
        soup=BeautifulSoup(page.data,"html.parser")
        temp=soup.find('table',id="Browse").find('tr').find('td').findAll('a')
        for i in temp:
            links.append(baseurl+str(i.get('href')).replace(" ",""))
        print "The number of links on this page is"
        print "\n"
        print len(links)
        for l in links:
            save(l)
    except Exception as e:
        print e
        pass

def getlinks(l):
    print l
    try:
        http=urllib3.PoolManager()
        temp=[]
        page=http.request("GET",l)
        soup=BeautifulSoup(page.data,"html.parser")
        li=soup.find('div',class_="Contrast").findAll('a')
        for i in li:
            temp.append(baseurl+str(i.get("href")))
        temp.pop(-1)
        temp.pop(-1)
        for l in temp:
            getdesclinks(l)
    except Exception as e:
        print e
        pass

linksOfInterest=['http://www.mtsamples.com/site/pages/browse.asp?type=6-Cardiovascular&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=18-Dermatology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=100-ENT+-+Otolaryngology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=24-Gastroenterology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=96-Hematology%20-%20Oncology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=41-Nephrology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=42-Neurology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=45-Obstetrics+%2F+Gynecology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=46-Ophthalmology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=49-Orthopedic&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=72-Psychiatry%20/%20Psychology&page=1',
                'http://www.mtsamples.com/site/pages/browse.asp?type=82-Urology&page=1']

for i in linksOfInterest:
    getlinks(i)
