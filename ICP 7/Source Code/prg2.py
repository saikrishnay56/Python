from bs4 import BeautifulSoup
import urllib.request


def search_spider():

    url = "https://en.wikipedia.org/wiki/Google"
    source_code = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")

    body = soup.find('div', {'class': 'mw-parser-output'})
    file.write(str(body.text))


search = input('type "s" to start wikiScrap, type "q" to exit')
if search == 'q' or search == 'Q':
    print("Quiting...")
    exit()
else:
    print("Creating .txt file ...")
    file = open('input.txt', 'a+', encoding='utf-8')
    search_spider()