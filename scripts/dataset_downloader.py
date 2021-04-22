import os

import requests
from bs4 import BeautifulSoup
import re
from utilities.config import OUTPUT_DIR

SOURCE = 'https://transcripts.foreverdreaming.org'
CHOSEN_CHARACTER = ['Ted', 'Narrator']
OUTPUT_FILE = os.path.join(OUTPUT_DIR, CHOSEN_CHARACTER[0] + '.txt')
os.makedirs(OUTPUT_DIR, exist_ok=True)
open(OUTPUT_FILE, 'w+', encoding='utf-8').close()


def scrape_episode(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    main_body = soup.find('div', {'class': 'postbody'})
    with open(OUTPUT_FILE, 'a+', encoding='utf-8') as f:
        relevant_scene = False
        scene = []
        for paragraph in main_body.find_all('p', text=re.compile("(^[a-zA-Z]+:.+)|(^Scene .+)")):
            if paragraph.text.startswith('Scene'):
                if relevant_scene:
                    f.write("\n".join(scene) + '\n')
                    relevant_scene = False
                    scene = []
                else:
                    scene = []
                continue
            found_alias_index = -1
            counter = 0
            for character in CHOSEN_CHARACTER:
                if paragraph.text.startswith(character):
                    relevant_scene = True
                    found_alias_index = counter
                    break
                counter += 1
            if found_alias_index == -1:
                scene.append(re.sub('(^[a-zA-Z]+:)', 'Person:', paragraph.text, 1))
            if found_alias_index == 0:
                scene.append(paragraph.text)
            if found_alias_index > 0:
                scene.append(re.sub('(^[a-zA-Z]+:)', CHOSEN_CHARACTER[0] + ':', paragraph.text, 1))
        if relevant_scene:
            f.write("\n".join(scene) + '\n')


def main():
    for i in range(0, 225, 25):
        url = SOURCE + '/viewforum.php?f=177&start={}'.format(i)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        for episode_tag in soup.find_all('a', text=re.compile(".+x.+")):
            scrape_episode(SOURCE + episode_tag['href'][1:])


if __name__ == '__main__':
    main()
