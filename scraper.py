from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_content_from_url(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        print("The URL entered cannot be scraped!")
        return None, None

    html_content = response.content
    dom = BeautifulSoup(html_content, 'html.parser')

    # Select div containing the README content
    div_tag = dom.find('div', class_='Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned')
    if not div_tag:
        print("No div tag found")
        return None, None

    # Select article tag within the div
    article_tag = div_tag.find('article', class_='markdown-body entry-content container-lg')
    if not article_tag:
        print("No article tag found")
        return None, None

    # Extract all paragraph and heading tags within the article
    content_tags = article_tag.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    # Extract additional content from specified div classes
    additional_div_classes = [
        'snippet-clipboard-content notranslate position-relative overflow-auto',
        'highlight highlight-source-shell notranslate position-relative overflow-auto'
    ]

    for class_name in additional_div_classes:
        additional_div = dom.find('div', class_=class_name)
        if additional_div:
            pre_tag = additional_div.find('pre')
            if pre_tag:
                code_tag = pre_tag.find('code')
                if code_tag:
                    content_tags.append(code_tag)

    return content_tags

def get_readme_content(url):
    content_tags = get_content_from_url(url)
    if not content_tags:
        return None, []

    content_variable = []
    link_texts = []
    link_hrefs = []

    for tag in content_tags:
        if tag.name == 'p':
            anchors = tag.find_all('a')
            for anchor in anchors:
                link_texts.append(anchor.get_text())
                link_hrefs.append(anchor.get('href'))
        content_variable.append(tag.get_text())

    repo_name = url.split("/")[-1]
    test_vec = model.encode([repo_name])[0]
    filtered_links = []

    for sent, href in zip(link_texts, link_hrefs):
        if 'github.com' in href:
            similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
            if similarity_score > 0.07:
                filtered_links.append(href)

    concatenated_content = "\n".join(content_variable)
    return concatenated_content, filtered_links
