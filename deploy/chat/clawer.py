from requests_html import HTMLSession
from urllib.request import quote, unquote

from lxml import etree

def claw_answer(answer):
    url = quote("https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word="+answer, safe=";/?:@&=+$,", encoding="gbk") # 编码
    print(url)
    html = get_xpath(url)
    href = html.xpath('.//div[@id="wgt-list"]/dl[1]/dt[1]/a/@href')
    html = get_xpath(href[0])
    contents = html.xpath('.//div[contains(@id,"best-content")]/*/text()|.//div[contains(@id,"best-content")]/text()')
    result = ''
    # print(contents)
    for content in contents:
        if content=='\n' or content=='\r': continue
        result += content+'\n'

    return result

def get_xpath(src):
    session = HTMLSession()
    r = session.get(src, verify=False)
    return etree.HTML(r.html.html)

# print(claw_answer('1111'))