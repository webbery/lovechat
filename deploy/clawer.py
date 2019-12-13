from requests_html import HTMLSession
from lxml import etree

def claw_answer(answer):
    session = HTMLSession()
    r = session.get("https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word="+answer, verify=False)
    html=etree.HTML(r.html.html)

claw_answer('hello world')