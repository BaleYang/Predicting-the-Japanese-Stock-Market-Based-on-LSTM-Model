import requests


headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15  '
}
url = 'https://hs.trkd-asia.com/rakutenseccht/data/hts?ric=9101.T&i=d&adj=1&fr=20201230&token=2B186984569726C9FF5CF9A4FE25777DAD53D4E80C8A6C119B09B9265702FD880ED8F1AE17EF243B67CB95CA26EA4FB5D6A0AF9E513CD3593D4552B0&qid=1677674773416'
response = requests.get(url=url, headers=headers)


