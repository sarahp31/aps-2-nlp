import requests

from base.apijobscrapper import APIJobScrapper


class PayPalApiJobScrapper(APIJobScrapper):
    LIMIT = 10

    def get_json_data(self, url):
        response = requests.get(url).json()
        return response['positions']

    def get_name(self):
        return "PayPal"

    def get_image_url(self):
        return "https://upload.wikimedia.org/wikipedia/commons/a/a4/Paypal_2014_logo.png"

    def get_jobs(self):
        url = 'https://paypal.eightfold.ai/api/apply/v2/jobs?num=10'
        offset = 0
        postings = []
        while True:
            url_with_offset = url + '&start=' + str(offset)
            data = self.get_json_data(url_with_offset)
            for doc in data:
                postings.append({
                    "title": doc['name'],
                    "description": doc['job_description'],
                    "location": doc['location'],
                    "url": 'https://paypal.eightfold.ai/careers/job?domain=paypal.com&pid=' + str(doc['id'])
                })
            if len(data) < self.LIMIT:
                break
            offset += self.LIMIT
        return postings
