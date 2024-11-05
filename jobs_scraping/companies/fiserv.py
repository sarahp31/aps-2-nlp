import requests

from base.apijobscrapper import APIJobScrapper


class FiservApiJobScrapper(APIJobScrapper):
    def __init__(self):
        self.body = {
            "limit": 20,
            "offset": 0,
            "searchText": ""
        }
        self.url = 'https://fiserv.wd5.myworkdayjobs.com/wday/cxs/fiserv/EXT/jobs'

    def get_json_data(self):
        return requests.post(self.url, json=self.body).json()

    def get_name(self):
        return "Fiserv"

    def get_image_url(self):
        return "https://fiserv.wd5.myworkdayjobs.com/EXT/assets/logo"

    def get_jobs(self):
        postings = []
        while True:
            data = self.get_json_data()
            job_postings = data['jobPostings']
            total_jobs = data['total']
            
            for doc in job_postings:
                url_job = 'https://fiserv.wd5.myworkdayjobs.com/wday/cxs/fiserv/EXT' + doc['externalPath']
                data_job = requests.get(url_job).json()['jobPostingInfo']
                
                posting = {
                    'title': doc['title'],
                    'url': 'https://fiserv.wd5.myworkdayjobs.com/en-US/EXT' + doc['externalPath'],
                    'description': data_job['jobDescription'],
                    'location': doc['locationsText']
                }
                postings.append(posting)

            self.body['offset'] += self.body['limit']
            
            if len(postings) >= total_jobs:
                break

        return postings
