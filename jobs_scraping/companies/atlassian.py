import requests

from base.apijobscrapper import APIJobScrapper


class AtlassianApiJobScrapper(APIJobScrapper):

    def get_json_data(self, url):
        return requests.get(url).json()

    def get_name(self):
        return "Atlassian"

    def get_image_url(self):
        return "https://seeklogo.com/images/A/atlassian-logo-DF2FCF6E4D-seeklogo.com.png"

    def get_jobs(self):
        url = 'https://www.atlassian.com/endpoint/careers/listings'
        data = self.get_json_data(url)
        postings = []
        for doc in data:
            postings.append({})
            postings[-1]['title'] = doc['title']
            postings[-1]['description'] = doc['overview'] + '\n' + doc['responsibilities'] + '\n' + doc['qualifications']
            postings[-1]['url'] = doc['applyUrl']
            postings[-1]['location'] = doc['locations']
        return postings
