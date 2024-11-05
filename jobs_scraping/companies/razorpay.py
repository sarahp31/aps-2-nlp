import requests

from base.apijobscrapper import APIJobScrapper


class RazorpayApiJobScrapper(APIJobScrapper):

    def get_json_data(self, url):
        return requests.get(url).json()['jobs']

    def get_name(self):
        return "Razorpay"

    def get_image_url(self):
        return "https://razorpay.com/jobs/images/logos/logo.svg"

    def get_jobs(self):
        url = 'https://boards-api.greenhouse.io/v1/boards/razorpaysoftwareprivatelimited/jobs'
        data = self.get_json_data(url)
        postings = []
        for doc in data:
            url_job = 'https://boards-api.greenhouse.io/v1/boards/razorpaysoftwareprivatelimited/jobs/' + str(doc['id'])
            doc_2 = requests.get(url_job).json()
            postings.append({})
            postings[-1]['title'] = doc_2['title']
            postings[-1]['description'] = doc_2['content']
            postings[-1]['url'] = doc_2['absolute_url']
            postings[-1]['location'] = doc_2['location']['name']
        return postings


if __name__ == '__main__':
    print(RazorpayApiJobScrapper().get_jobs())