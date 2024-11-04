import os
import sys
import pypdf
import unittest
import urllib.parse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from app import create_app

app = create_app()

def extract_text_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip().replace(" ", "").replace("\n", " ")

class TestQueryAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_query_yields_10_results(self):
        query_text = extract_text_from_pdf('data/resumes/Davi Reis - Sr Developer.pdf')
        encoded_query = urllib.parse.quote(query_text)

        response = self.client.get(f"/query?query={encoded_query}", )
        json_response = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(json_response["results"]), 10)
        self.assertEqual(json_response["message"], "OK")

    def test_query_yields_few_results(self):
        query_text = extract_text_from_pdf('data/resumes/Bob Miller - Recruiter.pdf')
        encoded_query = urllib.parse.quote(query_text)

        response = self.client.get(f"/query?query={encoded_query}")
        json_response = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(1 < len(json_response["results"]) < 10)
        self.assertEqual(json_response["message"], "OK")

    def test_query_yields_non_obvious_results(self):
        query_text = extract_text_from_pdf('data/resumes/Uibira Bernardi - Supply Chain.pdf')
        encoded_query = urllib.parse.quote(query_text)

        response = self.client.get(f"/query?query={encoded_query}")
        json_response = response.get_json()
        
        # TODO: add assert to verify non obvious results
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json_response["results"][0]['title'], 'Business Intelligence Engineer II, S&OP Automation')
        self.assertEqual(json_response["message"], "OK")

if __name__ == "__main__":
    unittest.main()