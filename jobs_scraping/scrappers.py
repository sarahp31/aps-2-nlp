from companies.amazon import AmazonApiJobScrapper
from companies.atlassian import AtlassianApiJobScrapper
from companies.cred import CREDApiJobScrapper
from companies.fiserv import FiservApiJobScrapper
from companies.nutanix import NutanixApiJobScrapper
from companies.paypal import PayPalApiJobScrapper
from companies.phonepe import PhonePeApiJobScrapper
from companies.razorpay import RazorpayApiJobScrapper
from companies.uber import UberApiJobScrapper
from companies.zeta import ZetaApiJobScrapper
from companies.zoho import ZohoApiJobScrapper

SCRAPPERS = [
    CREDApiJobScrapper(),
    AtlassianApiJobScrapper(),
    AmazonApiJobScrapper(),
    UberApiJobScrapper(),
    PhonePeApiJobScrapper(),
    ZetaApiJobScrapper(),
    NutanixApiJobScrapper(),
    PayPalApiJobScrapper(),
    ZohoApiJobScrapper(),
    FiservApiJobScrapper(),
    RazorpayApiJobScrapper()
]

COMPANY_LOGO_MAP = {scrapper.get_name(): scrapper.get_image_url() for scrapper in SCRAPPERS}
