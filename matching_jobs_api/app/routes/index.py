import logging
from flask import Blueprint, request
from app.services.get_query import get_query

logger = logging.getLogger(__name__)
index_bp = Blueprint('index_bp', __name__)

@index_bp.route('/')
def index():
    logger.debug('Index route called')
    return {
        'message': 'Welcome to APS 1 NLP API!',
        'status': 'OK'
    }

@index_bp.route('/query', methods=['GET'])
def query():
    query = request.args.get('query')
    threshold = request.args.get('threshold', default=0.2, type=float)
    return get_query(query, threshold)