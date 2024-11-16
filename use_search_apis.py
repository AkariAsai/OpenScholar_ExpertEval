import argparse
import csv
import json
import random
import re
import time
from xml.etree import ElementTree as ET

import requests

SS_API_KEY = "YOUR_SEMANTIC_SCHOLAR_API_KEY"


def is_integer_string(s):
    return s.isdigit()


def get_paper_data(paper_id):
    if is_integer_string(paper_id) is False:
        url = "https://api.semanticscholar.org/graph/v1/paper/" + paper_id
    else:
        url = "https://api.semanticscholar.org/graph/v1/paper/CorpusID:" + paper_id
    # Define which details about the paper you would like to receive in the response
    paper_data_query_params = {
        "fields": "title,year,abstract,url,authors.name,citationCount,year,openAccessPdf"
    }
    headers = {"x-api-key": SS_API_KEY}
    try:
        response = requests.get(url, params=paper_data_query_params, headers=headers)
        # time.sleep(0.1)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def search_paper_via_title(title):
    query_params = {
        "query": title,
        "fields": "title,year,abstract,authors.name,citationCount,year,url,externalIds,corpusId",
    }
    # Define headers with API key
    headers = {"x-api-key": SS_API_KEY}
    # Send the API request
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/match",
            params=query_params,
            headers=headers,
        )
        time.sleep(0.2)
        # Check response status
        if response.status_code == 200:
            response_data = response.json()
        # Process and print the response data as needed
        else:
            response_data = None
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
    except:
        response_data = None
    if response_data is None or len(response_data) == 0 or "data" not in response_data:
        return None
    else:
        return response_data["data"][0]


def batch_paper_data(arxiv_ids):
    headers = {"x-api-key": SS_API_KEY}
    r = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params={
            "fields": "referenceCount,citationCount,title,url,publicationDate,abstract"
        },
        json={"ids": ["ARXIV:{0}".format(id) for id in arxiv_ids]},
        headers=headers,
    )
    time.sleep(1)
    response_data = r.json()
    return {id: data for id, data in zip(arxiv_ids, response_data)}


def batch_paper_data_SS_ID(paper_ids):
    headers = {"x-api-key": SS_API_KEY}
    r = requests.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params={
            "fields": "referenceCount,citationCount,title,url,publicationDate,abstract,year,authors.name"
        },
        json={"ids": ["CorpusId:{0}".format(id) for id in paper_ids]},
        headers=headers,
    )
    time.sleep(0.1)
    response_data = r.json()
    return {id: data for id, data in zip(paper_ids, response_data)}
