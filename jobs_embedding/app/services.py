from app.sqlite3db import SQLPostingDatabase

driver = SQLPostingDatabase()


def format_payload(jobs):
    payload = {"job_id": [], "job_title": [], "job_description": []}

    for job in jobs:
        payload["job_id"].append(job["id"])
        payload["job_title"].append(job["title"])
        payload["job_description"].append(job["description"])

    return payload


def get_jobs_data(text=None):
    if text:
        jobs = driver.search(text)
    else:
        jobs = driver.get_all()

    return format_payload(jobs)
