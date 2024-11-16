import argparse
import json
import logging
import os
import re
import time

import requests
from fuzzy_match import algorithims
from nltk import sent_tokenize
from tqdm import tqdm
from use_search_apis import (
    batch_paper_data_SS_ID,
    get_paper_data,
    search_paper_via_title,
)
from utils import save_file_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
input_prompt = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"

global claim_autoais_model, claim_autoais_tokenizer
claim_autoais_model, claim_autoais_tokenizer = None, None

import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

OSU_AUTOAIS_MODEL = "osunlp/attrscore-flan-t5-xl"

ss_data = {}


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-6}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def remove_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r"\[\d+(?:,\s*\d+)*\]"
    # Remove all citations from the text
    cleaned_text = re.sub(citation_pattern, "", text)
    # Optionally, remove extra spaces that might result from removing citations
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    cleaned_text = cleaned_text.replace(" .", ".")
    cleaned_text = cleaned_text.replace(" ,", ",")
    return cleaned_text


def normalize_citations(text):
    # Regular expression to find citations like [1, 2, 3]
    citation_pattern = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

    def replace_citations(match):
        # Split the numbers by comma, strip spaces, and wrap each in square brackets
        citations = match.group(1).split(",")
        normalized_citations = "".join(f"[{num.strip()}]" for num in citations)
        return normalized_citations

    # Substitute the original citation format with the normalized one
    normalized_text = citation_pattern.sub(replace_citations, text)
    return normalized_text


def check_paper_details(title):

    query_params = {
        "query": title,
        "limit": 5,
        "fields": "title,year,abstract,authors.name,citationCount,url",
    }
    api_key = "iQz0z2jFsDa6aeKGWnmR62DjXHbrm92b6AsFdDPL"
    # Define headers with API key
    headers = {"x-api-key": api_key}
    # Send the API request
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=query_params,
            headers=headers,
        )
        # Check response status
        if response.status_code == 200:
            response_data = response.json()
        # Process and print the response data as needed
        else:
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
    except:
        response_data = []
    if len(response_data) == 0 or "data" not in response_data:
        return None
    else:
        fuzzy_math_scores = [
            paper
            for paper in response_data["data"]
            if algorithims.trigram(title.lower(), paper["title"].lower()) > 0.95
        ]
        if len(fuzzy_math_scores) == 0:
            return None
        else:
            return fuzzy_math_scores[0]


def extract_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    # Find all matches in the text
    matches = re.findall(citation_pattern, text)
    # Extract individual numbers and convert them to integers
    citations = []
    for match in matches:
        # Split by commas, strip any extra whitespace, and convert to integers
        citations.extend([int(num.strip()) for num in match.split(",")])
    return citations


def process_references(output, ctxs):
    global ss_data
    citations = extract_citations(output)
    used_ctxs = [(i, c) for i, c in enumerate(ctxs) if i in citations]
    reference_text = ""
    reference_list = []
    author_id_count = {}
    ref_to_author_ids = {}
    for ctx in used_ctxs:
        if "semantic_scholar_id" not in ctx[1] and "url" in ctx[1]:
            ctx[1]["semantic_scholar_id"] = ctx[1]["url"].split(
                "https://www.semanticscholar.org/paper/"
            )[1]
        elif "semantic_scholar_id" not in ctx[1] and "url" not in ctx[1]:
            title = ctx[1]["title"]
            print(title)
            paper_data = search_paper_via_title(title)

            ctx[1]["semantic_scholar_id"] = paper_data["corpusId"]

    ss_ids = [
        c[1]["semantic_scholar_id"]
        for c in used_ctxs
        if "semantic_scholar_id" in c[1] and c[1]["semantic_scholar_id"] not in ss_data
    ]
    ss_ids = list(set(ss_ids))
    paper_data = batch_paper_data_SS_ID(ss_ids)
    ss_data.update(paper_data)
    for c in tqdm(used_ctxs):
        if (
            "semantic_scholar_id" in c[1]
            and c[1]["semantic_scholar_id"] in ss_data
            and ss_data[c[1]["semantic_scholar_id"]] is not None
            and type(ss_data[c[1]["semantic_scholar_id"]]) is dict
        ):
            author = (
                ss_data[c[1]["semantic_scholar_id"]]["authors"][0]["name"]
                if len(ss_data[c[1]["semantic_scholar_id"]]["authors"]) > 0
                else "Unknown"
            )
            year = ss_data[c[1]["semantic_scholar_id"]]["year"]
        else:
            if "url" in c[1]:
                id = c[1]["url"].split("https://www.semanticscholar.org/paper/")[1]
                data = batch_paper_data_SS_ID([id])[id]
                if type(data) is dict:
                    ss_data[id] = data
                    author = ss_data[c[1]["semantic_scholar_id"]]["authors"][0]["name"]
                    year = ss_data[c[1]["semantic_scholar_id"]]["year"]
                else:
                    author, year = None, None
            else:
                author, year = None, None
        if author is None:
            if "title" in c[1]:
                author_id = c[1]["title"][:10]
            else:
                author_id = "Unknown"
        else:
            author_id = "{0} et al. {1}".format(author, year)
        author_id_count.setdefault(author_id, 0)
        author_id_count[author_id] += 1
    dup_author_ids = [
        author_id for author_id in author_id_count if author_id_count[author_id] > 1
    ]
    author_id_count = {}

    for c in tqdm(used_ctxs):
        if (
            "semantic_scholar_id" not in c[1]
            or c[1]["semantic_scholar_id"] not in ss_data
            or type(ss_data[c[1]["semantic_scholar_id"]]) is not dict
        ):
            if "title" in c[1]:
                author_id = c[1]["title"][:10]
            else:
                author_id = "unknown"
        else:
            author = (
                ss_data[c[1]["semantic_scholar_id"]]["authors"][0]["name"]
                if len(ss_data[c[1]["semantic_scholar_id"]]["authors"]) > 1
                else "Unknown"
            )
            year = ss_data[c[1]["semantic_scholar_id"]]["year"]
            author_id = "{0} et al. {1}".format(author, year)
        if author_id in dup_author_ids:
            author_id_count.setdefault(author_id, 0)
            author_id_count[author_id] += 1
            context_uid = author_id + " - {}".format(author_id_count[author_id])
        else:
            context_uid = author_id
        ref_to_author_ids[c[0]] = context_uid
        reference_list.append(
            {
                "title": c[1]["title"] if "title" in c[1] else "",
                "text": c[1]["text"],
                "id": context_uid,
                "url": c[1]["url"] if "url" in c[1] else "",
            }
        )
    print(ref_to_author_ids)

    return reference_text, reference_list, ref_to_author_ids


def process_references_fixed(output, ctxs):
    global ss_data
    citations = extract_citations(output)
    used_ctxs = [(i, c) for i, c in enumerate(ctxs) if i in citations]
    reference_text = ""
    reference_list = []
    author_id_count = {}
    ref_to_author_ids = {}
    for ctx in used_ctxs:
        if type(ctx) is float or type(ctx[1]) is float:
            continue

        if "semantic_scholar_id" not in ctx[1]:
            if (
                "url" in ctx[1]
                and type(ctx[1]["url"]) is str
                and "https://www.semanticscholar.org/paper" in ctx[1]["url"]
            ):
                ctx[1]["semantic_scholar_id"] = ctx[1]["url"].split(
                    "https://www.semanticscholar.org/paper/"
                )[1]

            else:
                title = ctx[1]["title"]
                if "Title: " in title:
                    title = title.split("Title: ")[1].split("\nText:")[0]
                paper_data = search_paper_via_title(title)
                if paper_data is not None:
                    ctx[1]["semantic_scholar_id"] = str(paper_data["corpusId"])
                else:
                    print(title)
                    ctx[1]["semantic_scholar_id"] = "000"

        if "authors" not in ctx[1]:
            if ctx[1]["semantic_scholar_id"] not in ss_data:
                paper_data = get_paper_data(ctx[1]["semantic_scholar_id"])
                ss_data[ctx[1]["semantic_scholar_id"]] = paper_data
            else:
                paper_data = ss_data[ctx[1]["semantic_scholar_id"]]

            if paper_data is None:
                time.sleep(0.3)
                paper_data = get_paper_data(ctx[1]["semantic_scholar_id"])
                if paper_data is None:
                    paper_data = {}
                    paper_data["authors"] = None
                    paper_data["year"] = "Unknown"
                    paper_data["title"] = "Unknown"
                    paper_data["url"] = "Unknown"
            author = (
                paper_data["authors"][0]["name"]
                if paper_data["authors"] is not None
                and len(paper_data["authors"]) > 0
                and paper_data["authors"][0] is not None
                else "Unknown"
            )
            year = paper_data["year"]
            author_id = "{0} et al. {1}".format(author, year)
            ctx[1]["title"] = paper_data["title"]
            ctx[1]["url"] = paper_data["url"]

        else:
            author = (
                ctx[1]["authors"][0]["name"]
                if ctx[1]["authors"] is not None
                and len(ctx[1]["authors"]) > 0
                and ctx[1]["authors"][0] is not None
                else "Unknown"
            )
            if author == "Unknown":
                print(ctx[1])
            year = ctx[1]["year"]
            author_id = "{0} et al. {1}".format(author, year)
        ctx[1]["author_id"] = author_id
        author_id_count.setdefault(author_id, 0)
        author_id_count[author_id] += 1

    dup_author_ids = [
        author_id for author_id in author_id_count if author_id_count[author_id] > 1
    ]
    author_id_count = {}

    for c in tqdm(used_ctxs):
        author_id = c[1]["author_id"]
        if author_id in dup_author_ids:
            author_id_count.setdefault(author_id, 0)
            author_id_count[author_id] += 1
            context_uid = author_id + " - {}".format(author_id_count[author_id])
        else:
            context_uid = author_id
        ref_to_author_ids[c[0]] = context_uid
        reference_list.append(
            {
                "title": c[1]["title"] if "title" in c[1] else "",
                "text": c[1]["text"],
                "id": context_uid,
                "url": c[1]["url"] if "url" in c[1] else "",
            }
        )

    return reference_text, reference_list, ref_to_author_ids


def _run_nli_autoais_osu_three_way(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global claim_autoais_model, claim_autoais_tokenizer
    input_text = input_prompt.format_map({"output": passage, "claim": claim})
    input_ids = claim_autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(
        claim_autoais_model.device
    )
    with torch.inference_mode():
        outputs = claim_autoais_model.generate(input_ids, max_new_tokens=10)
    result = claim_autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result == "Attributable":
        inference = 1.0
    elif result == "Extrapolatory":
        inference = 0.5
    else:
        inference = -10
    return inference


def _format_document(doc):
    """Format document for AutoAIS."""

    if "sent" in doc:
        # QA-extracted docs
        return "Title: %s\n%s" % (doc["title"], doc["sent"])
    else:
        # if "title" in doc:
        #     return "Title: %s\n%s" % (doc['title'], doc['text'])
        # else:
        return "Title: %s\n%s" % (doc["title"], doc["text"])


def pred_citations(output, citations):
    sents = sent_tokenize(output)
    target_sents = [remove_citations(sent).strip() for sent in sents]

    entail = 0
    entail_prec = 0
    total_citations = 0
    sent_final_results = []
    # previous_citations = None
    for sent_id, sent in enumerate(sents):
        target_sent = target_sents[
            sent_id
        ]  # Citation removed and (if opted for) decontextualized
        joint_entail = -1  # Undecided

        # Find references
        # ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
        ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
        # if len(ref) == 0 and previous_citations is not None:
        #     ref = previous_citations
        if len(ref) == 0:
            # No citations
            joint_entail = -1
        elif any([ref_id >= len(citations) for ref_id in ref]):
            # Citations out of range
            joint_entail = -1
        else:
            # previous_citations = ref
            ref = ref[:3]
            total_citations += len(ref)
            joint_passage = "\n".join(
                [
                    _format_document(citations[psgs_id])
                    for psgs_id in ref
                    if psgs_id >= 0
                    and type(_format_document(citations[psgs_id])) is str
                ]
            )
            joint_entail = _run_nli_autoais_osu_three_way(joint_passage, target_sent)
            if joint_entail == 1:
                entail += 1

        sent_final_results.append((sent, joint_entail))
    citation_recall = entail / len(sents) if len(sents) > 0 else 0.0
    return sent_final_results, citation_recall


def process_output_refs(output_text, ref_to_uid):
    output_text = normalize_citations(output_text)
    for ref_id in ref_to_uid:
        output_text = output_text.replace(
            "[{0}]".format(ref_id),
            '<span style="color:blue">[{0}]</span>'.format(ref_to_uid[ref_id]),
        ).replace("\n\n", "\n")
    return output_text


def process_output_refs_citations(output_text, ref_to_uid, pred_citations_results):
    final_outputs = ""
    total_sentences = 0
    for sent_score in pred_citations_results:
        if type(sent_score) is None:
            continue
        sent = sent_score[0]
        score = sent_score[1]
        if type(sent) is not str or len(sent) == 0:
            continue

        output_text = normalize_citations(sent)
        for ref_id in ref_to_uid:
            output_text = output_text.replace(
                "[{0}]".format(ref_id),
                '<span style="color:blue">[{0}]</span>'.format(ref_to_uid[ref_id]),
            ).replace("\n\n", "\n")
        if score == 1.0:
            output_text = '<span style="background-color:#bef3c5;">{0}</span> '.format(
                output_text
            )
        elif score == -1:
            output_text = "{0} ".format(output_text)
        elif score == -10:
            output_text = '<span style="background-color:#fcbfb6;">{0}</span> '.format(
                output_text
            )
        elif score == 0.5:
            output_text = '<span style="background-color:#ffe6b3;">{0}</span> '.format(
                output_text
            )
        final_outputs += output_text
    return final_outputs


def process_output_refs_citations_fix(output_text, ref_to_uid, pred_citations_results):
    final_outputs = output_text
    total_sentences = 0
    for sent_score in pred_citations_results:
        if type(sent_score) is None:
            continue
        sent = sent_score[0]
        score = sent_score[1]
        if type(sent) is not str or len(sent) == 0:
            continue

        output_text = normalize_citations(sent)
        for ref_id in ref_to_uid:
            output_text = output_text.replace(
                "[{0}]".format(ref_id),
                '<span style="color:blue">[{0}]</span>'.format(ref_to_uid[ref_id]),
            ).replace("\n\n", "\n")
        if score == 1.0:
            output_text = '<span style="background-color:#bef3c5;">{0}</span> '.format(
                output_text
            )
        elif score == -1:
            output_text = "{0} ".format(output_text)
        elif score == -10:
            output_text = '<span style="background-color:#fcbfb6;">{0}</span> '.format(
                output_text
            )
        elif score == 0.5:
            output_text = '<span style="background-color:#ffe6b3;">{0}</span> '.format(
                output_text
            )
        final_outputs = final_outputs.replace(sent, output_text)
    return final_outputs


def main(args):
    data_a = (
        json.load(open(args.file_a))["data"]
        if type(json.load(open(args.file_a))) is dict
        else json.load(open(args.file_a))
    )
    data_b = json.load(open(args.file_b))["data"]

    print(len(data_a))
    print(len(data_b))
    if len(data_a) != len(data_b):
        data_b_matched = []
        data_a_matched = []
        data_b_q2aswer = {item["input"]: item for item in data_b}
        for item in data_a:
            if item["input"] not in data_b_q2aswer:
                print(item["input"])
                continue
            data_b_instance = data_b_q2aswer[item["input"]]
            item["output"] = item["ouput"]
            data_b_matched.append(data_b_instance)
            data_a_matched.append(item)
        data_b = data_b_matched
        data_a = data_a_matched
    assert len(data_a) == len(data_b)

    print(data_a[0])

    total_citation_recall = 0
    # assert len(data_a) == len(data_b)
    processed_data = []
    for instance_id, (instance_a, instance_b) in tqdm(enumerate(zip(data_a, data_b))):

        if instance_a["input"] != instance_b["input"]:
            print("Question mismatch")
            continue
        instance_a["output"] = normalize_citations(instance_a["output"])
        instance_b["output"] = normalize_citations(instance_b["output"])
        input_query = instance_a["input"]
        processed_refs_a, processed_refs_a_list, ref_to_uid_a = (
            process_references_fixed(instance_a["output"], instance_a["ctxs"])
        )
        print("model b")
        processed_refs_b, processed_refs_b_list, ref_to_uid_b = (
            process_references_fixed(instance_b["output"], instance_b["ctxs"])
        )
        print(len(processed_refs_b_list))
        if args.add_citation_prediction is True:
            global claim_autoais_model, claim_autoais_tokenizer
            if claim_autoais_model is None:
                logger.info("Loading Claims AutoAIS model...")
                claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
                    OSU_AUTOAIS_MODEL,
                    torch_dtype=torch.bfloat16,
                    max_memory=get_max_memory(),
                    device_map="auto",
                )
                claim_autoais_tokenizer = AutoTokenizer.from_pretrained(
                    OSU_AUTOAIS_MODEL, use_fast=False
                )

            pred_citations_results_a, citation_recall_a = pred_citations(
                instance_a["output"], instance_a["ctxs"]
            )
            pred_citations_results_b, citation_recall_b = pred_citations(
                instance_b["output"], instance_b["ctxs"]
            )
            processed_output_completion_a_citations = process_output_refs_citations_fix(
                instance_a["output"], ref_to_uid_a, pred_citations_results_a
            )
            total_citation_recall += citation_recall_a
            processed_output_completion_b_citations = process_output_refs_citations_fix(
                instance_b["output"], ref_to_uid_b, pred_citations_results_b
            )
            processed_data.append(
                {
                    "instance_id": (
                        "{0}_{1}".format(args.prefix, instance_id)
                        if "id" not in instance_a
                        else instance_a["id"]
                    ),
                    "prompt": input_query,
                    "completions": [
                        {
                            "model": args.model_a,
                            "completion": processed_output_completion_a_citations,
                            "refs": processed_refs_a,
                            "refs_list": processed_refs_a_list,
                        },
                        {
                            "model": args.model_b,
                            "completion": processed_output_completion_b_citations,
                            "refs": processed_refs_b,
                            "refs_list": processed_refs_b_list,
                        },
                    ],
                    "scores_a": (
                        instance_a["scores"] if "scores" in instance_a else None
                    ),
                    "scores_b": (
                        instance_b["scores"] if "scores" in instance_b else None
                    ),
                    "decisions_a": (
                        instance_a["decisions"] if "scores" in instance_a else None
                    ),
                    "decisions_b": (
                        instance_b["decisions"] if "scores" in instance_b else None
                    ),
                }
            )

        else:
            processed_output_completion_a = process_output_refs(
                instance_a["output"], ref_to_uid_a
            )
            processed_output_completion_b = process_output_refs(
                instance_b["output"], ref_to_uid_b
            )
            processed_data.append(
                {
                    "instance_id": (
                        "{0}_{1}".format(args.prefix, instance_id)
                        if "id" not in instance_a
                        else instance_a["id"]
                    ),
                    "prompt": input_query,
                    "completions": [
                        {
                            "model": args.model_a,
                            "completion": processed_output_completion_a,
                            "refs": processed_refs_a,
                            "refs_list": processed_refs_a_list,
                        },
                        {
                            "model": args.model_b,
                            "completion": processed_output_completion_b,
                            "refs": processed_refs_b,
                            "refs_list": processed_refs_b_list,
                        },
                    ],
                }
            )

    print(total_citation_recall / len(data_a))
    print(len(data_a))
    print(len(processed_data))
    save_file_jsonl(processed_data, args.output_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_a",
        required=True,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--file_b",
        required=True,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_a",
        required=True,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_b",
        required=True,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_fn",
        required=True,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="nq",
    )

    parser.add_argument("--add_citation_prediction", action="store_true", default=None)

    args = parser.parse_args()
    main(args)
