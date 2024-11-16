import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd


def get_acceptance_results(records, target_model_a, target_model_b):
    acceptance_results = {
        target_model_a: {},
        target_model_b: {},
    }
    for record in records:
        instance_id = record.instance_id
        if instance_id not in acceptance_results[record.model_a]:
            acceptance_results[record.model_a][instance_id] = []

        if instance_id not in acceptance_results[record.model_b]:
            acceptance_results[record.model_b][instance_id] = []

    # count how many instances get multiple annotations
    instances_with_multiple_annotations = [
        instance_id
        for instance_id, results in acceptance_results[record.model_a].items()
        if len(results) > 1
    ]
    agreement_results = {
        "num_instances_with_multiple_annotations": len(
            instances_with_multiple_annotations
        ),
        "acceptance_agreement": None,
    }
    assert target_model_a in acceptance_results
    assert target_model_b in acceptance_results
    # get agreement on acceptance
    if len(instances_with_multiple_annotations) > 0:
        agreed_model_a_acceptance = 0
        agreed_model_b_acceptance = 0
        for instance_id in instances_with_multiple_annotations:
            if len(set(acceptance_results[target_model_a][instance_id][:2])) == 1:
                agreed_model_a_acceptance += 1
            if len(set(acceptance_results[target_model_b][instance_id][:2])) == 1:
                agreed_model_b_acceptance += 1
        agreement_results["acceptance_agreement"] = (
            agreed_model_a_acceptance + agreed_model_b_acceptance
        ) / (2 * len(instances_with_multiple_annotations))
        agreement_results[f"{target_model_a}_acceptance_agreement"] = (
            agreed_model_a_acceptance / len(instances_with_multiple_annotations)
        )
        agreement_results[f"{target_model_b}_acceptance_agreement"] = (
            agreed_model_b_acceptance / len(instances_with_multiple_annotations)
        )

    return {
        f"{target_model_a}": sum(
            [
                1 if x[0] in [4, 5] else 0
                for _, x in acceptance_results[target_model_a].items()
            ]
        )
        / len(acceptance_results[target_model_a]),
        f"{target_model_b}": sum(
            [
                1 if x[0] in [4, 5] else 0
                for _, x in acceptance_results[target_model_b].items()
            ]
        )
        / len(acceptance_results[target_model_b]),
        "agreement": agreement_results,
    }


def calculate_avg_fine_grained_scores(results, aspect):
    total_score = 0
    for instance_id in results:
        total_score += np.mean(results[instance_id][aspect])
    return total_score / len(results)


def get_finegrained_results(
    records,
    target_model_a,
    target_model_b,
    aspects=["fluency", "sufficient", "relevant"],
):
    finegrained_results = {
        target_model_a: {},
        target_model_b: {},
    }
    for record in records:
        instance_id = record.instance_id
        try:
            result = int(record[f"{aspects[0]}_model_a"])
            if instance_id not in finegrained_results[record.model_a]:
                finegrained_results[record.model_a][instance_id] = {
                    aspect: [] for aspect in aspects
                }
            if instance_id not in finegrained_results[record.model_b]:
                finegrained_results[record.model_b][instance_id] = {
                    aspect: [] for aspect in aspects
                }
            finegrained_results[record.model_a][instance_id]["useful"] = []
            finegrained_results[record.model_b][instance_id]["useful"] = []

        except:
            print("error in values.")
            continue

        for aspect in aspects:
            finegrained_results[record.model_a][instance_id][aspect].append(
                record[f"{aspect}_model_a"]
            )
            finegrained_results[record.model_b][instance_id][aspect].append(
                record[f"{aspect}_model_b"]
            )
        finegrained_results[record.model_b][instance_id]["useful"].append(
            record["completion_b_is_acceptable"]
        )
        finegrained_results[record.model_a][instance_id]["useful"].append(
            record["completion_a_is_acceptable"]
        )

    fine_grained_results = {
        target_model_a: {
            aspect: calculate_avg_fine_grained_scores(
                finegrained_results[target_model_a], aspect
            )
            for aspect in aspects + ["useful"]
        },
        target_model_b: {
            aspect: calculate_avg_fine_grained_scores(
                finegrained_results[target_model_b], aspect
            )
            for aspect in aspects + ["useful"]
        },
    }

    return fine_grained_results


def get_comparison_results(records, target_model_a, target_model_b):
    comparison_results = {}
    for record in records:
        instance_id = record.instance_id
        model_a = record.model_a
        model_b = record.model_b
        if instance_id not in comparison_results:
            comparison_results[instance_id] = []

        if record.preference == "a-is-better":
            comparison_results[instance_id].append(f"{model_a} is clearly better")
        elif record.preference == "b-is-better":
            comparison_results[instance_id].append(f"{model_b} is clearly better")
        elif record.preference == "tie":
            comparison_results[instance_id].append("tie")
        else:
            print("-------------------------------------")
            print("Unknown preference value.")
            print(record)

    earlies_comparison_results = [
        results[0] for _, results in comparison_results.items() if len(results) > 0
    ]
    model_wins_counter = Counter(earlies_comparison_results)
    model_wins_rates = {
        result: count / len(earlies_comparison_results)
        for result, count in model_wins_counter.items()
    }
    # merge the clearly better and slightly better results
    model_wins_rates[f"{target_model_a}_wins"] = sum(
        [v for k, v in model_wins_rates.items() if target_model_a in k]
    )
    model_wins_rates[f"{target_model_b}_wins"] = sum(
        [v for k, v in model_wins_rates.items() if target_model_b in k]
    )

    # count how many instances get multiple annotations
    instances_with_multiple_annotations = [
        instance_id
        for instance_id, results in comparison_results.items()
        if len(results) > 1
    ]
    agreement_results = {
        "num_instances_with_multiple_annotations": len(
            instances_with_multiple_annotations
        ),
        "comparison_agreement": None,
        "relexed_comparison_agreement": None,
    }
    if instances_with_multiple_annotations:
        agreed_comparison = 0
        relexed_agreed_comparison = 0
        for instance_id in instances_with_multiple_annotations:
            simplified_comparisons = []
            for comparison_result in comparison_results[instance_id]:
                if comparison_result == "tie":
                    simplified_comparisons.append("tie")
                elif target_model_a in comparison_result:
                    simplified_comparisons.append(target_model_a)
                elif target_model_b in comparison_result:
                    simplified_comparisons.append(target_model_b)
                else:
                    print("Unknown comparison result.")
                    print(comparison_result)
            if len(set(simplified_comparisons[:2])) == 1:
                agreed_comparison += 1
                relexed_agreed_comparison += 1
            else:
                if "tie" in simplified_comparisons[:2]:
                    relexed_agreed_comparison += 0.5
        agreement_results["comparison_agreement"] = agreed_comparison / len(
            instances_with_multiple_annotations
        )
        agreement_results["relexed_comparison_agreement"] = (
            relexed_agreed_comparison / len(instances_with_multiple_annotations)
        )

    model_wins_rates["agreement"] = agreement_results
    return model_wins_rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/eval_annotations.xlsx",
    )
    args = parser.parse_args()

    annotations = pd.read_excel(args.data_file, header=0)
    print("Num of annotations: {}".format(len(annotations)))

    instance_annotators = {}
    for record in annotations.iterrows():
        instance_index = record[1]["instance_index"]
        if instance_index not in instance_annotators:
            instance_annotators[instance_index] = []
        annotator = record[1]["evaluator"]
        instance_annotators[instance_index].append(annotator)

    instance_records = {}
    for record in annotations.iterrows():
        instance_index = record[1]["instance_index"]
        if instance_index not in instance_records:
            instance_records[instance_index] = []
        instance_records[instance_index].append(record[1])

    print("Removing duplicate records from the same evaluator...")
    for instance_index, records in instance_records.items():
        records = sorted(records, key=lambda x: x["timestamp"], reverse=True)
        evaluators = set()
        new_records = []
        for record in records:
            if record["evaluator"] not in evaluators:
                evaluators.add(record["evaluator"])
                new_records.append(record)
            else:
                print(
                    "duplicate record for instance {} by evaluator {}".format(
                        instance_index, record["evaluator"]
                    )
                )
        instance_records[instance_index] = new_records
    deduplicated_records = []
    for instance_index, records in instance_records.items():
        for record in records:
            deduplicated_records.append(record)

    deduplicated_records = sorted(deduplicated_records, key=lambda x: x["timestamp"])
    print("Num of deduplicated records: {}".format(len(deduplicated_records)))

    model_pairs = set()
    for record in deduplicated_records:
        model_pair = tuple(sorted([record["model_a"], record["model_b"]]))
        model_pairs.add(model_pair)
    print("Model pairs:")
    for model_pair in model_pairs:
        print(f"{model_pair[0]} vs {model_pair[1]}")

    results = {}
    for target_model_a, target_model_b in model_pairs:
        comparison_records = []
        for record in deduplicated_records:
            instance_id = record.instance_id

            if set([target_model_a, target_model_b]) != set(
                [record.model_a, record.model_b]
            ):
                assert any(
                    [
                        set([record.model_a, record.model_b]) == set(pair)
                        for pair in model_pairs
                    ]
                )
                continue

            comparison_records.append(record)

        comparison_results = get_comparison_results(
            comparison_records, target_model_a, target_model_b
        )
        fine_grained_results = get_finegrained_results(
            comparison_records, target_model_a, target_model_b
        )
        results[f"{target_model_a}_vs_{target_model_b}"] = {
            "comparison_results": comparison_results,
            "fine_grained_results": fine_grained_results,
        }
    print("Results:")
    for model_pair, result in results.items():
        print(model_pair)
        print(json.dumps(result, indent=4))
