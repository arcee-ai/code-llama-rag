import sys
import os
import csv
import importlib
import inspect
import argparse
import textwrap

import pkgutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump all function info from standard library"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="python_lib_functions.csv",
        help="output csv file name",
    )
    return parser.parse_args()


def clean_string(doc):
    doc = doc.strip()
    doc = textwrap.dedent(doc)
    return "\n".join(" ".join(line.split()) for line in doc.split("\n"))


def get_all_accessible_modules():
    modules = []
    for _, module_name, _ in pkgutil.iter_modules():
        modules.append(module_name)
    return modules


def gather_fs_from_module(module_name):
    try:
        module = importlib.import_module(module_name)
    except:
        return []

    functions = inspect.getmembers(module, inspect.isfunction)
    return [
        {
            "function_name": name,
            "doc": clean_string(obj.__doc__),
            "function": repr(inspect.getsource(obj)),
        }
        for name, obj in functions
    ]


def write_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["function_name", "doc", "function"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()
    all_data = []
    for module_name in get_all_accessible_modules():
        try:
            fs = [
                func
                for func in gather_fs_from_module(module_name)
                if func["doc"] is not None
            ]
            for i in fs:
                i["doc"] = repr(i["doc"])
            all_data.extend(fs)
        except:
            pass

    write_to_csv(all_data, args.output)
