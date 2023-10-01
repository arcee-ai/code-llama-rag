import csv
import importlib
import inspect
import argparse
import textwrap
import pkgutil
import tqdm


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
    parser.add_argument(
        "--empty_doc",
        action="store_true",
        help="boolean flag to seperate out the empty doc rows",
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


def extract_classes_and_functions(node):
    class_names = []
    function_names = []

    for name, child in inspect.getmembers(node):
        if inspect.isclass(child):
            try:
                class_names.append(
                    (name, inspect.getdoc(child), inspect.getsource(child))
                )
            except:
                pass
        elif inspect.isfunction(child):
            try:
                function_names.append(
                    (name, inspect.getdoc(child), inspect.getsource(child))
                )
            except Exception as e:
                pass

    return class_names, function_names


def extract(module_obj, already_extracted_modules, tqdm_obj):
    if module_obj.__name__ in already_extracted_modules or not hasattr(
        module_obj, "__file__"
    ):
        return [], already_extracted_modules

    tqdm_obj.set_description(f"Extracting from {module_obj.__name__}")

    already_seen = already_extracted_modules + [module_obj.__name__]

    inner_results = []

    classes_, functions_ = extract_classes_and_functions(module_obj)
    main_results = classes_ + functions_

    for mod_name, mod in inspect.getmembers(module_obj, inspect.ismodule):
        try:
            results_, seen = extract(mod, already_seen, tqdm_obj)
        except Exception as e:
            print(f"Error extracting from {mod_name}")
            print(e)
            continue
        inner_results.extend(results_)
        already_seen = seen

    final_results = []
    for n, doc, source in main_results:
        doc = clean_string(doc or "")
        if source is None:
            continue

        final_results.append(
            {
                "function_name": n,
                "doc": doc,
                "function": source,
            }
        )

    final_results.extend(inner_results)

    return final_results, already_seen


def write_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            row["doc"] = repr(row["doc"])
            row["function"] = repr(row["function"])
            writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()
    all_data = []
    already_extracted_modules = []
    modules = get_all_accessible_modules()
    tqdm_obj = tqdm.tqdm(total=len(modules), dynamic_ncols=True)
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            result, already_extracted_modules = extract(
                module, already_extracted_modules, tqdm_obj
            )
            all_data.extend(result)
            already_extracted_modules = already_extracted_modules + [module_name]
        except:
            print(f"Top level error: importing {module_name}")
            continue
        tqdm_obj.update(1)
    print("Length of all_data: ", len(all_data))

    # get stats on the data
    empty_doc = 0
    for row in all_data:
        if row["doc"] == "":
            empty_doc += 1

    print(
        "Number of rows with empty doc: ",
        empty_doc,
        "which is ",
        empty_doc / len(all_data) * 100,
        "% of the total data",
    )

    if args.empty_doc:
        empty_doc_data = []
        non_empty_doc_data = []
        for row in all_data:
            if row["doc"] == "":
                empty_doc_data.append(row)
            else:
                non_empty_doc_data.append(row)

        filename = args.output.split(".")[0]
        write_to_csv(empty_doc_data, filename + "_empty_docs.csv")
        write_to_csv(non_empty_doc_data, filename + "_non_empty_docs.csv")
    else:
        write_to_csv(all_data, args.output)