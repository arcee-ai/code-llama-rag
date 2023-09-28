## Contents

`scripts` directory has two scripts

1. `python_funcs_info_dump.py` looks at the python modules available to the runtime and makes a csv of each modules's functions (certain filters are applied)
2. `generate_qa.py` uses the csv generated from the above (made available via a commandline arg) and generates the required q&a file with four columns
