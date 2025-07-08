#!/bin/bash

echo Python: $(which python)
source /Users/deven367/miniforge3/bin/activate
conda activate nixtla

echo Activated env: $(which python)


mkdir -p ../tests

tst_flags='datasets distributed matplotlib polars pyarrow scipy'

nbs=$(ls ../nbs/src/*.ipynb)
# echo "Available notebooks: $nbs"


# this approach3 that was discussed on Slack
# mkdir -p ../tests3
# for flag in $tst_flags; do
#     # echo "Extracting $flag"

#     for nb in $nbs; do
#         # get name of notebook without extension
#         nb_name=$(basename "$nb" .ipynb)

#         # echo "Processing notebook: $nb"
#         # print_dir_in_nb "$nb" --dir_name no_dir_and_dir --dir "$flag" >> "../tests/test_$flag_$nb_name.py"
#         print_dir_in_nb "$nb" --dir_name no_dir_and_dir --dir "$flag" >> "../tests3/test_${flag}_$nb_name.py"
#     done
# done

for nb in $nbs; do
    # get name of notebook without extension
    nb_name=$(basename "$nb" .ipynb)

    # echo "Processing notebook: $nb"
    # print_dir_in_nb "$nb" --dir_name no_dir_and_dir --dir "$flag" >> "../tests/test_$flag_$nb_name.py"
    python cli.py "$nb" --dir_name get_all_tests2 >> "../tests/test_$nb_name.py"
done