#!/bin/bash

# Run each command and capture its return value
commands=(
    "tests/scripts/functional_tests/test_task.sh --type train --task aquila"
#     # TODO: need fix
#     # "tests/scripts/functional_tests/test_task.sh --type train --task deepseek"
#     # "tests/scripts/functional_tests/test_task.sh --type train --task mixtral"
#     # "tests/scripts/functional_tests/test_task.sh --type train --task llava_onevision"
#     # for hetero-train
#     # "tests/scripts/functional_tests/test_task.sh --type hetero_train --task aquila"
#     # Add in the feature
    # "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek"
    # "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3"
    # "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek_flaggems"
    # "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3_flaggems"
#     # For serve
#     # "tests/scripts/functional_tests/test_task.sh --type serve --task base"
)

if [ ! -f tests/functional_runtime.txt ];then
    touch tests/functional_runtime.txt
fi
echo -e "\n********************************" >> tests/functional_runtime.txt
echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/functional_runtime.txt
for cmd in "${commands[@]}"; do
    # Execute the command
    $cmd
    # Capture the return value
    return_value=$?
    # Check if the return value is non-zero (an error occurred)
    if [ $return_value -ne 0 ]; then
        # Output the error message and the failing command
        echo "Error: Command '$cmd' failed"
        # Throw an exception by exiting the script with a non-zero status
        exit 1
    fi
done
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/functional_runtime.txt