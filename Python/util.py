"""
File containing utility functions
"""
import os


def unique_file(dest_path):
    """
    Iterative increase the number on a file name to generate a unique file name
    :param dest_path: Original file name, which may already exist
    :return: Unique file name with appended index, for uniqueness
    """
    index = ""
    # Check if the folder name already exists
    while os.path.exists(dest_path + index):
        if index:
            index = "({})".format(str(int(index[1:-1]) + 1))
        else:
            index = "(1)"

    return dest_path + index
