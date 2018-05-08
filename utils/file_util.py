import fnmatch
import os
import sys

import shutil
import logging

if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle

import errno


class FileUtil:
    """
    Contains methods that manipulates files, such as inserting and deleting files
    """
    from functools import partial
    open_utf8 = partial(open, encoding='utf-8')

    @staticmethod
    def validate_file(file_path):
        """
        Method to ensure that the file and exists at the path
        If it doesn't exist it will create a blank file and
        the necessary folders at the path provided.
        :param file_path: Path to the file to be validated
        :return: None
        """

        if not os.path.isfile(file_path):
            try:
                # Create the directories if they do not alread exist
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            file = open(file_path, 'w+')
            file.close()

    @staticmethod
    def validate_folder(folder_path):
        """
        Same as validate_file but for folders
        :param folder_path: Path to the folder
        :return: Nonet
        """
        if not os.path.isdir(folder_path):
            try:
                os.makedirs(os.path.abspath(folder_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    @staticmethod
    def is_folder(folder_path):
        return os.path.isdir(folder_path)

    @staticmethod
    def is_file(file_path):
        return os.path.isfile(file_path)

    @staticmethod
    def find_files(base: str, pattern) -> []:
        '''Return list of files matching pattern in base folder.'''
        return [n for n in fnmatch.filter(os.listdir(base), pattern) if
                os.path.isfile(os.path.join(base, n))]

    @classmethod
    def clear_folder(cls, folder_path):
        """
        Clears all files at specified folder path
        :param folder_path:
        :return: None
        """
        cls.validate_folder(folder_path=folder_path)
        shutil.rmtree(path=folder_path, ignore_errors=True)
        cls.validate_folder(folder_path=folder_path)


    @classmethod
    def delete_file(cls, file_path):
        if os.path.exists(file_path) and cls.is_file(file_path=file_path):
            os.remove(path=file_path)

    @staticmethod
    def delete_folder(dir_name):
        try:
            shutil.rmtree(path=dir_name)
        except FileNotFoundError:
            logging.info("[delete_folder] {} does not exist. Nothing needs to be deleted.".format(dir_name))
            return

    @staticmethod
    def dump_object(obj, file_path):
        """
        Dumps an object into the a file at the path provided
        :param obj: Object that needs to be dumped
        :param file_path: Path of target pickle file
        :return:
        """
        FileUtil.validate_file(file_path)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object(file_path):

        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        return obj

    @classmethod
    def assert_file_type_exists(cls, file_dir: str, file_type_suffix: str):
        """
        Check if a given file_type exists in a directory. If not, throws an error
        Useful for input file checking.
        :param file_dir:
        :param file_type_suffix: such as ".pdf"
        :return:
        """
        count = 0
        files = os.listdir(path=file_dir)
        for file in files:
            if file_type_suffix in file:
                count += 1
        if count == 0:
            raise FileNotFoundError(
                "[check_file_type_exists] {} can not be found in directory {}".format(file_type_suffix, file_dir))




class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
