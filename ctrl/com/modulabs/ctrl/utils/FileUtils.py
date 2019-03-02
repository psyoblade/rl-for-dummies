import os
import shutil


class FileUtils:

    @staticmethod
    def exists_file(filename):
        return os.path.exists(filename)

    @staticmethod
    def remove_file(filename):
        if os.path.isfile(filename):
            os.remove(filename)

    @staticmethod
    def remove_dir(dirname):
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
