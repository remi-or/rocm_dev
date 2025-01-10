from typing import Dict, Any
import os
import os.path as osp
import shutil
import subprocess
import json
from hashlib import md5


PROJECTS_PATH = osp.join(osp.dirname(__file__), "projects")
BACKUPS_PATH = osp.join(osp.dirname(__file__), "backups")
DATA_PATH = osp.join(osp.dirname(__file__), "data")


def stringify_number(x: int, digits: int = 3) -> str:
    s = str(x)
    assert len(s) < digits, f"Number {x} has more than {digits = }"
    return "0" * (digits - len(s)) + s

def get_hash(file_path: str) -> str:
    if not osp.isfile(file_path):
        return ""
    m = md5()
    with open(file_path, "rb") as f:
        data = f.read()
    m.update(data)
    return m.hexdigest()

def temp_extension(file: str) -> bool:
    for ext in [".bc", ".hipi", ".hipfb", ".o", ".s", ".out", "out.resolution.txt"]:
        if file.endswith(ext):
            return True
    return False


class Project:

    def __init__(self, name: str) -> None:
        self.name = name
        self.project_dir = osp.join(PROJECTS_PATH, name)
        assert osp.exists(self.project_dir), f"Directory {self.project_dir} does not exist"
        self.prepare_backup()

    def prepare_backup(self) -> None:
        self.backup_dir = osp.join(BACKUPS_PATH, self.name)
        # Make sure the backup directory exists
        if not osp.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        # Create new backup
        self.backup_nb = len([file for file in os.listdir(self.backup_dir) if file[-4:] == ".zip"])

    def compile_and_run(self, c_file: str, arguments: str = "") -> str:
        file_path = osp.join(self.project_dir, c_file)
        name = c_file.split(".")[0]
        subprocess.check_call(f"hipcc -O3 {file_path} -save-temps -o {name}.out".split())
        new_files = list(filter(temp_extension, os.listdir(".")))
        shutil.rmtree("./build")
        os.mkdir("./build")
        for file in new_files:
            shutil.move(file, f"./build/{file}")
        return str(subprocess.check_output(f"./build/{name}.out {arguments}".split()))[2:-1]
    
    def backup(self) -> str:
        """Backups the current project and returns the hash of the backup."""
        # Create a temporary backup
        tmp_backup_no_ext = osp.join(self.backup_dir, "tmp")
        shutil.make_archive(tmp_backup_no_ext, "zip", self.project_dir)
        # Get its hash
        hash_ = get_hash(tmp_backup_no_ext + ".zip")
        # Rename the backup if its new or delete it if it isn't
        hashed_backup = osp.join(self.backup_dir, hash_ + ".zip")
        if osp.exists(hashed_backup):
            os.remove(tmp_backup_no_ext + ".zip")
        else:
            os.rename(tmp_backup_no_ext + ".zip", hashed_backup)
        return hash_

    def record(self, hash_: str, arguments: str, data: Dict[str, Any]):
        # Make sure data dir exists
        data_dir = osp.join(DATA_PATH, self.name)
        os.makedirs(data_dir, exist_ok=True)
        # Retrieve the current record
        record_path = osp.join(data_dir, f"{hash_}.json")
        if osp.exists(record_path):
            with open(record_path) as file:
                record = json.load(file)
        else:
            record = {}
        # Add new entry
        if arguments not in record:
            record[arguments] = []
        record[arguments].append(data)
        # Save new record
        with open(record_path, "w") as file:
            json.dump(record, file)
