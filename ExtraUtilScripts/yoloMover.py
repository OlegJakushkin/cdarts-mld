import concurrent.futures
import os
import pathlib
import shutil
from pathlib import Path

with concurrent.futures.ProcessPoolExecutor() as executor:
    # Get a list of files to process
    pathlist = Path('/headless/tmp/s/ftrain/').glob('**/*.jp*')
    i = 0
    # Process the list of files, but split the work across the process pool to use all CPUs!
    for image_file, thumbnail_file in zip(pathlist, executor.map(process, pathlist)):
        pass

def process(path_f):
    to = Path('/headless/tmp/yolo_t/')
    f_name = Path(path_f).stem
    path=os.path.dirname(path_f)
    class_name = os.path.basename(path)
    new_f_name = f_name+ "_" + class_name + ".png"
    shutil.copy2(path_f, to + new_f_name)
    pathlib.Path('train.list').write_text(new_f_name)
