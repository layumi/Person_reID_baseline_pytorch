import os
from shutil import copyfile


def copy_to_identity_folder(src_root, dst_root, relative_path):
    identity = relative_path.split('/')[0]
    dst_path = os.path.join(dst_root, identity)
    os.makedirs(dst_path, exist_ok=True)
    src_path = os.path.join(src_root, *relative_path.split('/'))
    copyfile(src_path, os.path.join(dst_path, os.path.basename(relative_path)))
