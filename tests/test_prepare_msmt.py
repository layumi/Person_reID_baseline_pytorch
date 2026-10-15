import tempfile
import unittest
from pathlib import Path

from prepare_helpers import copy_to_identity_folder


class CopyToIdentityFolderTests(unittest.TestCase):
    def test_validation_files_keep_their_listed_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'source'
            destination = Path(temp_dir) / 'train_all'
            source.mkdir()
            destination.mkdir()
            (source / '0001').mkdir()
            (source / '0002').mkdir()
            (source / '0001' / 'a.jpg').write_bytes(b'validation image 1')
            (source / '0002' / 'b.jpg').write_bytes(b'validation image 2')

            for relative_path in ('0001/a.jpg', '0002/b.jpg'):
                copy_to_identity_folder(source, destination, relative_path)

            self.assertEqual((destination / '0001' / 'a.jpg').read_bytes(), b'validation image 1')
            self.assertEqual((destination / '0002' / 'b.jpg').read_bytes(), b'validation image 2')
            self.assertFalse((destination / '0002' / 'a.jpg').exists())


if __name__ == '__main__':
    unittest.main()
