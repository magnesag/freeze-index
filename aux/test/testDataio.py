"""!
    DataIO Unittesting
    ==================

    @author A. Schaer
    @copyright Magnes AG, (C) 2024.
"""

import os
import unittest as ut

import matplotlib.pyplot as pltlib

from .. import dataio

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(FILE_DIR, "data")
DAPHNET_TEST_FILE = os.path.join(DATA_DIR, "daphnetset.txt")


class TestDirectoryDataDiscovery(ut.TestCase):
    def test_file_discovery(self):
        cr = dataio.get_files_in_dir(DATA_DIR, ".test")
        er = [os.path.join(DATA_DIR, "nodata.test")]
        self.assertEqual(er, cr)


class TestDaphnetLoading(ut.TestCase):
    def tearDown(self):
        pltlib.close("all")

    def test_laod_daphnet_data(self):
        cr = dataio.load_daphnet_txt(DAPHNET_TEST_FILE)

        fig, axs = pltlib.subplots()
        axs.plot(cr.t, cr.shank_xl.x, c="r")
        axs.plot(cr.t, cr.shank_xl.y, c="g")
        axs.plot(cr.t, cr.shank_xl.z, c="b")
        fig.tight_layout()
