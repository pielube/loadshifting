import sys
import os
import unittest

# Set root folder one level up, just for this example
sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')))


class TestBuildingSim(unittest.TestCase):

    def test_annualSimulation(self):
        runfile = open(os.path.join('examples', 'annualSimulation.py'))
        exec(runfile.read())
        runfile.close()

    def test_annualSimulation_importRadiation(self):
        runfile = open(os.path.join('examples', 'annualSimulation_importRadiation.py'))
        exec(runfile.read())
        runfile.close()

    def test_calculateRadiation(self):
        runfile = open(os.path.join('examples', 'calculateRadiation.py'))
        exec(runfile.read())
        runfile.close()

    def test_hourSimulation(self):
        runfile = open(os.path.join('examples', 'hourSimulation.py'))
        exec(runfile.read())
        runfile.close()

    def test_sunAngles(self):
        runfile = open(os.path.join('examples', 'sunAngles.py'))
        exec(runfile.read())
        runfile.close()


if __name__ == '__main__':
    unittest.main()
