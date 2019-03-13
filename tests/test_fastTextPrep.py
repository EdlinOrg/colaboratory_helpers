
import sys
sys.path.append("colaboratory_helpers")

import fastTextPrep

import unittest

class TestFastTextPrep(unittest.TestCase):

    def test_brutal(self):

        self.assertEqual('apa tt b123', fastTextPrep.brutal(" apa tt. b123  "))
        self.assertEqual('apa b123', fastTextPrep.brutal(" apa t. b123  4"))
        self.assertEqual('apa b123', fastTextPrep.brutal(" apa t. b123  4 1999"))

        stopdict = {
            'tennis': True,
            'golf': True
        }

        self.assertEqual('apa b123', fastTextPrep.brutal(" apa t. b123  golf 4", stopwordsdict=stopdict))

if __name__ == '__main__':
    unittest.main()
