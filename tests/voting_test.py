import unittest

import numpy as np

from ..Voting import voting


class Test_Voting_Methods(unittest.TestCase):
    def test_borda(self):
        feat_ranking = np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]])
        self.assertEqual(voting.voting_borda(feat_ranking), [1, 2, 3])

    def test_STV(self):
        return

    def test_kemeny_young(self):
        # 0 - memphis, M4t3jG4zd4+1 - nashville, 2 - chattanooga, 3- knoxville.
        arr_1 = np.tile(np.array([0, 1, 2, 3]), (42, 1))
        arr_2 = np.tile(np.array([1, 2, 3, 0]), (26, 1))
        arr_3 = np.tile(np.array([2, 3, 1, 0]), (15, 1))
        arr_4 = np.tile(np.array([3, 2, 1, 0]), (17, 1))
        arr_1 = arr_1.tolist()
        arr_2 = arr_2.tolist()
        arr_3 = arr_3.tolist()
        arr_4 = arr_4.tolist()
        arr = arr_1 + arr_2 + arr_3 + arr_4
        feat_ranking = np.array(arr)
        feat_ranking = voting.voting_kemeny_young(feat_ranking, 4).astype(int).tolist()
        self.assertEqual(feat_ranking, [1, 2, 3, 0])

    def test_min(self):
        arr = np.array([[1, 3, 0, 2, 7, 4, 5, 6], [2, 4, 0, 1, 3, 5, 6, 7], [3, 5, 1, 0, 2, 4, 6, 7]])
        feat_ranking = voting.voting_min(arr, 8).astype(int).tolist()
        self.assertEqual(feat_ranking, [2, 3, 0, 4, 1, 5, 6, 7])

    def test_max(self):
        arr = np.array([[1, 3, 0, 2, 7, 4, 5, 6], [2, 4, 0, 1, 3, 5, 6, 7], [3, 5, 1, 0, 2, 4, 6, 7]])
        feat_ranking = voting.voting_max(arr, 8).astype(int).tolist()
        self.assertEqual(feat_ranking, [2, 3, 0, 1, 5, 6, 4, 7])

    def test_mean(self):
        arr = np.array([[1, 2, 4, 3, 0], [2, 3, 0, 1, 4], [4, 2, 1, 0, 3]])
        feat_ranking = voting.voting_mean(arr, 5).astype(int).tolist()
        self.assertEqual(feat_ranking, [3, 2, 0, 1, 4])

    def test_median(self):
        arr = np.array([[1, 2, 4, 3, 0], [2, 3, 0, 1, 4], [4, 2, 1, 0, 3]])
        feat_ranking = voting.voting_median(arr, 5).astype(int).tolist()
        self.assertEqual(feat_ranking, [2, 3, 0, 1, 4])

    def test_plurality(self):
        arr = np.array([[2.2, 3, 4, 5, 1], [3, 4, 1, 2.2, 5], [4, 3, 1, 5, 2.2]])
        feat_ranking = voting.voting_plurality(arr, 5)
        self.assertEqual(feat_ranking, [2.2, 3, 4, 1, 5])

    def test_own_borda(self):
        arr = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [1, 0, 3, 2]])
        feat_ranking = voting.voting_own_borda(arr, 4)
        self.assertEqual(True, np.array_equal(feat_ranking, [1, 0, 2, 3]))

    def test_weighted_borda(self):
        arr = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [1, 0, 3, 2]])
        feat_ranking = voting.voting_borda_weighted(arr, 4, N=2, mode='step')
        self.assertEqual(True, np.array_equal(feat_ranking, [1, 0, 2, 3]))


def main():
    unittest.main()
if __name__ == '__main__':
    unittest.main()