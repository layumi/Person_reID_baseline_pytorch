import numpy as np
from scipy.io import loadmat

class evaluator():
	def __init__(self, result=None):
		self.result = result

	def load(self, path):
		self.result = loadmat(path)

	def evaluate(self):
		if self.result is None:
			print('No result mat given, please load it first')
			return None

		query_feature = self.result['query_f']
		query_cam = self.result['query_cam'][0]
		query_label = self.result['query_label'][0]
		gallery_feature = self.result['gallery_f']
		gallery_cam = self.result['gallery_cam'][0]
		gallery_label = self.result['gallery_label'][0]

		num_query = len(query_label)
		num_gallery = len(gallery_label)

		# compute scores(aka cosine similarity) & sort it (result stored in indexs)
		# scores are sorted from big to small, here is an example.
		# scores:
		# [ 0.1,  0.3,  0.7, -0.2]
		# [ 0.2,  0.5, -0.3, -0.2]
		# [-0.8,  0.5,  0.4,  0.6]
		# the indexs will be:
		# [2 1 0 3]
		# [1 0 3 2]
		# [3 1 2 0]
		scores = np.dot(query_feature, gallery_feature.transpose())
		indexs = np.argsort(scores, axis=1)
		indexs = np.flip(indexs, axis=1)
		self.scores = scores
		self.indexs = indexs

		# find correct_labels and same_cameras.
		# the shape of the two matix will be (num_query, num_gallery)
		# True means query_label==gallery_label or query_cam==gallery_cam
		match_cameras = gallery_cam[indexs]
		match_labels = gallery_label[indexs]
		query_cam_expand = np.broadcast_to(query_cam, (num_gallery, num_query)).transpose()
		query_label_expand = np.broadcast_to(query_label, (num_gallery, num_query)).transpose()
		correct_labels = (query_label_expand==match_labels)
		same_cameras = (query_cam_expand==match_cameras)

		# generate match_flags, shape=(num_query, num_gallery), value meaning:
		# 1: correct match (same label, different camera)
		# 0: ignored match (same label, same camera or ignore junk indexs.)
		# -1: wrong match (different label)
		match_flags = np.where(correct_labels & (~same_cameras), 1, -1) # set wrong match to -1, others 1
		match_flags = np.where(correct_labels & same_cameras, 0, match_flags) # set same camera as ignored
		match_flags = np.where(match_labels==-1, 0, match_flags) # set junk as ingored.
		self.match_flags = match_flags

		# remove ignored match & compute ap for each query.
		# aps stored the ap for each query, clean_match is the match_flags after removed ignored labels.
		# the values in clean_match is only 1 -1 -2, therie meaning are as follows:
		# 1: correct match
		# -1: wrong match
		# -2: empty area, after removing the ignored match, the length of a single match will not be
		# num_gallery, the left space are filled with -2.0. Below is a example of how it is:
		# match_flags:
		# [ 0  0  1 -1  0 -1  1 -1 -1 -1]
		# [ 1  0 -1  0  1 -1 -1 -1 -1 -1]
		# [ 1 -1  0  0 -1 -1 -1 -1 -1 -1]
		# the calculated clean_match matrix will be:
		# [ 1 -1 -1  1 -1 -1 -1 -2 -2 -2]
		# [ 1 -1  1 -1 -1 -1 -1 -1 -2 -2]
		# [ 1 -1 -1 -1 -1 -1 -1 -1 -2 -2]
		# after this, the Rank@N will be easy to compute.
		aps = []
		clean_match = np.ones(match_flags.shape) * (-2.0)
		for i in range(num_query):
			# clean a single match result.
			this_match = match_flags[i] # get a single match
			ignored_match = np.argwhere(this_match==0).reshape(-1) # find ignored match
			this_match = np.delete(this_match, ignored_match) # delete ignored match
			clean_match[i][:len(this_match)] = this_match # write this clean match to clean_match

			# compute this AP.
			good_match = np.argwhere(this_match==1).reshape(-1) + 1.0
			mask = np.arange(len(good_match)) + 1.0
			precision = mask / good_match
			old_precision = np.ones(precision.shape)
			old_precision[1:] = mask[:-1] / (good_match[1:] - 1.0)
			if good_match[0] == 1:
				old_precision[0] = 1.0
			else:
				old_precision[0] = 0
			ap = (precision + old_precision)/2.0
			ap = ap.mean()
			aps.append(ap)
		aps = np.asarray(aps, dtype=np.float32)
		self.clean_match = clean_match
		self.aps = aps

		# compute Rank@1, Rank@5, Rank@10 and mAP
		rank_1 = (clean_match[:, 0]==1).astype(np.float32).sum() / num_query
		rank_5 = (clean_match[:, :5]==1).any(axis=1).astype(np.float32).sum() / num_query
		rank_10 = (clean_match[:, :10]==1).any(axis=1).astype(np.float32).sum() / num_query
		mAP = aps.mean()
		self.rank_1, self.rank_5, self.rank_10, self.mAP = rank_1, rank_5, rank_10, mAP
		return rank_1, rank_5, rank_10, mAP

	def show(self, precision=6):
		info = 'Rank@1: %.{0}f\tRank@5: %.{0}f\tRank@10: %.{0}f\tmAP: %.{0}f\t'.format(precision)
		# print(info)
		info = info%(self.rank_1, self.rank_5, self.rank_10, self.mAP)
		print(info)

if __name__ == '__main__':
	evaluate = evaluator()
	evaluate.load('pytorch_result.mat')
	evaluate.evaluate()
	evaluate.show()
