import numpy as np
import os, sys

# Main path to your caffe installation
caffe_root = '/Users/chiteshtewani/Desktop/caffe/caffe/'
sys.path.append(caffe_root + "python/")
 
# Model prototxt file
# model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'

 
sys.path.insert(0, caffe_root + 'python')
import caffe

'''
classifier = Classifier(deploy_prototxt, model_trained)
im = [caffe.io.load_image('hand2_0_right_seg_1_cropped.png')]
out = classifier.predict(im)
'''

class Classifier:
	def __init__(self, deploy_prototxt, model_trained, mean_path):
		
		#load the model
		self.net = caffe.Net(deploy_prototxt,
		                model_trained,
		                caffe.TEST)

		# load input and configure preprocessing
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
		self.transformer.set_transpose('data', (2,0,1))
		self.transformer.set_channel_swap('data', (2,1,0))
		self.transformer.set_raw_scale('data', 255.0)



	def predict(self, image_name):
		
		#note we can change the batch size on-the-fly
		#since we classify only one image, we change batch size from 10 to 1

		#net.blobs['data'].reshape(1,3,227,227)

		#load the image in the data layer
		im = caffe.io.load_image(image_name)
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)

		#compute
		# out = self.net.forward()
		#out = solver.te

		# other possibility : 
		out = self.net.forward_all(data=np.asarray([self.transformer.preprocess('data', im)]))


		#predicted predicted class
		#print out['prob']
		#print "out['prob']", out['prob']
		#i,j = np.unravel_index(out['prob'].argmax(), out['prob'].shape)
		#print i, j
		top = out['prob'][0].argsort()[-5:][::-1]
		return top

#print predicted labels
#labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#print top_k

#
# #dirName = 'NewData/A'
# dirName = 'test'
# deploy_prototxt = 'deploy.prototxt'
# # Model caffemodel file
# model_trained = 'massey_iter_1000.caffemodel'
# # Path to the mean image (used for input processing)
# mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# classifier = Classifier(deploy_prototxt, model_trained, mean_path)
# dirs = os.listdir(dirName)
# for directory in dirs:
# 	if os.path.isdir(os.path.join(dirName, directory)):
# 		for files in os.listdir(os.path.join(dirName, directory)):
# 			image_name = os.path.join(dirName, directory, files)
# 			print image_name, chr(int(classifier.predict(image_name))+97)
# 			#classifier.predict(os.path.join(dirName, directory, files))
# 	else:
# 		image_name = os.path.join(dirName, directory)
# 		print image_name, chr(int(classifier.predict(image_name))+97)
# 		#classifier.predict(os.path.join(dirName, directory))