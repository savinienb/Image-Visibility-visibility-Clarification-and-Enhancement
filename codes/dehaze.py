
import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import numpy as np
import copy

class Dehaze:

	def dehaze(self,frame):
	    
	    frame_net=np.copy(np.double(frame))
	    
	    npstore = frame_net
	    height = npstore.shape[0]
	    width = npstore.shape[1]

	    batchdata = []
	    data = (npstore[:,:,::-1])
	    data = data.transpose((2, 0, 1))
	    batchdata.append(data)
	    self.net.blobs['data'].data[...] = batchdata;

	    self.net.forward();

	    data = self.net.blobs['sum'].data[0];
	    data = data.transpose((1, 2, 0));
	    data = data[:, :, ::-1]

	    return data


	    
	def EditTemplate(self,templateFile,template_dir, height, width):
		with open(templateFile, 'r') as ft:
			template = ft.read()
	        outFile = template_dir+'deploy.prototxt'
	        with open(outFile, 'w') as fd:
	            fd.write(template.format(height=height,width=width))



	def __init__(self,frame,templateFile,template_dir,caffemodel,gpu=1):

	    if gpu==1:
	        caffe.set_mode_gpu()
	        caffe.set_device(0)
	    else:
	        caffe.set_mode_cpu()

	    npstore = frame
	    height = npstore.shape[0]
	    width = npstore.shape[1]
	    self.EditTemplate(templateFile,template_dir, height, width)


	    
	    self.model=caffemodel;
	    self.net = caffe.Net(template_dir+'deploy.prototxt', self.model, caffe.TEST);
	    layer_list=self.net.layers

	    #conv_num_out = layer_list.property
	    #hasatrr(layers_list.layers)
	    print 'Model Changed'



