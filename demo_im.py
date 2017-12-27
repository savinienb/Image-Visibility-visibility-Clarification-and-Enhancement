import os

os.environ['GLOG_minloglevel'] = '2' 
import caffe
import time
import cv2
import numpy as np


def Print_out(img_name,model_cur,elaps,height,width):
    print ' '
    print '========================='
    print 'Model :'+model_cur
    print 'Dehazed image : '+img_name
    print 'Size :'+str(height)+'*'+str(width)+' pixels'
    print 'Elapsed time : '+ str(elaps[-1]*height*width)+' s'
    print '------------'
    print 'Mean time per pixel per image: '+str(np.mean(elaps))
    print '========================='

def EditTemplate(templateFile,outFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
        with open(outFile, 'w') as fd:
            fd.write(template.format(height=height,width=width))

def test(model,img_dir,img_type,result_dir,template_dir,template_name):
    #caffe.set_mode_gpu()
    #caffe.set_device(0)
    for mod in range (len(model)):
	    print ''
	    print '###########################################'
	    templateFile=template_dir+model[mod]+template_name
	    outFile=template_dir+'/deploy.prototxt'

	    info = os.listdir(img_dir);
	    imagesnum=0;

	    elaps=[]
	    for types in range(len(img_type)):

		    for root, dirs, files in (list(os.walk(img_dir))):
		        if len(files)==0 and types==0:
		            print 'No file in the directory'
		            break
		        for file in files:

		            if file.endswith('.'+img_type[types]):

		                img_path=img_dir+file
		                result_path=result_dir+file[:-4]+'_'+model[mod]+'.png'


		                npstore = caffe.io.load_image(img_path)

		                height = npstore.shape[0]
		                width = npstore.shape[1]

	                    #resize the image to a 2*n to allow upsampling
		                if height%2!=0:
		                     height=height-1
		                     npstore=npstore[:-1,:]
		                if width%2!=0:
		                     width=width-1
		                     npstore=npstore[:,:-1]

		                EditTemplate(templateFile,outFile, height, width)

		                net = caffe.Net(outFile, model[mod] + '.caffemodel', caffe.TEST);
		                batchdata = []
		                data = npstore
		                data = data.transpose((2, 0, 1))
		                batchdata.append(data)
		                net.blobs['data'].data[...] = batchdata;
		                start = time.clock()

		                net.forward();

		                elaps.append((time.clock() - start)/height/width)

		                data = net.blobs['sum'].data[0];
		                data = data.transpose((1, 2, 0));
		                data = data[:, :, ::-1]
		                
		                cv2.imwrite(result_path, data * 255.0)
		                Print_out(file,model[mod],elaps,height,width)
	     
def main():
    
    caffe.set_mode_gpu();
    #caffe.set_mode_cpu();

    os.environ['GLOG_minloglevel'] = '2' 
    template_dir='./template/'
    template_name='_template.prototxt'
    img_dir='./data/img/'
    img_type=['jpg','png']
    result_dir='./data/result/'

    model=['DeMisty_FI','DeMisty_RI']
    test(model,img_dir,img_type,result_dir,template_dir,template_name)


if __name__ == '__main__':
    main();


