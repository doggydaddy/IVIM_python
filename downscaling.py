import sys,  os,  getopt
import timeit
from timeit import default_timer as timer 
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import scipy.ndimage as ndimage
import dicom
import numpy as np
import dicom.UID
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm

def scroll_display(x_sol3D): #needs a 3d matrix as input, reshape if necessary
    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')
            self.X = X
            rows, cols, self.slices, = X.shape
            self.ind = self.slices//2
            self.im = ax.imshow(self.X[ :,:,self.ind],cmap=cm.bone,interpolation='None')  #for logscale (k-space images):  ,norm=colors.LogNorm()
            # self.im = ax.imshow(self.X[ self.ind,:, :],cmap=cm.coolwarm,interpolation='None')  #for logscale (k-space images):  ,norm=colors.LogNorm()
            # cbar = fig.colorbar(self.im,ticks=[0, 1,2,3,4,5,6,7,8])
            self.update()
        def onscroll(self, event):
            if event.button == 'up':
                self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
            else:
                self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
            self.update()
        def update(self):
            self.im.set_data(self.X[:,:,self.ind])
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X=x_sol3D
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def getHigherLevel(level):
    high = {'L': level['L']+1}
    maxIsotropy = 0
    for sx in [1, 2]:
        for sy in [1, 2]:
            for sz in [1, 2]:
                if (sx*sy*sz > 1 and level['nx'] >= sx and
                   level['ny'] >= sy and level['nz'] >= sz):
                    if (level['nx'] == 1):
                        iso = isotropy2D(level['dy']*sy, level['dz']*sz)
                    elif (level['ny'] == 1):
                        iso = isotropy2D(level['dx']*sx, level['dz']*sz)
                    elif (level['nz'] == 1):
                        iso = isotropy2D(level['dx']*sx, level['dy']*sy)
                    else:
                        iso = isotropy3D(
                          level['dx']*sx, level['dy']*sy, level['dz']*sz)
                    if iso > maxIsotropy:
                        maxIsotropy = iso
                        high['sx'] = sx
                        high['sy'] = sy
                        high['sz'] = sz
    high['dx'] = level['dx']*high['sx']
    high['dy'] = level['dy']*high['sy']
    high['dz'] = level['dz']*high['sz']
    high['nx'] = int(np.ceil(level['nx']/high['sx']))
    high['ny'] = int(np.ceil(level['ny']/high['sy']))
    high['nz'] = int(np.ceil(level['nz']/high['sz']))
    return high
    
def isotropy2D(dx, dy): return np.sqrt(dx*dy)/(2*(dx+dy))

def isotropy3D(dx, dy, dz): return (dx*dy*dz)**(2/3)/(2*(dx*dy+dx*dz+dy*dz))

def getHigher_level_image(w, higher_level, level):
    w_level=np.zeros((level['nx']+level['nx'] % higher_level['sx'],level['ny']+level['ny'] % higher_level['sy'],level['nz']+level['nz'] % higher_level['sz'],w.shape[-1]))
    w_level[ :level['nx'], :level['ny'], :level['nz'],:]=w    #zeropadded level image
    if level['nx'] % higher_level['sx'] == 1:
        w_level[-1,:,:,:]=w_level[-2,:,:,:]
    if level['ny'] % higher_level['sy'] == 1:
        w_level[:,-1,:,:]=w_level[:,-2,:,:]
    if level['nz'] % higher_level['sz'] == 1:
        w_level[:,:,-2,:]=w_level[:,:,-2,:]     ##datapadded level image
        
    w_higher_level=w_level[::higher_level['sx'],::higher_level['sy'],::higher_level['sz'],:]
    
    if higher_level['sx'] > 1:
        w_higher_level += w_level[1::higher_level['sx'], ::higher_level['sy'], ::higher_level['sz'],:]
    if higher_level['sy'] > 1:
        w_higher_level += w_level[ ::higher_level['sx'], 1::higher_level['sy'], ::higher_level['sz'],:]
    if higher_level['sz'] > 1:
        w_higher_level += w_level[ ::higher_level['sx'], ::higher_level['sy'], 1::higher_level['sz'],:]
    if higher_level['sx'] > 1 and higher_level['sy'] > 1:
        w_higher_level += w_level[ 1::higher_level['sx'], 1::higher_level['sy'], ::higher_level['sz'],:]
    if higher_level['sx'] > 1 and higher_level['sz'] > 1:
        w_higher_level += w_level[ 1::higher_level['sx'], ::higher_level['sy'], 1::higher_level['sz'],:]
    if higher_level['sy'] > 1 and higher_level['sz'] > 1:
        w_higher_level += w_level[ ::higher_level['sx'], 1::higher_level['sy'], 1::higher_level['sz'],:]
    if higher_level['sx'] > 1 and higher_level['sy'] > 1 and higher_level['sz'] > 1:
        w_higher_level += w_level[ 1::higher_level['sx'], 1::higher_level['sy'], 1::higher_level['sz'],:]
        
        

    return w_higher_level/higher_level['sx']/higher_level['sy']/higher_level['sz']    #normalisation, edges should be fine because of datapadding


def get_downscaled_image(high,w_high):
    w_from_downsampled_w=np.zeros((high['nx']*high['sx'], high['ny']*high['sy'],high['nz']*high['sz'],w_high.shape[3]), dtype=int)
    w_from_downsampled_w[::high['sx'],::high['sy'],::high['sz'],:]=w_high
    if high['sx']>1:
        w_from_downsampled_w[1::high['sx'],::high['sy'],::high['sz'],:]=w_high
    if high['sy']>1:
        w_from_downsampled_w[::high['sx'],1::high['sy'],::high['sz'],:]=w_high
    if high['sz']>1:
        w_from_downsampled_w[::high['sx'],::high['sy'],1::high['sz'],:]=w_high 
    if high['sx']>1 and high['sy']>1:
        w_from_downsampled_w[1::high['sx'],1::high['sy'],::high['sz'],:]=w_high
    if high['sx']>1 and high['sz']>1:
        w_from_downsampled_w[1::high['sx'],::high['sy'],1::high['sz'],:]=w_high
    if high['sy']>1 and high['sz']>1:
        w_from_downsampled_w[::high['sx'],1::high['sy'],1::high['sz'],:]=w_high
    if high['sx']>1 and high['sy']>1 and high['sz']>1:
        w_from_downsampled_w[1::high['sx'],1::high['sy'],1::high['sz'],:]=w_high
    return w_from_downsampled_w


# def downscale_image(w,dx=3,dy=3,dz=5):
#     nx,ny,nz=w.shape[:-1]
#     level = {'L': 0, 'nx': nx, 'ny': ny, 'nz': nz,'sx': 1, 'sy': 1, 'sz': 1,'dx': dx, 'dy': dy, 'dz': dz}
#     higher_level=getHigherLevel(level)
#     w_higher_level=getHigher_level_image(w,higher_level,level)
#     return w_higher_level

# # generate 3d gaussian image
# def gaussian(x,sigma,mu):
#     return np.exp(-    (x-mu)**2/2/sigma**2   ) 
# x=np.linspace(0,100,130)
# y=np.linspace(0,130,111)
# z=np.linspace(0,15,15)
# b=np.exp(np.linspace(10,1,10))/100
# vx=gaussian(x,25,50)
# vy=gaussian(y,30,40)
# vz=gaussian(z,10,10)
# w=vx[:,None,None,None]*vy[None,:,None,None]*vz[None,None,:,None]*b[None,None,None,:]
# 
# w=np.random.randint(10,size=(3,3,3,1))
# 
# 
# 
# 
# 
# nx,ny,nz=w.shape[:-1]
# dx,dy,dz=(1.6,1.6,4)  #assumed pixel sizes, fetch this from dicom data on real fcts
#  
# level = {'L': 0, 'nx': nx, 'ny': ny, 'nz': nz,'sx': 1, 'sy': 1, 'sz': 1,'dx': dx, 'dy': dy, 'dz': dz}
# 
# 
# higher_level=getHigherLevel(level)
# w_high=getHigher_level_image(w,higher_level,level)
# scroll_display(w_high[:,:,:,0])
# scroll_display(w[:,:,:,0])
# 
# higher_level=level
# w_higher_level=w
# level_list=[level]
# w_list=[w]
# n_iter=0
# print('Iteration #'+str(n_iter),'Image dims:',w_list[-1][:,:,:,0].shape)
# while w_list[-1][:,:,:,0].shape !=(1,1,1):
#     level_list.append(getHigherLevel(level_list[-1]))
#     w_list.append(getHigher_level_image(w_list[-1],level_list[-1],level_list[-2]))
#     n_iter+=1
#     print('Iteration #'+str(n_iter),'Image dims:',w_list[-1][:,:,:,0].shape)
# 
# 
# 
# #get a list of all downscaled images
# 
# dsc_w_list=[]
# for i in range(len(w_list)):
#     temp=w_list[i]
#     while i !=0:
#         temp=get_downscaled_image(level_list[i],temp)[:level_list[i-1]['nx'],:level_list[i-1]['ny'],:level_list[i-1]['nz'],:]
#         i-=1
#     dsc_w_list.append(temp)
#     
#     
# scroll_display(dsc_w_list[8][:,:,:,0])
# 

# 
































