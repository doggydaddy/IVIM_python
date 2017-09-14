#!/usr/bin/python
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
import downscaling
from downscaling import *

#for testing only, delete afterwards

def scroll_display(x_sol3D,cmap=cm.bone): #needs a 3d matrix as input, reshape if necessary
    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')
            self.X = X
            rows, cols, self.slices, = X.shape
            self.ind = self.slices//2
            self.im = ax.imshow(self.X[ :,:,self.ind],cmap=cmap,interpolation='None') 
            cbar = fig.colorbar(self.im)
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

class DataWrapper:
    def __init__(self, inputFolder, outputFolder,nstf):
        self.inputDir = inputFolder
        self.outputDir = outputFolder
        self.nstf=nstf    #testing

    def run(self):
        self.parseFiles()
        self.dataCheck01()
        self.parseHeaders()
        self.dataCheck02()
        self.parseData()

    def parseFiles(self):
        self.lstFilesDCM = []
        for dirName,  subdirList,  fileList in os.walk(self.inputDir):
            for filename in fileList:
                try:
                    df=dicom.read_file(os.path.join(dirName, filename))
                    self.lstFilesDCM.append(os.path.join(dirName, filename))
                except:
                    print('WARNING: Unreadable file detected. Please remove any non-DICOM file from the input folder.')
                    sys.exit(2)
                    
    def dataCheck01(self):
        print('')
        print("MESSAGE: Performing data check ...")
        print('MESSAGE: Found', len(self.lstFilesDCM),'files in the input data folder.')
        print('')

    def parseHeaders(self):
        self.b_values=[]
        self.Slices = []
        for file in self.lstFilesDCM:
            curD = dicom.read_file(file, stop_before_pixels=True)
            b_value=curD[0x19,0x0100c].value
            # b_dir=curD[0x19,0x100d].value                          #diff direction for future use
            Slice = curD[0x20, 0x32].value[2]
            if b_value not in self.b_values:
                self.b_values.append(b_value)
            if Slice not in self.Slices:
                self.Slices.append(Slice)
        self.b_values.sort()
        self.Slices.sort()
    
    def dataCheck02(self):
        print("MESSAGE: Performing data check ...")
        print('MESSAGE: found',len(self.b_values),'different b values in the input data set.')
        print('MESSAGE: found',len(self.Slices),'slices in the input data set.')
        if len(self.b_values)*len(self.Slices) != len(self.lstFilesDCM):
            print('ERROR: mismatch between mismatch between the number of slices, b-values and input files!')
            sys.exit(os.EX_DATAERR)
        print('')

    def parseData(self):
        self.RefDs = dicom.read_file(self.lstFilesDCM[0])   #taking the first file as reference for nx,ny,nz
        # Load dimension based on the number of rows, columns, and slices
        self.ConstPixelDims = (int(self.RefDs.Rows),  int(self.RefDs.Columns), len(self.Slices), len(self.b_values))
        self.Voxelsize= [float(i) for i in self.RefDs.PixelSpacing  +[self.RefDs.SliceThickness]]
        # The array is sized based on 'ConstPixelDims'
        self.ArrayDicom = np.zeros(self.ConstPixelDims, dtype=self.RefDs.pixel_array.dtype)
        # loop through all the DICOM files
        self.RefSlices=[]
        for filenameDCM in self.lstFilesDCM:
            # read the file
            ds = dicom.read_file(filenameDCM)
            b_value=ds[0x19,0x100c].value
            Slice = ds[0x20, 0x32].value[2]
            idx_b = self.b_values.index(b_value)
            idx_Slice = self.Slices.index(Slice)
            self.ArrayDicom[:,:,idx_Slice,idx_b]=ds.pixel_array            
            if idx_b is 0:
                self.RefSlices.append(dicom.read_file(filenameDCM))
        self.RefSlices.sort(key=lambda refslice: refslice[0x20,0x32].value)       #list of ref slices, sorted by loc
        if self.nstf != len(self.Slices):
            print('MESSAGE: Found',len(self.Slices),'slices. Only treating',self.nstf,'as specified by user.')
            self.ArrayDicom=self.ArrayDicom[:,:,(len(self.Slices)//2-self.nstf):(len(self.Slices)//2),:] #testing only
        
    def getDims(self, idx):
        return int(self.ConstPixelDims[idx])

    def getOutput(self,params,label_map):
        self.params=params
        self.label_map=label_map
        
    def writeDicomFromTemplate(self, pixel_array, filename, refd):
        ds = refd
        ds.SecondaryCaptureDeviceManufctur = 'python'
        ds.SeriesNumber = '99'
        tmpNr = ds.AccessionNumber
        ds.StudyID = tmpNr[-2:]
        ds.AccessionNumber = tmpNr[:-2]
        ds.ProtocolName = 'ADC_Map_monoexp_7_bees'
        uuid = dicom.UID.generate_uid()
        ds.SOPInstanceUID = ds.SOPInstanceUID[0:35] + uuid[ (len(uuid)-18):(len(uuid)-1) ]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = np.min(pixel_array).astype(np.uint16)
        ds.LargestImagePixelValue = np.max(pixel_array).astype(np.uint16)
        if pixel_array.dtype != np.uint16:
            pixel_array = np.round(pixel_array, 0)
            pixel_array = pixel_array.astype(np.uint16)
        ds.PixelData = pixel_array.tostring()
        ds.save_as(filename)

    def writeOutput(self):
        print('MESSAGE: Exporting output maps ...')
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)
        for i in range(self.params.shape[2]):
            filename='label_map_slice_'+str(i+1)+'.dcm'
            self.writeDicomFromTemplate(self.label_map[:,:,i]*1000,self.outputDir+'\\'+filename, self.RefSlices[i])
            for j in range(self.params.shape[3]):
                filename = 'parameter_'+str(j+1)+ '_slice_' + str(i+1) + '.dcm' 
                self.writeDicomFromTemplate(self.params[:,:,i,j]/self.params[:,:,i,j].max()*3000,self.outputDir+'\\'+filename, self.RefSlices[i])
        print('MESSAGE: ... export completed!')
        print('')
 
class diff_Fitting:
    def __init__(self, dims, i_b_values, i_ArrayDicom,Voxelsize):
        self.nRows,self.nColumns,self.nSlices=dims
        self.b_values = np.array(i_b_values)
        self.ArrayDicom = i_ArrayDicom
        self.Voxelsize=Voxelsize
        self.Nb=len(self.b_values)
        
    def ln_fit(self,b,ln_So,D):
        return ln_So-b*D 

    def ivim(self,b,So,f,Dp,Dd):
        return So*(f*np.exp(-b*Dp)+(1-f)*np.exp(-b*Dd))     
    
    def fit_fct(self,i,nvxl,xdata,data_i,So_treshold):
        try:
            params_lowb=curve_fit(self.ln_fit,xdata[:5],np.log(data_i[:5]),maxfev=100,p0=[np.log(data_i[0]),0.002],bounds=([-np.inf,0],[np.log(data_i[0])*2,0.01]))[0]
            params_hib=curve_fit(self.ln_fit,xdata[4:],np.log(data_i[4:]),maxfev=100,p0=[np.log(data_i[0]),0.002],bounds=([-np.inf,0],[np.log(data_i[0])*2,0.01]))[0]              
            #initial guesses for ivim fitting
            Dd_guess=max(params_hib[1] ,params_lowb[1] )
            Dp_guess=min(params_hib[1] ,params_lowb[1] )
            So_guess=data_i[0]
            #todo: better f initial guess, how? low b image is mostly perfusion (fast adc) weighted, hi b image is mostly slowadc diff weighted. -> ratio of So values form monoexp fits tells something?
            
            # f_guess=0.5
            # print(params_hib[0],params_lowb[0],np.log(data_i[0]))
            f_guess=np.exp(params_hib[0])/np.exp(params_lowb[0])
            if f_guess > 1:
                f_guess=1       #in case weird fits happen and give nonsensical values of So's
            # print(f_guess)
            #ivim fit with initial guesses above
            self.params[i]=curve_fit(self.ivim,xdata,data_i,maxfev=10000,p0=[So_guess,f_guess,Dp_guess,Dd_guess],bounds=([So_guess/2,0,Dp_guess/2,Dd_guess/2],[So_guess*2,1,Dp_guess*2,Dd_guess*2]))[0]
            # self.label_map[i]=1
            self.fit_fail_flag=1
        except(RuntimeError):
            self.ndst+=1
            self.label_map[i]=self.ndst
            pass
            

    def run(self):
        #run this to debug: self=diff_fit
        print('MESSAGE: Performing IVIM fitting...')
        nx,ny,nz=(self.nRows, self.nColumns, self.nSlices)
        xdata=np.array(self.b_values).astype(float)
        smoothed_ydata=ndimage.gaussian_filter(self.ArrayDicom.astype(float),sigma=(0,0,0,0),order=0)  #first two sigmas for x,y smoothing      
        nvxl=nx*ny*nz
        data=smoothed_ydata[:,:,:,:]+1   #+1 to avoid intensities=0 which mess with the log        
        
        #downsampling data set in order to perform fits where the fit failed at native scale
        print('\nMESSAGE: Performing Image Downsampling...')
        dx,dy,dz=self.Voxelsize
        level = {'L': 0, 'nx': nx, 'ny': ny, 'nz': nz,'sx': 1, 'sy': 1, 'sz': 1,'dx': dx, 'dy': dy, 'dz': dz}
        level_list=[level]
        w_list=[data.reshape(nx,ny,nz,self.Nb)]
        n_iter=0
        print('\nIteration #'+str(n_iter),'Image dimensions:',w_list[-1][:,:,:,0].shape)
        while w_list[-1][:,:,:,0].shape !=(1,1,1):
            level_list.append(getHigherLevel(level_list[-1]))
            w_list.append(getHigher_level_image(w_list[-1],level_list[-1],level_list[-2]))
            n_iter+=1
            print('Iteration #'+str(n_iter),'Image dims:',w_list[-1][:,:,:,0].shape)
        dsc_w_list=[]
        for i in range(len(w_list)):
            temp=w_list[i]
            while i !=0:
                temp=get_downscaled_image(level_list[i],temp)[:level_list[i-1]['nx'],:level_list[i-1]['ny'],:level_list[i-1]['nz'],:]
                i-=1
            dsc_w_list.append(temp.reshape(nvxl,self.Nb))
        
        #multiscale fitting
        So_treshold=1000
        nvxl=nx*ny*nz
        data=data.reshape(nvxl,len(xdata))
        self.params=np.zeros((nvxl,4))     
        progress_counter=0
        self.label_map=np.zeros(nvxl)  #label map
        start=timer()
        for i in range(nvxl):
            if (i/nvxl*100)%1==0:
                print(str(progress_counter)+'%,',end=' ')
                progress_counter+=1
            self.ndst=0
            self.fit_fail_flag=0
            while self.fit_fail_flag==0:
                self.fit_fct(i,nvxl,xdata,dsc_w_list[self.ndst][i],So_treshold)       
        self.label_map.shape=(nx,ny,nz)
        self.params.shape=(nx,ny,nz,4)
        end=timer()
        time=end-start
        print('MESSAGE: Fitting completed. Time taken:',time//3600,'hours +',time%3600//60,'hours +',time%3600%60,'seconds')

# def main(argv):
inputFolder = r'C:\Users\User\Desktop\shell_diff\input data'
outputFolder = r'C:\Users\User\Desktop\shell_diff\output data'

print( 'Input folder is "', inputFolder )
print( 'Output folder is "', outputFolder )


#for testing, Number of Slices To Fit
nstf=1
dw = DataWrapper(inputFolder, outputFolder,nstf)
dw.run()
diff_fit=diff_Fitting(dw.ArrayDicom.shape[0:3],dw.b_values,dw.ArrayDicom,dw.Voxelsize)
diff_fit.run()

# cmap=cm.jet
# cmap=cmap.from_list('Custom map',[(0,1,0,1),(1,0,0,1)],diff_fit.label_map.max()) #custom cmap for the label scale mapping, green to red
#  
# scroll_display(diff_fit.label_map,cmap)
# scroll_display(diff_fit.params[:,:,:,0])
# scroll_display(diff_fit.params[:,:,:,1])
# scroll_display(diff_fit.params[:,:,:,2])
# scroll_display(diff_fit.params[:,:,:,3])


print('Hi Lulu')

#ivim output
dw.getOutput(diff_fit.params,diff_fit.label_map)
dw.writeOutput()

# if __name__ == "__main__":
#     print("")
#     print("")
#     print("Initializing diffusion analysis")
#     print("Scripted for Dept. Central Radiology, MRI Unit")
#     # print("Authored by Hadrien Van Loo, MR Physics, Karolinska University Hospital, Solna, Sweden")
#     # print("09-2017")
#     print("")
#     main(sys.argv[1:])
#     print("")
#     print("Script finished sucessfully, exiting normally.")
#     print("")
#     # sys.exit()
# 

#todo: add support for downsampling limit?, parsing So_treshold value, maxfev, etc from an options file
#todo: add support for random Nb, directional information (extract from dicom?)


