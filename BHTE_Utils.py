import sys

import numpy as np
import h5py
import hdf5plugin 
from pydicom.uid import UID
from collections import OrderedDict

if sys.version_info > (3, 2):
    def bCheckIfStr(v):
        return type(v) is str
    def cStr(v):
        if type(v) is bytes:
            return v.decode("utf-8")
        else:
            return v
else:
    def bCheckIfStr(v):
        return (type(v) is str or type(v) is unicode)
    def cStr(v):
        return v

def GetThermalOutName(InputPData,DurationUS,DurationOff,DutyCycle,Isppa,PRF,Repetitions):
    suffix = '-ThermalField-Duration-%i-DurationOff-%i-DC-%i-Isppa-%2.1fW-PRF-%iHz' % (DurationUS,DurationOff,DutyCycle*1000,Isppa,PRF)
    if Repetitions >1:
        suffix+='-%iReps' % (Repetitions)
    if '__Steer_X' in InputPData:
        #we check if this a case for multifocal delivery
        return InputPData.split('__Steer_X')[0]+'_DataForSim'+suffix
    else:
        return InputPData.split('.h5')[0]+suffix

def AnalyzeLosses(pAmp,MaterialMap,LocIJK,Input,MaterialList,BrainID,pAmpWater,Isppa,SaveDict,xf,yf,zf):
    pAmpBrain=pAmp.copy()
    if 'MaterialMapCT' in Input:
        pAmpBrain[MaterialMap!=2]=0.0
    else:
        pAmpBrain[MaterialMap<4]=0.0

    cz=LocIJK[2]
    
    PlanAtMaximum=pAmpBrain[:,:,cz]
    AcousticEnergy=(PlanAtMaximum**2/2/MaterialList['Density'][BrainID]/ MaterialList['SoS'][BrainID]*((xf[1]-xf[0])**2)).sum()
    print('Acoustic Energy at maximum plane',AcousticEnergy)
    
    
    MateriaMapTissue=np.ascontiguousarray(np.flip(Input['MaterialMap'],axis=2))
    xfr=Input['x_vec']
    yfr=Input['y_vec']
    zfr=Input['z_vec']
    
    
    PlanAtMaximumWater=pAmpWater[:,:,2] 
    AcousticEnergyWater=(PlanAtMaximumWater**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy entering',AcousticEnergyWater)
    if 'MaterialMapCT' in Input:
        pAmpWater[MaterialMap!=2]=0.0
    else:
        pAmpWater[MaterialMap!=4]=0.0
    cxw,cyw,czw=np.where(pAmpWater==pAmpWater.max())
    cxw=cxw[0]
    cyw=cyw[0]
    czw=czw[0]
    print('Location Max Pessure Water',cxw,cyw,czw,'\n',
            xf[cxw],yf[cyw],zf[czw],pAmpWater.max()/1e6)
    
    pAmpTissue=np.ascontiguousarray(np.flip(Input['p_amp'],axis=2))
    if 'MaterialMapCT' in Input:
        pAmpTissue[MaterialMap!=2]=0.0
    else:
        pAmpTissue[MaterialMap!=4]=0.0

    cxr,cyr,czr=np.where(pAmpTissue==pAmpTissue.max())
    cxr=cxr[0]
    cyr=cyr[0]
    czr=czr[0]
    print('Location Max Pressure Tissue',cxr,cyr,czr,'\n',
            xfr[cxr],yfr[cyr],zfr[czr],pAmpTissue.max()/1e6)
    

    PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czw]
    AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy at maximum plane water max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
    
    PlanAtMaximumWaterMaxLoc=pAmpWater[:,:,czr]
    AcousticEnergyWaterMaxLoc=(PlanAtMaximumWaterMaxLoc**2/2/MaterialList['Density'][0]/ MaterialList['SoS'][0]*((xf[1]-xf[0])**2)).sum()
    print('Water Acoustic Energy at maximum plane tissue max loc',AcousticEnergyWaterMaxLoc) #must be very close to AcousticEnergyWater
    
    PlanAtMaximumTissue=pAmpTissue[:,:,czr] 
    AcousticEnergyTissue=(PlanAtMaximumTissue**2/2/MaterialList['Density'][BrainID]/ MaterialList['SoS'][BrainID]*((xf[1]-xf[0])**2)).sum()
    print('Tissue Acoustic Energy at maximum plane tissue',AcousticEnergyTissue)
    
    RatioLosses=AcousticEnergyTissue/AcousticEnergyWaterMaxLoc
    print('Total losses ratio and in dB',RatioLosses,np.log10(RatioLosses)*10)
        
    PressureAdjust=np.sqrt(Isppa*1e4*2.0*SaveDict['MaterialList']['SoS'][BrainID]*SaveDict['MaterialList']['Density'][BrainID])
    PressureRatio=PressureAdjust/pAmpTissue.max()
    return PressureRatio,RatioLosses

def getBHTECoefficient( kappa,rho,c_t,h,t_int,dt=0.1):
    """ calculates the Bioheat Transfer Equation coefficient required (time step/density*conductivity*voxelsize"""
    # get the bioheat coefficients for a tissue type -- independent of surrounding tissue types
    # dt = t_int/nt
    # h - voxel resolution - default 1e-3

    bhc_coeff = kappa * dt / (rho * c_t * h**2)
    if bhc_coeff >= (1 / 6):
        best_nt = np.ceil(6 * kappa * t_int) / (rho * c_t *h**2)
        print("The conditions %f,%f,%f does not meet the C-F-L condition and may not be stable. Use nt of %f or greater." %\
            (dt,t_int,bhc_coeff,best_nt))
    return bhc_coeff

def getPerfusionCoefficient( w_b,c_t,blood_rho,blood_ct,dt=0.1):
    """Calculates the perfusion coefficient based on the simulation parameters and time step """
    # get the perfusion coeff for a speicfic tissue type and time period  -- independent of surrounding tissue types
    # wb is in ml/min/kg, needs to be converted to m3/s/kg (1min/60 * 1e-6 m3/ml)

    coeff = w_b/60*1.0e-6* blood_rho * blood_ct * dt / c_t  

    return coeff

def getQCoeff(rho,SoS,alpha,c_t,Absorption,h,dt):
    coeff=dt/(2*rho**2*SoS*h*c_t)*Absorption*(1-np.exp(-2*h*alpha))
    return coeff

def factors_gpu(x):
    res=[]
    for i in range(2, x + 1):
        if x % i == 0:
            res.append(i)
    res=np.array(res)
    return res

def FitSpeedCorticalShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1577.0,1498.0,1313.0]).mean()
    Cs836=np.array([1758.0,1674.0,1545.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularShear(frequency):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    FRef=np.array([270e3,836e3])
    Cs270=np.array([1227.0,1365.0,1200.0]).mean()
    Cs836=np.array([1574.0,1252.0,1327.0]).mean()
    CsRef=np.array([Cs270,Cs836])
    p=np.polyfit(FRef, CsRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitAttBoneShear(frequency,reductionFactor=1.0):
    #from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc 
    PichardoData=(57.0/.27 +373/0.836)/2
    return np.round(PichardoData*(frequency/1e6)*reductionFactor) 

def FitSpeedCorticalLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014 
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2448.0,2516])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitSpeedTrabecularLong(frequency):
    #from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    FRef=np.array([270e3,836e3])
    ClRef=np.array([2140.0,2300])
    p=np.polyfit(FRef, ClRef, 1)
    return(np.round(np.poly1d(p)(frequency)))

def FitAttCorticalLong_Goss(frequency,reductionFactor=1):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=(2.15+1.67)/2*100*reductionFactor
    return np.round(JasaAtt1MHz*(frequency/1e6)) 

def FitAttTrabecularLong_Goss(frequency,reductionFactor=1):
    #from J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    JasaAtt1MHz=1.5*100*reductionFactor
    return np.round(JasaAtt1MHz*(frequency/1e6)) 

def FitAttCorticalLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
    # fitting from data obtained from
    #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
    
    return np.round(203.25090263*((frequency/1e6)**bcoeff)*reductionFactor)

def FitAttTrabecularLong_Multiple(frequency,bcoeff=1,reductionFactor=0.8):
    #reduction factor 
    # fitting from data obtained from
    #J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
    return np.round(202.76362433*((frequency/1e6)**bcoeff)*reductionFactor)

def ReadFromH5py(f,group=None,typeattr=None):
    if bCheckIfStr(f):
        fileobj=h5py.File(f, "r")
    elif type(f) is h5py._hl.files.File:
        fileobj=f
    else:
        raise TypeError( "only string or h5py.file objects are accepted for 'f' object")
    if group is None:
        group=fileobj
        MyDict={}
    else:
        if typeattr=='OrderedDict':
            MyDict=OrderedDict()
        else:
            MyDict={}
    for (namevar,val) in group.items():

        NameList=None
        typeattr=cStr(val.attrs["type"])
        if type(val) is h5py._hl.group.Group:
            ValGroup=ReadFromH5py(fileobj,val)
            if typeattr=="list" or  typeattr=="tuple" :
                NameList=namevar
            if NameList is not None:
                ListVal=[]
                for n in range(len(ValGroup)):
                    itemname = "item_%d"%n
                    ListVal.append(ValGroup[itemname])
                if typeattr=="list":
                    MyDict[NameList] = ListVal
                else:
                    MyDict[NameList] = tuple(ListVal)
            else:

                if "type_key" in val.attrs:
                    typekey= cStr(val.attrs["type_key"])
                    if typekey=='int':
                        snamevar=int(namevar)
                    elif typekey=='float':
                        snamevar=int(namevar)
                    elif typekey=='str':
                        snamevar=namevar
                    elif typekey=='unicode':
                        snamevar=namevar
                    else:
                        raise TypeError( "the type of dictionary key is not supported " + namevar + ' (' + typekey + ')')
                else:
                    snamevar=namevar
                MyDict[snamevar] = ValGroup
        elif typeattr == "None":
             MyDict[namevar]=None
        elif typeattr == "UID":
            MyDict[namevar]=UID(val[()])
        else:
            if cStr(val.attrs["type"])=="scalar":
                if type(val[()]) is np.int32 or type(val[()]) is np.int64:
                    MyDict[namevar]=int(val[()])
                else:
                    MyDict[namevar] = val[()]
            elif cStr(val.attrs["type"])=="<type 'str'>":
                MyDict[namevar] = cStr(val[()])
            else:
                MyDict[namevar] = val[()]
    if bCheckIfStr(f):
        fileobj.close()
    return MyDict

def GetMaterialList(Frequency,BaselineTemperature):
    MatFreq={}
    Material={}
    #Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
    Material['Water']=     np.array([1000.0, 1500.0, 0.0   ,   0.0,                   0.0] )
    Material['Cortical']=  np.array([1896.5, FitSpeedCorticalLong(Frequency), 
                                                FitSpeedCorticalShear(Frequency),  
                                                FitAttCorticalLong_Multiple(Frequency)  , 
                                                FitAttBoneShear(Frequency)])
    Material['Trabecular']=np.array([1738.0, FitSpeedTrabecularLong(Frequency),
                                                FitSpeedTrabecularShear(Frequency),
                                                FitAttTrabecularLong_Multiple(Frequency) , 
                                                FitAttBoneShear(Frequency)])
    Material['Skin']=      np.array([1116.0, 1537.0, 0.0   ,  2.3*Frequency/500e3 , 0])
    Material['Brain']=     np.array([1041.0, 1562.0, 0.0   ,  3.45*Frequency/500e3 , 0])

    MatFreq[Frequency]=Material

    Input = {}
    Materials = []
    for k in ['Water','Skin','Cortical','Trabecular','Brain']:
        SelM = MatFreq[Frequency][k]
        Materials.append([SelM[0], # Density
                        SelM[1], # Longitudinal SOS
                        SelM[2], # Shear SOS
                        SelM[3], # Long Attenuation
                        SelM[4]]) # Shear Attenuation
    Materials = np.array(Materials)
    MaterialList = {}
    MaterialList['Density'] = Materials[:,0]
    MaterialList['SoS'] = Materials[:,1]
    MaterialList['Attenuation'] = Materials[:,3]

    #Water, Skin, Cortical, Trabecular, Brain

    #https://itis.swiss/virtual-population/tissue-properties/database/heat-capacity/
    MaterialList['SpecificHeat']=[4178,3391,1313,2274,3630] #(J/kg/°C)
    #https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
    MaterialList['Conductivity']=[0.6,0.37,0.32,0.31,0.51] # (W/m/°C)
    #https://itis.swiss/virtual-population/tissue-properties/database/heat-transfer-rate/
    MaterialList['Perfusion']=np.array([0,106,10,30,559])

    MaterialList['Absorption']=np.array([0,0.85,0.16,0.15,0.85])

    MaterialList['InitTemperature']=[BaselineTemperature,BaselineTemperature,
                                        BaselineTemperature,BaselineTemperature,BaselineTemperature]
    
    return MaterialList