from enum import Enum


class Variable(str, Enum):
    """"""

    pr = "pr"
    tas = "tas"
    # tasmax = "tasmax"
    # hurs = "hurs"
    # wgsmax = "wsgsmax"
    # sfcWind = "sfcWind"
    # sfcWindmax = "sfcWindmax"


class TimeFrequency(str, Enum):
    """"""

    _1hr = "1hr"
    _3hr = "3hr"
    _6hr = "6hr"
    day = "day"
    month = "mon"


class Domain(str, Enum):
    """"""

    # --- Africa domain ---
    AFR22 = "AFR-22"
    AFR44 = "AFR-44"
    # --- Antarctica domain ---
    ANT44 = "ANT-44"
    ANT44i = "ANT-44i"
    # --- Arctic domain ---
    ARC22 = "ARC-22"
    ARC44 = "ARC-44"
    ARC44i = "ARC-44i"
    # --- Australasia domain ---
    AUS22 = "AUS-22"
    AUS44 = "AUS-44"
    AUS44i = "AUS-44i"
    # --- Central America domain ---
    CAM22 = "CAM-22"
    CAM44 = "CAM-44"
    # --- Central Asia domain ---
    CAS22 = "CAS-22"
    CAS44 = "CAS-44"
    # --- East Asia domain ---
    EAS22 = "EAS-22"
    EAS44 = "EAS-44"
    # --- Europe domain ---
    EUR11 = "EUR-11"
    EUR22 = "EUR-22"
    EUR44 = "EUR-44"
    EUR44i = "EUR-44i"
    # --- ??? domain ---
    GER0275 = "GER-0275"
    # --- Mediterranean domain ---
    MED11 = "MED-11"
    # --- Meddle East North Africa domain ---
    MNA22 = "MNA-22"
    MNA44 = "MNA-44"
    # --- North America domain ---
    NAM22 = "NAM-22"
    NAM44 = "NAM-44"
    # --- South America domain ---
    SAM20 = "SAM-20"
    SAM22 = "SAM-22"
    SAM44 = "SAM-44"
    # --- South East Asia domain ---
    SEA22 = "SEA-22"
    # --- South Asia domain ---
    WAS22 = "WAS-22"
    WAS44 = "WAS-44"
    WAS44i = "WAS-44i"


class Experiment(str, Enum):
    historical = "historical"
    rcp26 = "rcp26"
    rcp45 = "rcp45"
    rcp85 = "rcp85"
    # ssp126 = "ssp126"
    # ssp585 = "ssp585"


class Project(str, Enum):
    CORDEX = "CORDEX"
    cmip6 = "cmip6"
    cmip3 = "cmip3"
    CMIP5 = "CMIP5"
    CMIP3 = "CMIP3"
    CMIP6_Adjust = "CMIP6-Adjust"
    reklies_index = "reklies-index"
    CORDEX_Adjust = "CORDEX-Adjust"
    CORDEX_ESD = "CORDEX-ESD"
    CORDEX_FPSCONV = "CORDEX-FPSCONV"
    CORDEX_Reklies = "CORDEX-Reklies"
    COSMO_REA = "COSMO-REA"
    c3s_cmip5_adjust = "c3s-cmip5-adjust"
    cordex_adjust = "cordex-adjust"
    input4MIPs = "input4MIPs"
    obs4MIPs = "obs4MIPs"


class DrivingModel(str, Enum):
    """"""

    HadGEM2_CC = "HadGEM2-CC"
    hadGEM2_ES = "hadGEM2-ES"
    MOHC_HadGEM2_CC = "MOHC-HadGEM2-CC"
    MOHC_HadGEM2_ES = "MOHC-HadGEM2-ES"
    ICHEC_EC_EARTH = "ICHEC-EC-EARTH"
    IPSL_CM5A_LR = "IPSL-CM5A-LR"
    IPSL_CM5A_MR = "IPSL-CM5A-MR"
    IPSL_CM5B_LR = "IPSL-CM5B-LR"
    IPSL_IPSL_CM5A_LR = "IPSL-IPSL-CM5A-LR"
    IPSL_IPSL_CM5A_MR = "IPSL-IPSL-CM5A-MR"
    CMCC_CMS = "CMCC-CMS"
    CanESM2 = "CanESM2"
    EC_EARTH = "EC-EARTH"
    ECHAM5 = "ECHAM5"
    NCC_NorESM1_M = "NCC-NorESM1-M"
    NCC_NorESM1_ME = "NCC-NorESM1-ME"
    NorESM1_M = "NorESM1-M"
    NorESM2_MM = "NorESM2-MM"
    MPI_ESM_LR = "MPI-ESM-LR"
    MPI_ESM_MR = "MPI-ESM-MR"
    MPI_M_MPI_ESM_LR = "MPI-M-MPI-ESM-LR"
    MPI_M_MPI_ESM_MR = "MPI-M-MPI-ESM-MR"
    MIROC_ESM = "MIROC-ESM"
    MIROC_ESM_CHEM = "MIROC-ESM-CHEM"
    MIROC_MIROC5 = "MIROC-MIROC5"
    MIROC5 = "MIROC5"
    # CCCma_CanESM2 = "CCCma-CanESM2"
    # CCSM4 = "CCSM4"
    # ACCESS1-0 = "ACCESS1-0"
    # ACCESS1-3 = "ACCESS1-3"
    # BNU-ESM = "BNU-ESM"
    # CMCC-CMCC-CM = "CMCC-CMCC-CM"
    # CNRM_CERFACS-CNRM-CM5 = "CNRM-CERFACS-CNRM-CM5"
    # CNRM_CM5 = "CNRM-CM5"
    # CNRM_ESM2-1 = "CNRM-ESM2-1"
    # CSIRO-BOM-ACCESS1-0 = "CSIRO-BOM-ACCESS1-0"
    # CSIRO-BOM-ACCESS1-3 = "CSIRO-BOM-ACCESS1-3"
    # CSIRO-Mk3-6-0 = "CSIRO-Mk3-6-0"
    # CSIRO-QCCCE-CSIRO-Mk3-6-0 = "CSIRO-QCCCE-CSIRO-Mk3-6-0"
    # ECMWF-ERA40 = "ECMWF-ERA40"
    # ECMWF-ERA5 = "ECMWF-ERA5"
    # ECMWF-ERAINT = "ECMWF-ERAINT"
    # GFDL-ESM2G = "GFDL-ESM2G"
    # GFDL-ESM2M = "GFDL-ESM2M"
    # MRI_CGCM3 = "MRI-CGCM3"
    # NCAR_CCSM4 = "NCAR-CCSM4"
    # NOAA-GFDL-GFDL-CM3 = "NOAA-GFDL-GFDL-CM3"
    # NOAA-GFDL-GFDL-ESM2G = "NOAA-GFDL-GFDL-ESM2G"
    # NOAA-GFDL-GFDL-ESM2M = "NOAA-GFDL-GFDL-ESM2M"
    # SMHI-EC_EARTH = "SMHI-EC-EARTH"
    # bcc-csm1-1-m = "bcc-csm1-1-m"


class RcmModel(str, Enum):
    """"""

    HIRHAM5 = "HIRHAM5"
    HadGEM3_RA = "HadGEM3-RA"
    HadREM3_GA7_05 = "HadREM3-GA7-05"
    HadRM3P = "HadRM3P"
    RCA4 = "RCA4"
    REMO2009 = "REMO2009"
    REMO2015 = "REMO2015"
    RRCM = "RRCM"
    RegCM4 = "RegCM4"
    RegCM4_0 = "RegCM4-0"
    RegCM4_2 = "RegCM4-2"
    RegCM4_3 = "RegCM4-3"
    RegCM4_4 = "RegCM4-4"
    RegCM4_6 = "RegCM4-6"
    RegCM4_7 = "RegCM4-7"
    RACMO22E = "RACMO22E"
    RACMO22T = "RACMO22T"
    RCA4_SN = "RCA4-SN"
    # ALADIN52 = "ALADIN52"
    # ALADIN53 = "ALADIN53"
    # ALADIN63 = "ALADIN63"
    # ALADIN64 = "ALADIN64"
    # ALARO-0 = "ALARO-0"
    # AROME41t1 = "AROME41t1"
    # ARPEGE51 = "ARPEGE51"
    # BOM-SDM = "BOM-SDM"
    # CCAM = "CCAM"
    # CCAM-1704 = "CCAM-1704"
    # CCAM-2008 = "CCAM-2008"
    # CCLM-0-9 = "CCLM-0-9"
    # CCLM4-21-2 = "CCLM4-21-2"
    # CCLM4-8-17 = "CCLM4-8-17"
    # CCLM4-8-17-CLM3-5 = "CCLM4-8-17-CLM3-5"
    # CCLM5-0-15 = "CCLM5-0-15"
    # CCLM5-0-16 = "CCLM5-0-16"
    # CCLM5-0-2 = "CCLM5-0-2"
    # CCLM5-0-6 = "CCLM5-0-6"
    # CCLM5-0-9 = "CCLM5-0-9"
    # CCLM5-0-9-NEMOMED12-3-6 = "CCLM5-0-9-NEMOMED12-3-6"
    # CLM3 = "CLM3"
    # CLM4 = "CLM4"
    # COSMO = "COSMO"
    # COSMO-crCLIM = "COSMO-crCLIM"
    # COSMO-crCLIM-v1-1 = "COSMO-crCLIM-v1-1"
    # CRCM5 = "CRCM5"
    # CRCM5-SN = "CRCM5-SN"
    # CanRCM4 = "CanRCM4"
    # DeepESD-EE = "DeepESD-EE"
    # EPISODES2018 = "EPISODES2018"
    # Eta = "Eta"
    # HCLIM38-AROME = "HCLIM38-AROME"
    # ICON-2024-01 = "ICON-2024-01"
    # ICON2_6_6 = "ICON2-6-6"
    # MAR311 = "MAR311"
    # MAR36 = "MAR36"
    # RA = "RA"
    # RACMO21P = "RACMO21P"
    # SNURCM = "SNURCM"
    # STARS3 = "STARS3"
    # VRF370 = "VRF370"
    # WETTREG2013 = "WETTREG2013"
    # WRF = "WRF"
    # WRF331 = "WRF331"
    # WRF331F = "WRF331F"
    # WRF331G = "WRF331G"
    # WRF341E = "WRF341E"
    # WRF341I = "WRF341I"
    # WRF351 = "WRF351"
    # WRF360J = "WRF360J"
    # WRF360K = "WRF360K"
    # WRF360L = "WRF360L"
    # WRF361H = "WRF361H"
    # WRF381BB = "WRF381BB"
    # WRF381BG = "WRF381BG"
    # WRF381BI = "WRF381BI"
    # WRF381CA = "WRF381CA"
    # WRF381DA = "WRF381DA"
    # WRF381P = "WRF381P"

class Institute(str, Enum):
    ''''''
    AUTH_MC = "AUTH-MC"
    AWI = "AWI"
    BCC = "BCC"
    BCCR = "BCCR"
    BNU = "BNU"
    BOM = "BOM"
    BOUN = "BOUN"
    CALTECH = "CALTECH"
    CAU_GEOMAR = "CAU-GEOMAR"
    CCCMA = "CCCMA"
    CEC = "CEC"
    CLIVAR_GSOP = "CLIVAR-GSOP"
    CLMcom = "CLMcom"
    CLMcom_BTU = "CLMcom-BTU"
    CLMcom_CMCC = "CLMcom-CMCC"
    CLMcom_DWD = "CLMcom-DWD"
    CLMcom_ETH = "CLMcom-ETH"
    CLMcom_GUF = "CLMcom-GUF"
    CLMcom_HZG = "CLMcom-HZG"
    CLMcom_KIT = "CLMcom-KIT"
    CMCC = "CMCC"
    CNRM = "CNRM"
    CNRM_CERFACS = "CNRM-CERFACS"
    COLA_CFS = "COLA-CFS"
    CRNM_CERFACS = "CRNM CERFACS"
    CSIRO = "CSIRO"
    CSIRO_BOM = "CSIRO-BOM"
    CSIRO_QCCCE = "CSIRO-QCCCE"
    CYI = "CYI"
    DHMZ = "DHMZ"
    DMI = "DMI"
    DWD = "DWD"
    ECMWF = "ECMWF"
    ENEA = "ENEA"
    ETH = "ETH"
    FUB = "FUB"
    FUB_DWD = "FUB-DWD"
    FZJ = "FZJ"
    FZJ_IDL = "FZJ-IDL"
    GERICS = "GERICS"
    HCLIMcom = "HCLIMcom"
    HMS = "HMS"
    HU = "HU"
    IAP_UA = "IAP-UA"
    ICHEC = "ICHEC"
    ICTP = "ICTP"
    IITM = "IITM"
    INGV = "INGV"
    INM = "INM"
    INPE = "INPE"
    IPSL = "IPSL"
    IPSL_INERIS = "IPSL-INERIS"
    ISU = "ISU"
    IUP = "IUP"
    JMA = "JMA"
    KNMI = "KNMI"
    KNU = "KNU"
    LASG_CESS = "LASG-CESS"
    LASG_IAP = "LASG-IAP"
    LDEO = "LDEO"
    LOA_IPSL = "LOA-IPSL"
    LOCEAN = "LOCEAN"
    MGO = "MGO"
    MIROC = "MIROC"
    MIUB = "MIUB"
    MOHC = "MOHC"
    MPI_CSC = "MPI-CSC"
    MPI_M = "MPI-M"
    MRI = "MRI"
    MRI_JMA = "MRI-JMA"
    NASA_Ames = "NASA-Ames"
    NASA_GISS = "NASA-GISS"
    NASA_GMAO = "NASA-GMAO"
    NASA_GSFC = "NASA-GSFC"
    NASA_LaRC = "NASA-LaRC"
    NASA_NCCS = "NASA-NCCS"
    NCAR = "NCAR"
    NCAS = "NCAS"
    NCC = "NCC"
    NCEI = "NCEI"
    NCEP = "NCEP"
    NICAM = "NICAM"
    NIMR_KMA = "NIMR-KMA"
    NIMS_KMA = "NIMS-KMA"
    NOAA_ESRLandCIRES = "NOAA-ESRLandCIRES"
    NOAA_GFDL = "NOAA-GFDL"
    NUIM = "NUIM"
    ORNL = "ORNL"
    OURANOS = "OURANOS"
    PCMDI = "PCMDI"
    PIK = "PIK"
    PNNL = "PNNL"
    PNU = "PNU"
    POSTECH = "POSTECH"
    RMIB_UGent = "RMIB-UGent"
    RU_CORE = "RU-CORE"
    SCU = "SCU"
    SMHI = "SMHI"
    SU = "SU"
    UA = "UA"
    UAlbany = "UAlbany"
    UCAN = "UCAN"
    UHOH = "UHOH"
    ULAQ = "ULAQ"
    ULSAN = "ULSAN"
    ULg = "ULg"
    UNIST = "UNIST"
    UNSW = "UNSW"
    UOE = "UOE"
    UOED = "UOED"
    UQAM = "UQAM"
    UW = "UW"
    UW_MADISON = "UW-MADISON"
    University_College_London = "University College London"
    University_Hamburg = "University-Hamburg"