import os
import numpy as np
import pandas as pd
import  numpy.polynomial.chebyshev as cheb

'''
This script are referred from the url below. I add a few changes for our codes.
https://github.com/BensonRen/AEM_DIM_Bench/blob/1ff82bfdcd6b0a736bf184f0bcb8a533743aacbb/Data/Chen/generate_chen.py
'''


def sind(x):
    y = np.sin(np.radians(x))
    I = x/180.
    if type(I) is not np.ndarray:   #Returns zero for elements where 'X/180' is an integer
        if(I == np.trunc(I) and np.isfinite(I)):
            return 0
    return y

def jreftran_rt(wavelength, d, n, t0, polarization, M=None, M_t=None):
    if M is None:  # 在反复多次调用的情况下，可以一次性分配M,M_t
        M = np.zeros((2, 2, d.shape[0]), dtype=complex)
    if M_t is None:
        M_t = np.identity(2, dtype=complex)

    # x = sind(np.array([0,90,180,359, 360]))
    Z0 = 376.730313  # impedance of free space, Ohms
    Y = n / Z0
    g = (
        1j * 2 * np.pi * n / wavelength
    )  # propagation constant in terms of free space wavelength and refractive index
    t = n[0] / n * sind(t0)
    t2 = (
        t * t
    )  # python All arithmetic operates elementwise,Array multiplication is not matrix multiplication!!!
    ct = np.sqrt(1 - t2)
    # ct=sqrt(1-(n(1)./n*sin(t0)).^2); %cosine theta
    if polarization == 0:  # tilted admittance(斜导纳)
        eta = Y * ct
        # tilted admittance, TE case
    else:
        eta = Y / ct
        # tilted admittance, TM case
    delta = 1j * g * d * ct
    ld = d.shape[0]
    # M = np.zeros((2, 2, ld),dtype=complex)
    for j in range(ld):
        a = delta[j]
        M[0, 0, j] = np.cos(a)
        M[0, 1, j] = 1j / eta[j] * np.sin(a)
        M[1, 0, j] = 1j * eta[j] * np.sin(a)
        M[1, 1, j] = np.cos(a)
        # ("M(:,:,{})={}\n\n".format(j,M[:,:,j]))
    # M_t = np.identity(2,dtype=complex)        #toal charateristic matrix
    for j in range(1, ld - 1):
        M_t = np.matmul(M_t, M[:, :, j])
    # s1 = '({0.real:.2f} + {0.imag:.2f}i)'.format(eta[0])
    # np.set_printoptions(precision=3)
    # print("M_t={}\n\neta={}".format(M_t,eta))

    e_1, e_2 = eta[0], eta[-1]
    # m_1, m_2 = M_t[0, 0] + M_t[0, 1] * e_2, M_t[1, 0] + M_t[1, 1] * e_2
    De = M_t[0, 0] + M_t[0, 1] * eta[-1]
    Nu = M_t[1, 0] + M_t[1, 1] * eta[-1]
    if False:  # Add by Dr.zhu
        Y_tot = (M_t(2, 1) + M_t(2, 2) * eta(len(d))) / (
            M_t(1, 1) + M_t(1, 2) * eta(len(d))
        )
        eta_one = eta[0]
        Re = Y_tot.real
        Im = Y_tot.imag
        fx = 2 * Im * eta[0]

    e_de_nu = e_1 * De + Nu
    r = (e_1 * De - Nu) / e_de_nu
    t = 2 * e_1 / e_de_nu

    R = abs(r) * abs(r)
    T = (e_2.real / e_1) * abs(t) * abs(t)
    T = T.real
    a = De * np.conj(Nu) - e_2
    A = (4 * e_1 * a.real) / (abs(e_de_nu) * abs(e_de_nu))
    A = A.real
    # return r,t,R,T,A,Y_tot,eta_one,fx,Re,Im
    return r, t, R, T, A


class N_Dict(object):
    map2 = {}

    def __init__(self,config=None):
        self.dicts = {}
        self.config = config
        return

    def InitMap2(self, maters, lendas):
        for mater in maters:
            for lenda in lendas:
                self.map2[mater, lenda] = self.Get(mater, lenda, isInterPolate=True)

    def Load(self, material, path, scale=1):
        df = pd.read_csv(path, delimiter="\t", header=None, names=['lenda', 're', 'im'], na_values=0).fillna(0)
        if scale is not 1:
            df['lenda'] = df['lenda'] * scale
            df['lenda'] = df['lenda'].astype(int)
        self.dicts[material] = df
        rows, columns = df.shape
        # if columns==3:
        #print("{}@@@{} shape={}\n{}".format(material, path, df.shape, df.head()))

    def Get(self, material, lenda, isInterPolate=False):
        # lenda = 1547
        n = 1 + 0j
        if material == "air":
            return n
        if material == "Si3N4":
            if self.config is None or self.config.model=='v1':
                return 2. + 0j
            else:
                return 2.46 + 0j
            #return 2.0
        assert self.dicts.get(material) is not None
        df = self.dicts[material]
        assert df is not None
        pos = df[df['lenda'] == lenda].index.tolist()
        # assert len(pos)>=1
        if len(pos) == 0:
            if isInterPolate:
                A = df['lenda'].values  #CHANGE BY A.M. df.as_matrix(columns=['lenda'])
                idx = (np.abs(A - lenda)).argmin()
                if idx == 0:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                else:
                    lenda_1, re_1, im_1 = df['lenda'].loc[idx - 1], df['re'].loc[idx - 1], df['im'].loc[idx - 1]
                lenda_2, re_2, im_2 = df['lenda'].loc[idx], df['re'].loc[idx], df['im'].loc[idx]
                re = np.interp(lenda, [lenda_1, lenda_2], [re_1, re_2])
                im = np.interp(lenda, [lenda_1, lenda_2], [im_1, im_2])
            else:
                return None
        elif len(pos) > 1:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        else:
            re, im = df['re'].loc[pos[0]], df['im'].loc[pos[0]]
        n = re + im * 1j
        return n


def chen_simulate(Xpred, cfg):
    """
    Generates y from x data. This code refers from the code below.
    https://github.com/BensonRen/AEM_DIM_Bench/blob/1ff82bfdcd6b0a736bf184f0bcb8a533743aacbb/Data/Chen/generate_chen.py#L508
    """
    lambda_0 = 240
    lambda_f = 2000
    n_spct = 256

    FOLDER_PATH = cfg.simulator_files_dir.Stack

    cheb_plot = cheb.chebpts2(n_spct)
    scale = (lambda_f - lambda_0) / 2
    offset = lambda_0 + scale
    scaled_cheb = [i * scale + offset for i in cheb_plot]

    # Load refractive indices of materials at different sizes
    n_dict = N_Dict()
    n_dict.Load(
        "Si3N4", os.path.join(FOLDER_PATH, "Si3N4_310nm-14280nm.txt"), scale=1000
    )
    n_dict.Load("Graphene", os.path.join(FOLDER_PATH, "Graphene_240nm-30000nm.txt"))
    n_dict.InitMap2(["Si3N4", "Graphene"], scaled_cheb)
    map2 = n_dict.map2

    Ypred = np.zeros((Xpred.shape[0], n_spct))

    for c, x in enumerate(Xpred):
        dataY = np.zeros((n_spct, 3))

        t_layers = np.array([])
        for val in x:
            # Graphene is preset to always be 0.35 thick
            t_layers = np.concatenate((t_layers, np.array([0.35, val])))
        t_layers = np.concatenate((np.array([np.nan]), t_layers, np.array([np.nan])))

        # Each y-point has different ns in each layer
        for row, lenda in enumerate(scaled_cheb):
            n_layers = np.array([])
            for i in range(len(t_layers) - 2):
                # -2 added to subtract added substrate parameters from length
                if i % 2 == 0:
                    n_layers = np.concatenate(
                        (n_layers, np.array([map2["Graphene", lenda]]))
                    )
                else:
                    n_layers = np.concatenate(
                        (n_layers, np.array([map2["Si3N4", lenda]]))
                    )
            # Sandwich by substrate
            n_layers = np.concatenate(
                (np.array([1.46 + 0j]), n_layers, np.array([1 + 0j]))
            )

            # TARGET PARAMETERS OF THE PROBLEM!
            xita = 0
            polar = 0

            r, t, R, T, A = jreftran_rt(lenda, t_layers, n_layers, xita, polar)

            dataY[row, 0] = R
            dataY[row, 1] = T
            dataY[row, 2] = A

        # Because graphs are of absorbance 2nd value is selected
        Ypred[c, :] = dataY[0:n_spct, 2]

    return Ypred


def Stack_task_simulator(input: np.array, cfg) -> np.array:
    """
    Stack task (Chen) simulator. This code refers from the code below.
    https://github.com/BensonRen/AEM_DIM_Bench/blob/1ff82bfdcd6b0a736bf184f0bcb8a533743aacbb/utils/helper_functions.py#L196
    Args:
        input(np.array) : input data for the simulator
        cfg : cfg file.
    Outputs:
        signal(np.array) : output data for the simulator
    """
    assert len(input.shape)==2
    assert input.shape[-1] == 5
    output = input * 22.5 + 27.5
    return chen_simulate(output, cfg)


