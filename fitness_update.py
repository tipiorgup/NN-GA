#!/home/lgomez/anaconda3/envs/dftb-nn/bin/python3.6
import os                                                              
import time   
import pickle
import math
import numpy as np                                                     
import pandas as pd   
import seaborn as sns                                                  
import matplotlib.pyplot as plt    
import random                                    
                                                                       
from sklearn.metrics import mean_absolute_error                        
from sklearn.metrics import mean_squared_error                         
from sklearn.model_selection import train_test_split                   
from sklearn.preprocessing import StandardScaler                       
from sklearn.model_selection import train_test_split                   
                                                                       
import tensorflow.keras                                                           
from tensorflow.keras.optimizers import Adam                                      
from tensorflow.keras.models import Model, Sequential                             
from tensorflow.keras.layers import Input, Dense, Dropout                  
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping   
                                                                       
import ase                                                             
from ase import io
import ase.build                                                       
from ase import Atoms                                                  
from ase.atoms import Atoms                                            
from ase.io import read, write                                         
from ase.calculators.dftb import Dftb                                  
from ase.units import Hartree, mol, kcal, Bohr                       
                                                                       
from Calculator import src_nogrd                                                       
from Calculator.src_nogrd import sym_func_show                                    
from Calculator.src_nogrd import xyzArr_generator                                 
from Calculator.src_nogrd import feat_scaling_func                                
from Calculator.src_nogrd import at_idx_map_generator
from Calculator.src_nogrd import at_idx_map_generator_old
from Calculator.store_models import write_subnet_text
                                                                                                                           
import pickle
from itertools import combinations_with_replacement as comb_replace
                                                                       
import Utils.DirNav                                                
from Utils.dftb_traj_io import read_scan_traj
import Utils.netBuilder
from Utils.netBuilder import netBuilder
from Utils.netTrainer import netTrainer

train_name = 'data'
save_name = 'data_model'
root_dir = '/nfs/home/lgomez/dftb-nn/FF/'
train_dir = os.path.join(root_dir, train_name)
save_dir = os.path.join(root_dir, save_name)

geom_filename          = os.path.join(train_dir, 'eTDs.xyz')
md_train_arr_origin    = read_scan_traj(filename=geom_filename)
md_low_rel_e_arr_name  = os.path.join(train_dir, 'DFTB.csv')
md_high_rel_e_arr_name = os.path.join(train_dir, 'DFT.csv')
try:
    md_dftb_ref_e_arr = np.loadtxt(md_low_rel_e_arr_name)
    md_pbe_ref_e_arr = np.loadtxt(md_high_rel_e_arr_name)
except:
    raise OSError("Cannot Read md_calc_rel_e_arr")
    #os.rmdir(save_dir)
    
print(md_dftb_ref_e_arr, md_pbe_ref_e_arr)
print("Before Copy")
print(md_train_arr_origin)
#WARNING: To Get the sample without index
md_train_arr = md_train_arr_origin.copy(deep=False).reset_index(drop=True)
print("After Copy")
print(md_train_arr)

md_rel_energy_arr = md_pbe_ref_e_arr - md_dftb_ref_e_arr
# Get rid of error values.
nan_index = np.where(np.isnan(md_rel_energy_arr))

for idx in nan_index:
    md_train_arr.drop(idx)
    md_rel_energy_arr = md_rel_energy_arr[~np.isnan(md_rel_energy_arr)]

struct = io.read(geom_filename, format = "xyz")
elem = struct.get_chemical_symbols()
SUPPORTED_ELEMENTS = list(dict.fromkeys(elem))


# Maintainence: Natom is a global variable 
# Assumes that the configuration does not change the Number of atoms. 
nAtoms, xyzArr = xyzArr_generator(md_train_arr)

# Calculate distance dataframe from xyz coordinates
distances = src_nogrd.distances_from_xyz(xyzArr, nAtoms)

at_idx_map_naive = at_idx_map_generator_old(md_train_arr[0])
## Hotfix for atom ordering without touching at_idx_map_generator
at_idx_map_old = {el : at_idx_map_naive[el] for el in SUPPORTED_ELEMENTS}
print(at_idx_map_old)

at_idx_map = at_idx_map_generator(md_train_arr[0])
print(at_idx_map)
print(at_idx_map_naive)


def individual(variables, min, max):
    'Create a member of the population.'
    return [random.randint(min,max) for x in range(variables)]

def population(count, variables, min, max):
    return [individual(variables, min, max) for x in range(count)]

def fitness(individual, target):
    #genome variables, ie which variables you want to optimize
    g=individual[0] #grid
    r=individual[1] #cut off rad
    a=individual[2] #cut off ang
    m=individual[3] #max distance

    # radial symmetry function parameters
    # Need to automate the Rs_array part
    cutoff_rad = r
    #Rs_array = np.linspace(0.8, 5, num=24)   # based on max and min of the distances
    Rs_array = np.linspace(0.2, m, num=g)   # based on max and min of the distances
    eta_array = 5/(np.square(0.2*Rs_array))
    rad_params = np.array([(Rs_array[i], eta_array[i], cutoff_rad) for i in range(len(Rs_array)) ])

    # angular symmetry function parameters
    cutoff_ang = a
    lambd_array = np.array([-1, 1])
    #zeta_array = np.array([1, 4, 16])
    zeta_array = np.array([1,4,16])
    #eta_ang_array = np.array([0.001, 0.01, 0.05])
    eta_ang_array = np.array( [0.001, 0.01, 0.05])
            
    # Each of the element need to be parametrized for all of the list. 
    angList = np.array([e1+e2 for e1, e2 in comb_replace(SUPPORTED_ELEMENTS, 2)])
    # print(angList)
    ang_comp = {el : angList for el in SUPPORTED_ELEMENTS}
    ang_params = np.array([[eta, zeta, lambd, cutoff_ang] for eta in eta_ang_array for zeta in zeta_array for lambd in lambd_array])

    Gparam_dict = {}
    for at_type in at_idx_map.keys():
        Gparam_dict[at_type] = {}
        Gparam_dict[at_type]['rad'] = {}
        for at2_rad in at_idx_map.keys():
            Gparam_dict[at_type]['rad'][at2_rad] = rad_params

        # This Section is already designed to be general
        Gparam_dict[at_type]['ang'] = {}
        for at23_ang in ang_comp[at_type]:
            Gparam_dict[at_type]['ang'][at23_ang] = ang_params


    def cutoff(Rc, Rij):
        """ Cutoff function for both angular and radial symmetry function
            Args:
                Rc: the cutoff radius
                Rij: (arr) distance between two atoms of index i and j.

            Outputs:
                cutoff_arr: the value of the cut off function
                f = 0.5 * (Cos[ pi * Rij / Rc] + 1)

        """
        const = 1 / Rc

        cutoff_arr = 0.5 * ( np.cos( np.pi * Rij * const ) + 1) *  (Rij < Rc)

        return cutoff_arr


    def radial_filter(Rs, eta, Rij):
        """radial filter for symmetry functions
        # Arguments
            Rs, eta: radial symmetry function parameters; float
            Rij: distance values between two given atoms i and j;
                    1D numpy array of length Nsamples

        # Returns
            G_rad_ij: radial filter values; 1D numpy array of length nb_samples
        """
        G_rad_ij = np.exp(-eta * (Rij-Rs)**2)
        return G_rad_ij

    def angular_filter(Rij, Rik, Rjk, eta, zeta, lambd):
        """angular filter for angular symmetry functions
        # Arguments
            eta, zeta, lambd: angular symmetry function parameters
            Rij, Rik, Rjk: distances among three atoms i, j, k; 1D arrays of length nb_samples

        # Returns
            G_ang_ij: angular filter values; 1D numpy array of length nb_samples

        """
        cos_angle = (Rij**2 + Rik**2 - Rjk**2)/(2.0 * Rij * Rik)
        rad_filter = np.exp(-eta*(Rij + Rik + Rjk)**2)
        G_ang_ijk = 2**(1.0-zeta) * (1.0 + lambd * cos_angle)**zeta * rad_filter


        return G_ang_ijk


    Nsamples = distances.shape[0]
    Gfunc_data = pd.Series([])

    # This for loop goes through elements
    # Are together
    for at_type in at_idx_map.keys():
        Gparam_rad = Gparam_dict[at_type]['rad']
        Gparam_ang = Gparam_dict[at_type]['ang']

        Gfunc_data[at_type] = pd.Series([])

        rad_count = sum([Gparam_rad[t].shape[0] for t in Gparam_rad.keys()])
        ang_count = sum([Gparam_ang[t].shape[0] for t in Gparam_ang.keys()])


        ## This for loop goes through all the atoms (belong to the same element)
        for at1 in at_idx_map[at_type]:
            Gfunc_data[at_type][at1] = np.zeros((Nsamples, rad_count + ang_count))

            G_temp_count = 0

            # radial components
            for at2_type in Gparam_rad.keys():
                comp_count =  Gparam_rad[at2_type].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))

                for count, values in enumerate(Gparam_rad[at2_type]):
                    #pdb.set_trace()
                    for at2 in at_idx_map[at2_type][at_idx_map[at2_type]!=at1]:
                        # Problem Located: The following code does not work.
                        # The dist does not put into the allowance.
                        dist = tuple(sorted([at1, at2]))
                        #pdb.set_trace()
                        R12_array = distances[dist].values[:Nsamples]
                        # values[0] = Rs, values[1] = eta (integer, not array), values[2] = Rc (cutoff)
                        # Then Calculate the radial symmetric function -> value of G.
                        rad_temp = radial_filter(values[0], values[1], R12_array) * cutoff(values[2], R12_array)
                        G_temp_component[:,count] += rad_temp

                Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
                G_temp_count += comp_count

            # ======================
            # angular components
            for atAatB_type in Gparam_ang.keys():
                comp_count = Gparam_ang[atAatB_type].shape[0]
                G_temp_component = np.zeros((Nsamples, comp_count))
                                # This for loop goes through all 'HH', 'HO' combo?
            for count, values in enumerate(Gparam_ang[atAatB_type]):
                    atA_list = at_idx_map[atAatB_type[0]][at_idx_map[atAatB_type[0]]!=at1]
                    for atA in atA_list:
                        dist_1A = tuple(sorted([at1, atA]))
                        R1A_array = distances[dist_1A].values[:Nsamples]

                        if atAatB_type[0] == atAatB_type[1]:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1) & (at_idx_map[atAatB_type[1]]>atA)]
                        else:
                            atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=at1)]

                        for atB in atB_list:
                            dist_1B = tuple(sorted([at1, atB]))
                            dist_AB = tuple(sorted([atA, atB]))
                            R1B_array = distances[dist_1B].values[:Nsamples]
                            RAB_array = distances[dist_AB].values[:Nsamples]

                            if np.any(R1B_array == 0):
                                import pdb; pdb.set_trace()
                            if np.any(RAB_array == 0):
                                import pdb; pdb.set_trace();
                            ang_temp = angular_filter(R1A_array, R1B_array, RAB_array, values[0], values[1], values[2]) \
                                        * cutoff(values[3], R1A_array) * cutoff(values[3], R1B_array) * cutoff(values[3], RAB_array)

                            G_temp_component[:, count] += ang_temp
            Gfunc_data[at_type][at1][:,G_temp_count:G_temp_count+comp_count] = G_temp_component
            G_temp_count += comp_count

    n_symm_func =Gfunc_data[SUPPORTED_ELEMENTS[0]][0][0].shape[0]


    input_dim_list = {}
    for at_type in at_idx_map:
        input_dim_list[at_type] = (n_symm_func, 1)


    builder = netBuilder(SUPPORTED_ELEMENTS, n_symm_func)
    subnets = builder.build_subnets(n_dense_layers=3, n_units=34, 
                        hidden_activation='tanh',
                        dropout_type="NoFirstDrop", dropout_ratio=0.2)
    model = builder.build_molecular_net(at_idx_map, subnets)


    def idx_generator(n_samples, val_ratio, test_ratio):
        """
        Function:
        Randomly shuffle the indexes and to generate indexes for the training, validation and test set.
        
            Args:
                n_samples: number of samples, an interger
                val_ratio: ratio of the validation set (compared with all data set)
                test_ratio: 
        
            Warning: 0 < val_ratio + test_ratio < 1.
        
            Output:
                train_idx: indexes for training set
                val_idx: indexes for the validation set
                test_idx: indexes for the test set    
        """
        if val_ratio + test_ratio >= 1 or val_ratio + test_ratio <= 0:
            raise  ValueError("idx_generator: the val_ratio and test_ratio must be in between 0 and 1")
        
        shuffled_indices = np.random.permutation(n_samples)
        
        
        val_set_size = int(n_samples * val_ratio)
        val_idx  = shuffled_indices[:val_set_size]
        
        test_set_size= int(n_samples * val_ratio)
        test_idx = shuffled_indices[val_set_size:val_set_size+test_set_size]
        
        train_idx = shuffled_indices[val_set_size + test_set_size:]
        
        return train_idx, val_idx, test_idx
        
    ## Split the Training, Validation & Test Data 
    n_samples = len(md_train_arr)
    train_idx, val_idx, test_idx = idx_generator(n_samples, 0.2,0.2)
    #print(train_idx.shape)
    #print(val_idx.shape)
    # Check whether it is totally splitted
    if train_idx.shape[0] + test_idx.shape[0] + val_idx.shape[0] != n_samples:
        raise ValueError("Splitting Test does not equal to the entire set!")
        


    # rescale target values 
    # All training values in kcal/mol unit

    y_train = md_rel_energy_arr[train_idx] 
    y_val   = md_rel_energy_arr[val_idx]
    y_test  = md_rel_energy_arr[test_idx]

    #print('y_train min, max = ', '%.5f  %.5f' %(y_train.min(), y_train.max() ))
    #print('y_test min, max = ', '%.5f  %.5f' %(y_test.min(), y_test.max()) )

    def split_training_data(Feat_data, at_idx_map, train_idx, val_idx, test_idx):
        """
        Function:
        Split the training set, 
            
        Input:
        Feat_data_train: Strucutre for the feat data Feat_data_train['element'][atom]
        at_idx_map: Atom Index Map
        train_idx: the indices used for the training 
        
        
        Output:
        Return the Feat_train, Feat_val and Feat_test set in the shape
        Feat_scaler['element'][atom][Feature Number]    
        """
        
        
        Feat_train_scaled = {}
        Feat_val_scaled = {}
        Feat_test_scaled = {}

        
        for at_type in at_idx_map.keys():
            Feat_train_scaled[at_type] = {}
            Feat_val_scaled[at_type] = {}
            Feat_test_scaled[at_type] = {}
            
            for at in at_idx_map[at_type]:
                Feat_train_scaled[at_type][at] = Feat_data[at_type][at][train_idx,]
                #import pdb; pdb.set_trace()
                Feat_val_scaled[at_type][at]   = Feat_data[at_type][at][val_idx,]
                Feat_test_scaled[at_type][at]  = Feat_data[at_type][at][test_idx,]
                

                
        
        return Feat_train_scaled, Feat_val_scaled, Feat_test_scaled
        


    train_scaled, val_scaled, test_scaled = split_training_data(Gfunc_data, at_idx_map, train_idx, val_idx, test_idx)
    #print(test_scaled['C'][0].shape)
    #Feat_train_scaled, Feat_val_scaled, Feat_test_scaled = split_training_data(Feat_data, at_idx_map, train_idx, val_idx, test_idx)
    #print(Feat_train_scaled['H'][4].shape)

    inp_train = []
    inp_val   = []
    inp_test  = []
    for at_type in at_idx_map.keys():

        for atA in at_idx_map[at_type]:
            inp_train.append(train_scaled[at_type][atA])
            #inp_train.append(Feat_train_scaled[at_type][atA])
            
            inp_val.append(val_scaled[at_type][atA])
            #inp_val.append(Feat_val_scaled[at_type][atA])
            
            inp_test.append(test_scaled[at_type][atA])
            #inp_test.append(Feat_test_scaled[at_type][atA])
            
    def get_inp(at_idx_map, Gfunc_scaled, Feat_scaled):
        inp_arr = []
        for at_type in at_idx_map.keys():
            for at_idx in at_idx_map[at_type]:
                inp_arr.append(Gfunc_scaled[at_type][at_idx])
                inp_arr.append(Feat_scaled[at_type][at_idx])
        
        return pd.Series(inp_arr)
        
    model_folder = "model_2B_nUnit"
    trainer = netTrainer(model, verbose=1, folder=model_folder)
    nUnit = n

    check1 = model_folder +'/' + str(nUnit) + '.hdf5'
    checkpointer = ModelCheckpoint(filepath=check1, verbose=0,  monitor='val_mean_squared_error',\
                                mode='min', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=100, verbose=0)

    #Mode == 'NoFirstDrop':

    def repeat(times, f):
        for i in range(times): f()
    def session():
        model.compile(loss='mean_squared_error',
                        optimizer=Adam(lr=0.01,decay=0.01),
                        metrics=['mean_squared_error', 'mean_absolute_error'])
        
        history = model.fit(inp_train, y_train, \
                            callbacks=[checkpointer, earlystop],
                            batch_size=64, epochs=100, shuffle=True,
                            verbose=0, validation_data=(inp_val, y_val)
                            )
    repeat(3, session)
                        
    # Load model weights
    model.load_weights(check1)

    # Error on TEST set 
    y_pred_scaled = model.predict(inp_test)
    y_pred = y_pred_scaled.T[0]  # in kcal/mol unit
    y_obs = y_test #/Eunit

    err_test = np.sqrt(mean_squared_error(y_pred, y_obs))
    errAbs_test = mean_absolute_error(y_pred, y_obs) 

    save_dir

    sys.stdout = open('%s-%s-%s-%s.txt' %(g,r,a,m), "w")

    print('RMSE_test:', '%.4f' % err_test)
    print(n_symm_func)

    sys.stdout.close()
    
    return abs(target-np.sqrt(mean_squared_error(y_pred, y_obs)))

def grade(pop, target):
    'Find average fitness for a population.'
    summed = 0.0
    for x in pop:
        summed += fitness(x,target)
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
            
    # mutate some individuals
    for individual in parents:
        if mutate > random.random():
            pos_to_mutate = random.randint(0, len(individual)-1)
            # function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = random.randint(
                min(individual), max(individual))
    
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents

target = 0
p_count = 20 #population, set minimum of 10
i_length = 4 #number of genes
# set the range of the parameters to optimize
i_min = 3  
i_max = 10 
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]

for i in range(5):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))


