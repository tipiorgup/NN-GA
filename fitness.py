#!/home/lgomez/anaconda3/envs/dftb-nn/bin/python3.6
import os  
import sys                                                            
import time   
import pickle
import numpy as np                                                     
import pandas as pd   
import seaborn as sns                                                  
import matplotlib.pyplot as plt                                        
                                                                       
from sklearn.metrics import mean_absolute_error                        
from sklearn.metrics import mean_squared_error                         
from sklearn.model_selection import train_test_split                   
from sklearn.preprocessing import StandardScaler                       
from sklearn.model_selection import train_test_split                   
                                                                       
import keras                                                           
from keras.optimizers import Adam                                      
from keras.models import Model, Sequential                             
from keras.layers import Input, Dense, Dropout, merge                  
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping   
                                                                       
import ase                                                             
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

train_name = 'data'
save_name = 'data_model'
root_dir = '/nfs/home/lgomez/dftb-nn/FF/'
train_dir = os.path.join(root_dir, train_name)
save_dir = os.path.join(root_dir, save_name)
print(train_dir)
print(save_dir)

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


# Maintainence: Natom is a global variable 
# Assumes that the configuration does not change the Number of atoms. 
nAtoms, xyzArr = xyzArr_generator(md_train_arr)

# Calculate distance dataframe from xyz coordinates
distances = src_nogrd.distances_from_xyz(xyzArr, nAtoms)
distances.head()
print(distances.shape)

SUPPORTED_ELEMENTS = ['H', 'C', 'S']

at_idx_map_naive = at_idx_map_generator_old(md_train_arr[0])
## Hotfix for atom ordering without touching at_idx_map_generator
at_idx_map_old = {el : at_idx_map_naive[el] for el in SUPPORTED_ELEMENTS}
print(at_idx_map_old)

at_idx_map = at_idx_map_generator(md_train_arr[0])
print(at_idx_map)
print(at_idx_map_naive)

distances.max

import random

def individual(variables, min, max):
    'Create a member of the population.'
    return [random.randint(min,max) for x in range(variables)]

def population(count, variables, min, max):
    return [individual(variables, min, max) for x in range(count)]

def fitness(individual, target):
    #genome variables
    g=individual[0]
    r=individual[1]
    a=individual[2]
    m=individual[3]
    s=individual[4]
    b=individual[5]*10
    n=15

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
    #for at_type in Gparam_dict.keys():
     #   print(Gparam_dict[at_type]['rad'].keys())
            

    #print(rad_params)
    #print(ang_params)

    path = "/nfs/home/lgomez/dftb-nn/src"
    rad_name = "symFunc_rad.param"
    ang_name = "symFunc_ang.param"
    with open(os.path.join(path, rad_name), "w") as rad_f:
        rad_f.write(str(cutoff_rad)+"\n")
        for dist, eta in zip(Rs_array, eta_array):
            rad_f.write(f"{dist} {eta}\n")

    with open(os.path.join(path, ang_name), "w") as ang_f:
        ang_f.write(str(cutoff_ang)+"\n")
        for row in ang_params:
            ang_f.write(f"{row[0]} {row[1]} {row[2]}\n")

    Gfunc_data = src_nogrd.symmetry_function(distances, at_idx_map, Gparam_dict)



    nUnit = n #int(os.environ['unit_number'])

    def Nguyen_Widrow_init(shape, dtype=None):
        '''Nguyen-Widrow initialization for weights and biases
        Initialize the weight and biases for each layer.

                Args:
                    shape: ???? (Guess: Shape of the layer to create)

                Output:
                    [w,b]
                    w: represents an array of weight
                    b: represents an array of bias

        Comment:
        This is the function provided by Nguyen's original code.
        Junmian Zhu did the documentation. It is not 100% sure about the
        algorithm used here.

        '''
        n_input, n_unit = shape[0], shape[1]  # n_input: number of units of prev. layer without bias
        w_init = np.random.rand(n_input, n_unit) *2 -1
        norm = 0.7 * n_unit ** (1. / n_input)
        # normalize
        w = norm * w_init/np.sqrt(np.square(w_init).sum(axis=0).reshape(1, n_unit))
        if n_unit>1:
            b = norm * np.linspace(-1,1,n_unit) * np.sign(w[0,:])
        else:
            b = np.zeros((n_unit,))
        return [w, b]

    def create_base_network_NoFirstDrop(input_dim, name):
        '''3-layer base network to be shared among atoms of the same species
        '''
        seq = Sequential()
        seq.add(Dense(nUnit, input_shape=(input_dim,), activation='tanh'))
        shape0 = (seq.layers[0].input_shape[1], seq.layers[0].output_shape[1])
        seq.layers[0].set_weights(Nguyen_Widrow_init(shape0))
        #seq.add(Dropout(0.1))

        seq.add(Dense(nUnit, activation='tanh'))
        shape1 = (seq.layers[1].input_shape[1], seq.layers[1].output_shape[1])
        seq.layers[1].set_weights(Nguyen_Widrow_init(shape1))
        seq.add(Dropout(rate=0))

        seq.add(Dense(nUnit, activation='tanh'))
        shape2 = (seq.layers[3].input_shape[1], seq.layers[3].output_shape[1])
        seq.layers[3].set_weights(Nguyen_Widrow_init(shape2))
        seq.add(Dropout(rate=0.015))

        seq.add(Dense(1, activation='linear'))
        shape3 = (seq.layers[5].input_shape[1], seq.layers[5].output_shape[1])
        seq.layers[5].set_weights(Nguyen_Widrow_init(shape3))
        seq.add(Dropout(rate=0.05))

        seq.name = name
        return seq

    def create_base_network_NoDrop(input_dim, name):
        '''3-layer base network to be shared among atoms of the same species
        No Dropout Layers
        '''
        seq = Sequential()
        seq.add(Dense(nUnit, input_shape=(input_dim,), activation='tanh'))
        shape0 = (seq.layers[0].input_shape[1], seq.layers[0].output_shape[1])
        seq.layers[0].set_weights(Nguyen_Widrow_init(shape0))

        seq.add(Dense(nUnit, activation='tanh'))
        shape1 = (seq.layers[1].input_shape[1], seq.layers[1].output_shape[1])
        seq.layers[1].set_weights(Nguyen_Widrow_init(shape1))

        seq.add(Dense(nUnit, activation='tanh'))
        shape2 = (seq.layers[2].input_shape[1], seq.layers[2].output_shape[1])
        seq.layers[2].set_weights(Nguyen_Widrow_init(shape2))

        seq.add(Dense(1, activation='linear'))
        shape3 = (seq.layers[3].input_shape[1], seq.layers[3].output_shape[1])
        seq.layers[3].set_weights(Nguyen_Widrow_init(shape3))

        seq.name = name
        return seq

    def main_network(input_dim_list, at_idx_map, DropOut):
        '''
        Main neural network with shared sub-structures.
        Currently it supports a sub-network with additional features.
        '''
        type_nb = len(input_dim_list)
        element_network = {}
        inp = {'element':{}, 'rc':{}}

        element_processed = {}  # Will keep the input of the atomic symmetry function
        #rc_processed = []       # Will keep the input of the features, Mulliken Charge of the atom, etc.

        # The sub-network that computes E(RC), the correction for geometry
        # dims[2] = number of features in the second input.
        # Assume that it must have hydrogen
        #rc_network = create_base_network(input_dim_list['H'][1], "rc-subnet")
        # atomic inputs of the same species are processed through the same sub-network.


        #for at_type in range(type_nb):
        for at_type in at_idx_map:
            dims = input_dim_list[at_type]
            # Create a sub-network for each element.
            if DropOut == 'NoDrop':
                element_network[at_type] = create_base_network_NoDrop(dims[0], at_type + '-subnet')
            elif DropOut == 'NoFirstDrop':
                element_network[at_type] = create_base_network_NoFirstDrop(dims[0], at_type + '-subnet')

            element_processed[at_type] = {}
            inp['element'][at_type] = {}
            #inp['rc'][at_type]  = {}
            #for atA in range(dims[0]):
            for atA in at_idx_map[at_type]:
                # input preparation
                inp['element'][at_type][atA] = Input(shape=(dims[0],),dtype='float32', name=at_type+'-'+str(atA)+'-ele')
                #inp['rc'][at_type][atA]      = Input(shape=(dims[1],),dtype='float32', name=at_type+'-'+str(atA)+'-rc')

                # because we re-use the same instance `base_network`,
                # the weights of the sub-network will be shared across same-type atomic branches
                element_processed[at_type][atA] = element_network[at_type](inp['element'][at_type][atA])
                #rc_processed.append(rc_network(inp['rc'][at_type][atA]))


        # Define the output (predicted total energy) as a sum of all atomic energies
        element_outputs = [element_processed[u][v] for u in element_processed.keys() for v in element_processed[u].keys()]
        total_outputs = element_outputs #+ rc_processed
        main_output = keras.layers.add(total_outputs)

        # Organize the input structure by using the structure as the network defined above.
        # because Keras will take multiple input one by one according to the sequence.
        inputs = []
        #for at_type in range(type_nb):
        for at_type in at_idx_map:
            dims = input_dim_list[at_type]

            #for atA in range(dims[0]):
            for atA in at_idx_map[at_type]:
                inputs.append(inp['element'][at_type][atA])
                #inputs.append(inp['rc'][at_type][atA])

        # Finalizing building the main network model.
        main_network = Model(inputs, main_output)

        return main_network


    # Glboal Var
    n_samples = len(md_train_arr)


    # Maintainence // Problem: In case we have no atom for that particular element, 
    # it is difficult to give the input dimension
        
    # Maintain: I assume that same amount of symmetry function is used for each atom  
    # Maintain: Need a input_dim_list_generator function in the future
    """
    def input_dim_list_generator(element):
        # In case of element with 0 atoms in the configuration
        nAtomsForEle = len(at_idx_map(element));
        if len nAtomsForEle == 0:
            return (nAtomsForEle, )
            """
    #nSymFunc = Gfunc_data['O'][0].shape[1]
    n_symm_func = Gfunc_data['C'][  at_idx_map['C'][0]].shape[1]
    #input_dim_list = [(len(at_idx_map[at]) ,nSymFunc, nb_feat) for at in at_idx_map.keys()]

    input_dim_list = {}
    for at_type in at_idx_map:
        input_dim_list[at_type] = (n_symm_func, 1)
        

    Mode='NoFirstDrop'
    #model = main_network(input_dim_list)
    model = main_network(input_dim_list, at_idx_map, 'NoFirstDrop')
    #model.summary()

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
    #!mkdir $model_folder
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
                            batch_size=b, epochs=100, shuffle=True,
                            verbose=0, validation_data=(inp_val, y_val)
                            )
    repeat(s, session)
                        
    # Load model weights
    model.load_weights(check1)

    # Error on TEST set 
    y_pred_scaled = model.predict(inp_test)
    y_pred = y_pred_scaled.T[0]  # in kcal/mol unit
    y_obs = y_test #/Eunit

    err_test = np.sqrt(mean_squared_error(y_pred, y_obs))
    errAbs_test = mean_absolute_error(y_pred, y_obs) 


    #print('The mean value of energies in test set: ', '%.4f' %E_ref_orig[test_idx].mean())
    """
    # Scatter plot of predicted and true values of energies in the test set
    x2 = pd.Series(y_pred, name="Predicted")
    x1 = pd.Series(y_obs, name="Observed")

    data1 = pd.concat([x1, x2], axis=1)
    g = sns.lmplot("Observed", "Predicted", data1,
            scatter_kws={"marker": ".", "color": "navy", "alpha": 0.4 },
            line_kws={"linewidth": 1, "color": "orange"},
            height=8, aspect=1);
    plt.plot(ls="--", c=".1")
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='purple')
    plt.grid(which='minor', linestyle='-', linewidth='0.5', color='blue', alpha=0.25)
    plt.minorticks_on()
    plt.title('Test Set Prediction', fontsize=20)
    plt.savefig('%s-%s-%s-%s.png' %(g,r,a,m), figsize=[4,4], dpi=300)
    """
    save_dir

    sys.stdout = open('s-%s-%s-%s-%s-%s-%s.txt' %(g,r,a,m,s,b), "w")

    print('RMSE_test:', '%.4f' % err_test)
    #print('MAE_test:','%.4f' % errAbs_test)
    print(Gfunc_data['S'][6].shape)

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
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
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

target = 0.5
p_count = 20 #population, set minimum of 10
i_length = 6 #number of genes
# set the range of the parameters to optimize
i_min = 3  
i_max = 10 
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]

for i in range(5):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

sys.stdout = open('evolucion.txt', "w")
print (fitness_history)
sys.stdout.close()

sys.stdout = open('valores.txt', "w")
print(p[0])
sys.stdout.close()
