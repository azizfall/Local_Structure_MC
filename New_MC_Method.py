import numpy as np
import math
import matplotlib.pyplot as plt 
from random import randrange 
import random
import csv
import sys
from numba import jit, types, typed
from numba.experimental import jitclass
from numba.typed import List
import numba as nb
import ctypes

prob_bb_base = 0.0
prob_bi_base = 0.0
prob_ii_base = 0.0



prob_bb_expansion = 0.0
prob_bi_expansion = 0.0
prob_ii_expansion = 0.0


#hash table hashes row_col node to corresponding
#index in its list --- useful for fast deletes
boundary_atoms_dict_specie1 = {}
boundary_atoms_list_specie1 = []

interior_atoms_dict_specie1 = {}
interior_atoms_list_specie1 = []

boundary_atoms_dict_specie2 = {}
boundary_atoms_list_specie2 = []

interior_atoms_dict_specie2 = {}
interior_atoms_list_specie2 = []


#2D ISING MODEL 
# H = -J Sum( Si Sj) 
spin_array = 0
spin_array_initial = 0


#type_ is either 0, 1, 2
#type_ = 0 all spins are down
#type_ = 1 all spins are up 
#type_ = 2 all initialized randomly 
def initialize_spin_lattice(num_row, num_col, type_):
	global spin_array
	spin_array = np.zeros((num_row, num_col))
	for i in range(num_row):
		for j in range(num_col):
			if type_ == 0: 
				spin_array[i][j] = -1 
			elif type_ == 1: 
				spin_array[i][j] = 1
			elif type_ == 2: 
				x = random.random() 
				if x < 0.5: 
					spin_array[i][j] = -1
				else:
					spin_array[i][j] = 1


@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
	""" returns a void pointer from a given memory address """
	from numba.core import types, cgutils
	sig = types.voidptr(src)

	def codegen(cgctx, builder, sig, args):
		return builder.inttoptr(args[0], cgutils.voidptr_t)
	return sig, codegen


@jit(nopython=True)
def is_boundary_atom(spin_tuple, row, col):
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	spin_type = spin_array[row][col]
	
	#an atom is a boundary if atleast one
	#of its a neighbors is of opposite spin 
	#type otherwise it is an interior atom 
	is_boundary = False

	#check rigt neighbor 
	if spin_array[row][(col + 1) % num_col] != spin_type:
		is_boundary = True
	#check bottom neighbor 
	elif spin_array[(row + 1)%num_row][col] != spin_type:
		is_boundary = True
	#check left neighbor 
	elif spin_array[row][(col - 1) % num_col] != spin_type:
		is_boundary = True 
	#check above neighbor 
	elif spin_array[(row - 1) % num_row][col] != spin_type:
		is_boundary = True

	return is_boundary


#returns a string that 
#is either bb, bi, or ii
#indicating the type of swap
#it is 
@jit(nopython=True)
def swap_type_atoms(spin_tuple, row1, col1, row2, col2):
	atom1 = ""
	atom2 = ""

	if is_boundary_atom(spin_tuple,row1, col1):
		atom1 = atom1 + "b"
	else:
		atom1 = atom1 + "i"

	if is_boundary_atom(spin_tuple, row2, col2):
		atom2 = atom2 + "b"
	else:
		atom2 = atom2 + "i" 

	if (atom1 == "b" and atom2 == "i") or (atom1 == "i" and atom2 == "b"):
		return "bi"
	elif (atom1 == "b" and atom2 == "b"):
		return "bb"

	return "ii"

#max_power is the (power + 1) of the maximum edge of the 2x2 lattice 
@jit(nopython=True)
def row_col_to_num(row, col, max_power):
	power = 0
	num = 0 
	i = 1
	j = 1
	while i <= max_power:
		x = col % 10 
		num = num + x*(10**power) 
		col = col//10 
		power = power + 1
		i = i + 1
  
	power = power + max_power

	while j <= max_power: 
		x = row % 10 
		num = num + x*(10**power)
		row = row//10 
		power = power + 1 
		j = j + 1
		
	return num 

#converts the number back into the (row,col)
@jit(nopython=True)
def num_to_row_col(num, max_power):
	row = 0 
	col = 0 
	power = 0
	for i in range(max_power):
		x = num % 10 
		col = col + x*(10**power)
		num = num//10 
		power = power + 1  

	power = 0 
	for i in range(max_power):
		num = num//10 

	for i in range(max_power):
		x = num % 10 
		row = row + x*(10**power)
		num = num//10 
		power = power + 1 

	return (row,col)

#computes the power of the maximum edge
@jit(nopython=True)
def compute_max_power(num_row, num_col):
	x = 1
	while(num_row//10 > 0):
		x = x + 1 
		num_row = num_row//10  

	y = 1 
	while(num_col//10 > 0):
		y = y + 1
		num_col = num_col//10 

	if x > y:
		return x 

	return y 

#initialize boundary/interior atoms list and dicts
def compute_all_atom_types():
	global spin_array

	num_row, num_col = np.shape(spin_array) 
	max_power = compute_max_power(num_row, num_col)

	addr = spin_array.ctypes.data
	spin_tuple = (addr, spin_array.shape, spin_array.dtype, max_power)
	
	for row in range(num_row):
		for col in range(num_col):
			
			row_col = row_col_to_num(row, col, max_power)
			#if (row,col) is specie1
			if spin_array[row][col] == -1:
				#atom is a boundary atom
				if is_boundary_atom(spin_tuple, row, col):
					index = len(boundary_atoms_list_specie1)
					boundary_atoms_list_specie1.append(row_col)
					boundary_atoms_dict_specie1[row_col] = index 
				else:
					index = len(interior_atoms_list_specie1)
					interior_atoms_list_specie1.append(row_col)
					interior_atoms_dict_specie1[row_col] = index 
			#atom is specie 2
			else:
				if is_boundary_atom(spin_tuple, row, col):
					index = len(boundary_atoms_list_specie2)
					boundary_atoms_list_specie2.append(row_col)
					boundary_atoms_dict_specie2[row_col] = index 
				else:
					index = len(interior_atoms_list_specie2)
					interior_atoms_list_specie2.append(row_col)
					interior_atoms_dict_specie2[row_col] = index 



#computes the probability for the different
#possible swap events 
def compute_probabilities_swap_events(bJ,numb_specie1,numb_specie2,numi_specie1,numi_specie2):
	
	global prob_bb_base
	global prob_bi_base
	global prob_ii_base

	num_bb_pair = numb_specie1*numb_specie2 
	num_bi_pair = numb_specie1*numi_specie2 + numi_specie1*numb_specie2
	num_ii_pair = numi_specie1*numi_specie2

	norm = num_bi_pair*prob_bi_base + num_ii_pair*prob_ii_base + num_bb_pair*prob_bb_base
	prob_bb = (num_bb_pair*prob_bb_base)/norm
	prob_bi = (num_bi_pair*prob_bi_base)/norm
	prob_ii = (num_ii_pair*prob_ii_base)/norm

	return [prob_bb, prob_bi, prob_ii]


#add neighbors in clockwise direction
@jit(nopython=True)
def generate_neighbors(spin_tuple, row, col, spin_type, visited_list):
	nei_list = []

	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	max_power = spin_tuple[3]

	#right neighbor
	if spin_array[row][(col + 1)%num_col] == spin_type:
		row_col = row_col_to_num(row, (col + 1)%num_col, max_power)
		if visited_list.get(row_col, 0) == 0: #node has not been visited
			nei_list.append(row_col)

	#bottom neighor
	if spin_array[(row + 1)%num_row][col] == spin_type:
		row_col = row_col_to_num((row + 1)%num_row, col, max_power)
		if visited_list.get(row_col, 0) == 0: #node has not been visited
			nei_list.append(row_col)

	#left neighbor
	if spin_array[row][(col - 1)%num_col] == spin_type:
		row_col = row_col_to_num(row,(col - 1)%num_col, max_power)
		if visited_list.get(row_col, 0) == 0: #node has not been visited 
			nei_list.append(row_col)

	#above neighbor 
	if spin_array[(row - 1)%num_row][col] == spin_type:
		row_col = row_col_to_num((row - 1)%num_row, col, max_power)
		if visited_list.get(row_col, 0) == 0: #node has not been visited
			nei_list.append(row_col) 

	return nei_list


@jit(nopython=True)
def fact(n):
	res = 1
	for i in range(2, n + 1):
		res = res*i 
	return res

@jit(nopython=True)
def permute(x):
	perm = np.random.permutation(len(x))
	ret = []
	for i in range(len(perm)):
		ret.append(x[perm[i]])
	return ret


@jit(nopython=True)
def compute_prob_of_config(spin_tuple, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
 prob_bb_expansion, prob_bi_expansion, prob_ii_expansion,
 changed_atoms_specie, row1, col1, row2, col2, compute_only_partial=True):
	
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	max_power = spin_tuple[3]

	num_atom = len(changed_atoms_specie)
	
	Atom_set_specie1 = []
	Atom_set_specie2 = []

	if compute_only_partial == True:

		if spin_array[row1][col1] == -1:
			Atom_set_specie1.append([row1, col1])
		else:
			Atom_set_specie2.append([row1, col1])


		if spin_array[row2][col2] == -1:
			Atom_set_specie1.append([row2, col2])
		else:
			Atom_set_specie2.append([row2, col2])

	else:
		for i in range(num_atom):
			row, col = changed_atoms_specie[i]
			
			if spin_array[row][col] == -1:
				Atom_set_specie1.append(changed_atoms_specie[i])

			elif spin_array[row][col] == 1:
				Atom_set_specie2.append(changed_atoms_specie[i])

		
	assert len(Atom_set_specie1) == len(Atom_set_specie2), "Sets of Specie NEQ"

	num_pairs = len(Atom_set_specie1)

	#hash of all atoms that changed
	Total_Atoms_hash = {}
	for i in range(num_atom):
		row, col = changed_atoms_specie[i]
		row_col = row_col_to_num(row, col, max_power)
		Total_Atoms_hash[row_col] = 1

	#total probability
	total_prob = 0.0

	for m in range(num_pairs):
		row1_start, col1_start = Atom_set_specie1[m]
		for n in range(num_pairs):
			row2_start, col2_start = Atom_set_specie2[n]
			swap_type = swap_type_atoms(spin_tuple, row1_start, col1_start, row2_start, col2_start)
			prob_partial = 1.0

			prob_expansion = 0.0
			
			if swap_type == "bb":
				prob_partial = prob_partial*prob_bb*1.0/(num_bb)
				prob_expansion = prob_bb_expansion

			elif swap_type == "bi":
				prob_partial = prob_partial*prob_bi*1.0/(num_bi)
				prob_expansion = prob_bi_expansion

			elif swap_type == "ii":
				prob_partial = prob_partial*prob_ii*1.0/(num_ii)
				prob_expansion = prob_ii_expansion
	

			spin_type1 = spin_array[row1_start][col1_start]
			spin_type2 = spin_array[row2_start][col2_start] 

			assert spin_type1 != spin_type2, "Error Spin types are the same"

			row_col_atom1 = row_col_to_num(row1_start, col1_start, max_power)
			row_col_atom2 = row_col_to_num(row2_start, col2_start, max_power)

			visited_list = {}

			queue1 = [] 
			queue2 = []
			queue1.append(row_col_atom1)
			queue2.append(row_col_atom2)

			visited_list[row_col_atom1] = 1
			visited_list[row_col_atom2] = 1


			num_added = 2
			while (len(queue1) != 0):
				
				
				nei_list1 = []
				nei_list2 = []

				for i in range(len(queue1)):
					row, col = num_to_row_col(queue1[i], max_power)
					spin_type = spin_array[row][col]
					neibs = generate_neighbors(spin_tuple, row, col, spin_type, visited_list)
					for j in range(len(neibs)):
						nei_list1.append(neibs[j])
						visited_list[neibs[j]] = 1

				for i in range(len(queue2)):
					row, col = num_to_row_col(queue2[i], max_power)
					spin_type = spin_array[row][col]
					neibs = generate_neighbors(spin_tuple, row, col, spin_type, visited_list)
					for j in range(len(neibs)):
						nei_list2.append(neibs[j])
						visited_list[neibs[j]] = 1

				
				#Empty queues 
				queue1 = []
				queue2 = []

				if (len(nei_list1) == 0) or (len(nei_list2) == 0):
					continue 

				num_swap_list1 = 0 
				num_swap_list2 = 0

				atoms_to_add_list1 = []
				atoms_to_add_list2 = []
				
				for i in range(len(nei_list1)):
					if (Total_Atoms_hash.get(nei_list1[i], 0) == 1):
						num_swap_list1 = num_swap_list1 + 1 
						atoms_to_add_list1.append(nei_list1[i])


				for i in range(len(nei_list2)):
					if (Total_Atoms_hash.get(nei_list2[i], 0) == 1):
						num_swap_list2 = num_swap_list2 + 1 
						atoms_to_add_list2.append(nei_list2[i])

				if num_swap_list1 != num_swap_list2:
					prob_partial = 0.0
					break

				min_size = min(len(nei_list1), len(nei_list2))

				#lets call s the number of swaps needed
				s = num_swap_list1 
				prob_s = (fact(min_size))/(fact(s)*fact(min_size - s))
				prob_s = prob_s*pow(prob_expansion, s)*pow((1.0 - prob_expansion), (min_size - s)) 

				#find the probability of picking the correct atoms in nei_lists
				N = len(nei_list1)
				M = len(nei_list2) 

				prob_N = fact(s)*fact(N - s)/(fact(N))
				prob_M = fact(s)*fact(M - s)/(fact(M))

				prob_partial = prob_partial*prob_s*prob_N*prob_M

				for i in range(s):
					queue1.append(atoms_to_add_list1[i])
					queue2.append(atoms_to_add_list2[i])

					row_nei1, col_nei1 = num_to_row_col(atoms_to_add_list1[i], max_power)
					row_nei2, col_nei2 = num_to_row_col(atoms_to_add_list2[i], max_power)

					spin_type1 = spin_array[row_nei1][col_nei1]
					spin_type2 = spin_array[row_nei2][col_nei2]

					assert spin_type1 != spin_type2, "Error Spin types are the same"


				num_added = num_added + 2*s

			if prob_partial > 0:
				if num_added != len(changed_atoms_specie):
					prob_partial = 0.0

			total_prob = total_prob + prob_partial

	return total_prob



#returns set of atoms whose specie needs to change
@jit(nopython=True)
def Expand_Swap(spin_tuple, row_atom1, col_atom1, row_atom2, col_atom2, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
	prob_bb_expansion, prob_bi_expansion, prob_ii_expansion):

	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	max_power = spin_tuple[3]

	changed_atoms_specie = List()
	
	changed_atoms_specie.append([row_atom1, col_atom1])
	changed_atoms_specie.append([row_atom2, col_atom2])


	spin_type1 = spin_array[row_atom1][col_atom1]
	spin_type2 = spin_array[row_atom2][col_atom2] 

	assert spin_type1 != spin_type2, "Error Spin types are the same"


	row_col_atom1 = row_col_to_num(row_atom1, col_atom1, max_power)
	row_col_atom2 = row_col_to_num(row_atom2, col_atom2, max_power)

	#Determine which expansion probability to use according to 
	#swap type of the initial pair of atoms 
	prob_expansion = 0.0

	swap_type = swap_type_atoms(spin_tuple, row_atom1, col_atom1, row_atom2, col_atom2)

	if swap_type == "bb":
		prob_expansion = prob_bb_expansion 
	elif swap_type == "bi":
		prob_expansion = prob_bi_expansion
	elif swap_type == "ii":
		prob_expansion = prob_ii_expansion


	visited_list = {}


	queue1 = [] 
	queue2 = []
	queue1.append(row_col_atom1)
	queue2.append(row_col_atom2)

	visited_list[row_col_atom1] = 1
	visited_list[row_col_atom2] = 1

	while (len(queue1) != 0):
		
		nei_list1 = []
		nei_list2 = []

		for i in range(len(queue1)):
			row, col = num_to_row_col(queue1[i], max_power)
			spin_type = spin_array[row][col]
			neibs = generate_neighbors(spin_tuple, row, col, spin_type, visited_list)
			for j in range(len(neibs)):
				nei_list1.append(neibs[j])
				visited_list[neibs[j]] = 1

		for i in range(len(queue2)):
			row, col = num_to_row_col(queue2[i], max_power)
			spin_type = spin_array[row][col]
			neibs = generate_neighbors(spin_tuple, row, col, spin_type, visited_list)
			for j in range(len(neibs)):
				nei_list2.append(neibs[j])
				visited_list[neibs[j]] = 1

		#Empty queues 
		queue1 = []
		queue2 = []

		if (len(nei_list1) == 0) or (len(nei_list2) == 0):
			continue

	
		nei_list1 = permute(nei_list1)
		nei_list2 = permute(nei_list2)

	
		num_to_swap = 0 
		min_size = min(len(nei_list1), len(nei_list2)) 

		for i in range(min_size):
			r = random.random() 
			if r < prob_expansion:
				num_to_swap = num_to_swap + 1


		#shuffle both neighbor lists and select the sets of atoms to swap 
		if num_to_swap > 0: 

			for j in range(num_to_swap):
				row_nei1, col_nei1 = num_to_row_col(nei_list1[j], max_power)
				row_nei2, col_nei2 = num_to_row_col(nei_list2[j], max_power)

				spin_type1 = spin_array[row_nei1][col_nei1]
				spin_type2 = spin_array[row_nei2][col_nei2]

				assert spin_type1 != spin_type2, "Error Spin types are the same"

				changed_atoms_specie.append([row_nei1, col_nei1])
				changed_atoms_specie.append([row_nei2, col_nei2])

				queue1.append(nei_list1[j])
				queue2.append(nei_list2[j])


	total_forward_prob = compute_prob_of_config(spin_tuple, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii, prob_bb_expansion, prob_bi_expansion, prob_ii_expansion, \
		changed_atoms_specie, row_atom1, col_atom1, row_atom2, col_atom2)

	return changed_atoms_specie, total_forward_prob

#All atoms need to be flipped before calling this function
def compute_reverse_prob(spin_tuple, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
 prob_bb_expansion, prob_bi_expansion, prob_ii_expansion, \
 changed_atoms_specie, row1, col1, row2, col2):
	return compute_prob_of_config(spin_tuple, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
	prob_bb_expansion, prob_bi_expansion, prob_ii_expansion, \
	changed_atoms_specie, row1, col1, row2, col2)

	
#copy arr1 into arr2
def copy_arr(arr1 , arr2):
	num_row, num_col = np.shape(arr1) 
	for i in range(num_row):
		for j in range(num_col):
			arr2[i][j] = arr1[i][j] 


def arr_to_str(arr):
	if len(arr) == 0:
		return "[]"
	str_out = "["
	for i in range(len(arr)):
		str_out = str_out + str(arr[i])
		if i == (len(arr) - 1): 
			str_out = str_out + "]"
		else:
			str_out = str_out + ","

	return str_out


@jit(nopython=True)
def Energy_of_spin_nei(spin_tuple, row,col,J):
	#global spin_array
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape

	spin_type = spin_array[row][col] 
	E = 0.0
	#neighbor above
	if row == 0:
		E = E - J*spin_array[num_row - 1][col]*spin_type	 
	else:
		E = E - J*spin_array[row - 1][col]*spin_type
		
	#neighbor below
	if row == num_row - 1: 
		E = E - J*spin_array[0][col]*spin_type	
	else: 
		E = E - J*spin_array[row + 1][col]*spin_type
		
	#neighbor right 
	if col == num_col - 1: 
		E = E - J*spin_array[row][0]*spin_type	
	else: 
		E = E - J*spin_array[row][col + 1]*spin_type
		
	#neighbor left 
	if col == 0: 
		E = E - J*spin_array[row][num_col - 1]*spin_type
	else: 
		E = E - J*spin_array[row][col - 1]*spin_type
		
	return E 

@jit(nopython=True)
def Calculate_E(spin_tuple, J):
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	E = 0.0 
	for i in range(num_row): 
		for j in range(num_col):
			E = E + Energy_of_spin_nei(spin_tuple, i,j,J)
	#multiplied by 0.5 due to double counting
	return 0.5*E 

#calculates delta_E based on 
#row and col of spin flip
#and the old Energy E_old 
@jit(nopython=True)
def delta_E_func(spin_tuple, E_old, J, changed_atoms_specie):
	
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)

	num_atoms = len(changed_atoms_specie)
	
	E_new = E_old

	for i in range(num_atoms):
		row, col = changed_atoms_specie[i]
		E_nei = Energy_of_spin_nei(spin_tuple, row,col,J)
		#change species
		spin_array[row][col] = -1*spin_array[row][col]
		delta_E = Energy_of_spin_nei(spin_tuple, row,col,J) - E_nei
		E_new = E_new + delta_E

	#Undo All changes 
	for i in range(num_atoms):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]

	return (E_new - E_old, E_new) 

spec = [
	('row', types.int64),
	('col', types.int64),
	('specie_before', types.int64),
	('specie_after', types.int64),
	('atom_type_before', types.string),
	('atom_type_after', types.string)
]
@jitclass(spec)
class changes_in_atom:
	def __init__(self):
		#the row_col node
		self.row = 0
		self.col = 0
		self.specie_before = 0
		self.specie_after = 0
		self.atom_type_before = ""
		self.atom_type_after = ""

	# takes in list as [row,col]
	def set_row_col(self, row_col_arr):
		self.row = row_col_arr[0]
		self.col = row_col_arr[1]

	#checks the current type of this 
	#atom and sets the before variables 
	def compute_types_before(self, spin_tuple):
		addr = spin_tuple[0]
		shape = spin_tuple[1]
		dtype = spin_tuple[2]
		spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)

		if spin_array[self.row][self.col] == -1:
			self.specie_before = 1 
		else:
			self.specie_before = 2

		if is_boundary_atom(spin_tuple, self.row, self.col):
			self.atom_type_before = "b"
		else:
			self.atom_type_before = "i"

	#checks the current type of this
	#atom and sets the after the variables 
	def compute_types_after(self, spin_tuple):
		addr = spin_tuple[0]
		shape = spin_tuple[1]
		dtype = spin_tuple[2]
		spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)

		if spin_array[self.row][self.col] == -1:
			self.specie_after = 1 
		else:
			self.specie_after = 2

		if is_boundary_atom(spin_tuple, self.row, self.col):
			self.atom_type_after = "b"
		else:
			self.atom_type_after = "i"

#returns list of neighbors
@jit(nopython=True)
def compute_neighbors(row, col, num_row, num_col): 
	neighbors = []
	#right neighbor 
	nei_right = [row, (col + 1)%num_col]
	nei_bottom = [(row + 1)%num_row , col]
	nei_left = [row, (col - 1)%num_col]
	nei_above = [ (row - 1)%num_row, col]
	neighbors.append(nei_right)
	neighbors.append(nei_bottom)
	neighbors.append(nei_left)
	neighbors.append(nei_above)
	return neighbors



#computes the changes in atomic type (boundary or interior)
#and specie type (specie1 or specie2) after a single swap
#the function returns the array changes_in_atom_arr which is an 
#array where the changes in atomic type
#and specie is stored for each atom that undergoes a change
@jit(nopython=True)
def compute_atom_redist(spin_tuple, row1, col1, row2, col2, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
	prob_bb_expansion, prob_bi_expansion, prob_ii_expansion):
	
	addr = spin_tuple[0]
	shape = spin_tuple[1]
	dtype = spin_tuple[2]
	spin_array = nb.carray(address_as_void_pointer(addr), shape, dtype=dtype)
	num_row, num_col = shape
	max_power = spin_tuple[3]
	
	#changed_atoms_specie are the list of atoms that need to undergo
	#a change in specie
	changed_atoms_specie, total_forward_prob = Expand_Swap(spin_tuple, row1, col1, row2, col2, prob_bb, prob_bi, prob_ii, num_bb, num_bi, num_ii,\
		prob_bb_expansion, prob_bi_expansion, prob_ii_expansion)

	atoms_to_check = {}

	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		row_col = row_col_to_num(row, col, max_power)
		atoms_to_check[row_col] = 1 
		neighbors = compute_neighbors(row, col, num_row, num_col)
		for j in range(len(neighbors)):
			row_nei, col_nei = neighbors[j]
			row_col_nei = row_col_to_num(row_nei, col_nei, max_power)
			if atoms_to_check.get(row_col_nei, 0) == 0:
				atoms_to_check[row_col_nei] = 1

	changes_in_atom_arr = [] 

	#compute the atomic types before 
	for row_col in atoms_to_check.keys():
		row, col = num_to_row_col(row_col, max_power)
		row_col_arr = [row,col]
		change = changes_in_atom()
		change.set_row_col(row_col_arr)
		change.compute_types_before(spin_tuple) 
		changes_in_atom_arr.append(change) 

	#make All the swaps
	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]

	#compute changes after 
	for i in range(len(changes_in_atom_arr)):
		changes_in_atom_arr[i].compute_types_after(spin_tuple) 

	#undo All the swaps
	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]

	return changes_in_atom_arr, total_forward_prob, changed_atoms_specie


#compute the new lengths of 
#boundary_atoms_list_specie1, interior_atoms_list_specie1
#boundary_atoms_list_specie2, interior_atoms_list_specie2
#after a single swap, w/o modifying the arrays
@jit(nopython=True)
def compute_new_lengths(changes_in_atom_arr, numb_specie1, numb_specie2,\
 numi_specie1, numi_specie2):

	
	for i in range(len(changes_in_atom_arr)):

		
		specie_before_i = changes_in_atom_arr[i].specie_before 
		specie_after_i = changes_in_atom_arr[i].specie_after

		atom_type_before_i = changes_in_atom_arr[i].atom_type_before
		atom_type_after_i = changes_in_atom_arr[i].atom_type_after


		#if the species are the same this must be one
		#of the neighbor atoms
		if specie_before_i == specie_after_i:
			
			if specie_before_i == 1:
				
				if atom_type_before_i != atom_type_after_i:
					
					if atom_type_before_i == "i":
						numi_specie1 = numi_specie1 - 1 
						numb_specie1 = numb_specie1 + 1
					else:
						numi_specie1 = numi_specie1 + 1 
						numb_specie1 = numb_specie1 - 1
			#species is 2 if not 1
			else:

				if atom_type_before_i != atom_type_after_i:

					if atom_type_before_i == "i":
						numi_specie2 = numi_specie2 - 1 
						numb_specie2 = numb_specie2 + 1
					else:
						numi_specie2 = numi_specie2 + 1
						numb_specie2 = numb_specie2 - 1 

		elif specie_before_i != specie_after_i:

			#if specie_before_i = 1 then  specie_after_i = 2
			if specie_before_i == 1:

				if atom_type_before_i == atom_type_after_i:

					if atom_type_before_i == "i":
						numi_specie1 = numi_specie1 - 1
						numi_specie2 = numi_specie2 + 1
					else:
						numb_specie1 = numb_specie1 - 1
						numb_specie2 = numb_specie2 + 1

				elif atom_type_before_i != atom_type_after_i:

					if atom_type_before_i == "i":
						numi_specie1 = numi_specie1 - 1
						numb_specie2 = numb_specie2 + 1
					else:
						numb_specie1 = numb_specie1 - 1
						numi_specie2 = numi_specie2 + 1 

			#specie_before_i = 2
			else:

				if atom_type_before_i == atom_type_after_i:

					if atom_type_before_i == "i":
						numi_specie1 = numi_specie1 + 1
						numi_specie2 = numi_specie2 - 1
					else:
						numb_specie1 = numb_specie1 + 1
						numb_specie2 = numb_specie2 - 1

				elif atom_type_before_i != atom_type_after_i:

					if atom_type_before_i == "i":
						numb_specie1 = numb_specie1 + 1
						numi_specie2 = numi_specie2 - 1
					else:
						numi_specie1 = numi_specie1 + 1
						numb_specie2 = numb_specie2 - 1 

	return (numb_specie1,numb_specie2,numi_specie1,numi_specie2)


def compute_cdf(prob_arr):
	sum_ = 0.0 
	cdf = []
	for i in range(len(prob_arr)):
		sum_ = sum_ + prob_arr[i]
		cdf.append(sum_)

	return cdf 


#self.specie_before = 0
#self.specie_after = 0
#self.atom_type_before = ""
#self.atom_type_after = ""


#boundary_atoms_list_specie1
#interior_atoms_list_specie1
#boundary_atoms_list_specie2
#interior_atoms_list_specie2


#returns the string of the list an atom is associated with before 
#and after the swap, takes the class changes_in_atom as input
def lists_b4_and_after(change_atom):
	
	list_before = ""
	list_after = ""

	if change_atom.specie_before == 1:
		if change_atom.atom_type_before == "i":
			list_before = "interior_atoms_list_specie1"
		else: 
			list_before = "boundary_atoms_list_specie1"

	elif change_atom.specie_before == 2:
		if change_atom.atom_type_before == "i":
			list_before = "interior_atoms_list_specie2"
		else:
			list_before = "boundary_atoms_list_specie2"


	if change_atom.specie_after == 1:
		if change_atom.atom_type_after == "i":
			list_after = "interior_atoms_list_specie1"
		else: 
			list_after = "boundary_atoms_list_specie1"

	elif change_atom.specie_after == 2:
		if change_atom.atom_type_after == "i":
			list_after = "interior_atoms_list_specie2"
		else:
			list_after = "boundary_atoms_list_specie2"


	return list_before, list_after


#deletes row_col from the list list_str
def delete_atom(row_col, list_str):
	if list_str == "interior_atoms_list_specie1":
		#index in the list to delete
		index = interior_atoms_dict_specie1[row_col]
		index_last = len(interior_atoms_list_specie1) - 1
		#the row_col_node associated w/ last index
		row_col_last = interior_atoms_list_specie1[index_last]
		interior_atoms_list_specie1[index_last] = interior_atoms_list_specie1[index]
		interior_atoms_list_specie1[index] = row_col_last 

		#change index of row_col_last in dictionary 
		interior_atoms_dict_specie1[row_col_last] = index

		#remove the elements 
		interior_atoms_list_specie1.pop()
		del interior_atoms_dict_specie1[row_col]

	elif list_str == "boundary_atoms_list_specie1":
		#index in the list to delete 
		index = boundary_atoms_dict_specie1[row_col]
		index_last = len(boundary_atoms_list_specie1) - 1
		#the row_col node associated w/ last index 
		row_col_last = boundary_atoms_list_specie1[index_last]
		boundary_atoms_list_specie1[index_last] = boundary_atoms_list_specie1[index]
		boundary_atoms_list_specie1[index] = row_col_last

		#change index of row_col_last in dictionary 
		boundary_atoms_dict_specie1[row_col_last] = index

		#remove elements 
		boundary_atoms_list_specie1.pop() 
		del boundary_atoms_dict_specie1[row_col]

	elif list_str == "interior_atoms_list_specie2":
		#index in the list to delete 
		index = interior_atoms_dict_specie2[row_col]
		index_last = len(interior_atoms_list_specie2) - 1
		#the row_col node associated w/ last index 
		row_col_last = interior_atoms_list_specie2[index_last]
		interior_atoms_list_specie2[index_last] = interior_atoms_list_specie2[index]
		interior_atoms_list_specie2[index] = row_col_last 

		#change index of row_col_last in dictionary 
		interior_atoms_dict_specie2[row_col_last] = index

		#remove elements 
		interior_atoms_list_specie2.pop()
		del interior_atoms_dict_specie2[row_col]

	elif list_str == "boundary_atoms_list_specie2":
		#index in the list to delete 
		index = boundary_atoms_dict_specie2[row_col]
		index_last = len(boundary_atoms_list_specie2) - 1
		#the row_col node associated w/ last index 
		row_col_last = boundary_atoms_list_specie2[index_last]
		boundary_atoms_list_specie2[index_last] = boundary_atoms_list_specie2[index]
		boundary_atoms_list_specie2[index] = row_col_last 

		#change index of row_col_last in dictionary 
		boundary_atoms_dict_specie2[row_col_last] = index 

		#remove elements 
		boundary_atoms_list_specie2.pop() 
		del boundary_atoms_dict_specie2[row_col] 


#inserts row_col into list_str
def insert_atom(row_col, list_str):
	if list_str == "interior_atoms_list_specie1":
		index = len(interior_atoms_list_specie1)
		interior_atoms_list_specie1.append(row_col)
		interior_atoms_dict_specie1[row_col] = index 

	elif list_str == "boundary_atoms_list_specie1":
		index = len(boundary_atoms_list_specie1)
		boundary_atoms_list_specie1.append(row_col)
		boundary_atoms_dict_specie1[row_col] = index 

	elif list_str == "interior_atoms_list_specie2":
		index = len(interior_atoms_list_specie2)
		interior_atoms_list_specie2.append(row_col)
		interior_atoms_dict_specie2[row_col] = index 

	elif list_str == "boundary_atoms_list_specie2":
		index = len(boundary_atoms_list_specie2)
		boundary_atoms_list_specie2.append(row_col)
		boundary_atoms_dict_specie2[row_col] = index



def accept_move(changes_in_atom_arr, changed_atoms_specie):
	global spin_array
	num_row, num_col = np.shape(spin_array) 
	max_power = compute_max_power(num_row, num_col)

	#Make the swaps
	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]
	

	for i in range(len(changes_in_atom_arr)):
		list_before, list_after = lists_b4_and_after(changes_in_atom_arr[i])
		
		if list_before == list_after:
			continue

		curr_row = changes_in_atom_arr[i].row 
		curr_col = changes_in_atom_arr[i].col 
		row_col = row_col_to_num(curr_row, curr_col, max_power)

		delete_atom(row_col, list_before)
		insert_atom(row_col, list_after)



#makes a move and returns whether 
#the move was accepted or rejected along
#with the new Energy 
def propose_move(E_old, J, kT, max_power):
	
	bJ = J/(kT)

	global boundary_atoms_list_specie1
	global interior_atoms_list_specie1
	global boundary_atoms_list_specie2
	global interior_atoms_list_specie2

	global prob_bb_expansion
	global prob_bi_expansion
	global prob_ii_expansion

	global spin_array
	num_row, num_col = np.shape(spin_array) 

	numb_specie1 = len(boundary_atoms_list_specie1)
	numb_specie2 = len(boundary_atoms_list_specie2)
	numi_specie1 = len(interior_atoms_list_specie1)
	numi_specie2 = len(interior_atoms_list_specie2)

	num_bb_pair = numb_specie1*numb_specie2 
	num_bi_pair = numb_specie1*numi_specie2 + numi_specie1*numb_specie2
	num_ii_pair = numi_specie1*numi_specie2

	#make a proposed swap according to the forward probability
	prob_bb, prob_bi, prob_ii\
	 = compute_probabilities_swap_events(bJ,numb_specie1,numb_specie2,numi_specie1,numi_specie2)

	#compute cdf from probabilities 
	prob_arr = [prob_bb, prob_bi, prob_ii]
	cdf = compute_cdf(prob_arr)
	#0 --> for bb swaps
	#1 --> for bi swaps
	#2 --> for ii swaps
	sample = [] 
	number_of_samples_to_draw = 1
	for s in range(number_of_samples_to_draw):
		r = random.random() 
		if r <= cdf[0]:
			sample.append(0)
			 
		else:
			for i in range(1,len(prob_arr)):
				if r > cdf[i - 1] and r <= cdf[i]:
					sample.append(i)  


	row1 = 0 
	col1 = 0 
	row2 = 0 
	col2 = 0
	   
	draw = sample[randrange(len(sample))] 
	#propose a bb swap
	if draw == 0:
		#find row col from boundary list
		row_col_b_specie1 = boundary_atoms_list_specie1[randrange(numb_specie1)]
		row1, col1 = num_to_row_col(row_col_b_specie1, max_power)

		#find second row col from boundary list
		row_col_b_specie2 = boundary_atoms_list_specie2[randrange(numb_specie2)]
		row2, col2 = num_to_row_col(row_col_b_specie2, max_power)


	#propose a bi swap
	elif draw == 1:
		#in order to have a symmetric probability 
		#between boundary and interior atoms of 
		#different specie type we need to create a pmf 


		pmf = [(numb_specie1*numi_specie2)/num_bi_pair, (numi_specie1*numb_specie2)/num_bi_pair]

		cdf = compute_cdf(pmf)
		#0 --> sample atoms from list boundary_atoms_list_specie1, interior_atoms_list_specie2 
		#1 --> sample atoms from list interior_atoms_list_specie1, boundary_atoms_list_specie2
		sample_bi = [] 
		number_of_samples_to_draw = 1
		
		for s in range(number_of_samples_to_draw):
			r = random.random() 
			if r <= cdf[0]:
				sample_bi.append(0)
			 
			else:
				for i in range(1,len(pmf)):
					if r > cdf[i - 1] and r <= cdf[i]:
						sample_bi.append(i)

		draw_bi = sample_bi[randrange(len(sample_bi))]   

		if draw_bi == 0:
			row_col1 = boundary_atoms_list_specie1[randrange(numb_specie1)]
			row1, col1 = num_to_row_col(row_col1 , max_power)

			row_col2 = interior_atoms_list_specie2[randrange(numi_specie2)]
			row2, col2 = num_to_row_col(row_col2 , max_power)

		elif draw_bi == 1:
			row_col1 = interior_atoms_list_specie1[randrange(numi_specie1)]
			row1, col1 = num_to_row_col(row_col1 , max_power)

			row_col2 = boundary_atoms_list_specie2[randrange(numb_specie2)]
			row2, col2 = num_to_row_col(row_col2 , max_power)


	#propose an ii swap
	else:
		row_col1 = interior_atoms_list_specie1[randrange(numi_specie1)]
		row1, col1 = num_to_row_col(row_col1 , max_power)

		row_col2 = interior_atoms_list_specie2[randrange(numi_specie2)]
		row2, col2 = num_to_row_col(row_col2 , max_power)

		 

	
	addr = spin_array.ctypes.data
	spin_tuple = (addr, spin_array.shape, spin_array.dtype, max_power)

	#Adds Neighboring Atoms Using Expansion Probabilities 
	#Also outputs the forward probability for the set of atoms undergoing
	#swaps
	changes_in_atom_arr, forward_probability, changed_atoms_specie = compute_atom_redist(spin_tuple, row1, col1,\
		row2, col2, prob_bb, prob_bi, prob_ii, num_bb_pair, num_bi_pair, num_ii_pair,\
		prob_bb_expansion, prob_bi_expansion, prob_ii_expansion)


	numb_specie1_after ,numb_specie2_after,\
	numi_specie1_after ,numi_specie2_after = compute_new_lengths(changes_in_atom_arr, numb_specie1, numb_specie2,\
 numi_specie1, numi_specie2)

	num_bb_pair_after = numb_specie1_after*numb_specie2_after
	num_bi_pair_after = numb_specie1_after*numi_specie2_after + numi_specie1_after*numb_specie2_after
	num_ii_pair_after = numi_specie1_after*numi_specie2_after

	prob_bb_after, prob_bi_after, prob_ii_after\
	 = compute_probabilities_swap_events(bJ,numb_specie1_after,numb_specie2_after,\
		numi_specie1_after,numi_specie2_after)



	#Make Swaps to compute the Reverse Probability
	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]


	reverse_probability = compute_reverse_prob(spin_tuple, prob_bb_after, prob_bi_after, prob_ii_after, \
		num_bb_pair_after, num_bi_pair_after, num_ii_pair_after,\
		prob_bb_expansion, prob_bi_expansion, prob_ii_expansion, \
		changed_atoms_specie, row1, col1, row2, col2)

	#Undo Swaps
	for i in range(len(changed_atoms_specie)):
		row, col = changed_atoms_specie[i]
		spin_array[row][col] = -1*spin_array[row][col]	


	delta_E, E_new = delta_E_func(spin_tuple, E_old, J, changed_atoms_specie)

	acceptance_prob = min(1, np.exp(-delta_E/kT)*(reverse_probability/forward_probability))

	#This is the sate of the proposed
	#move which can either be rejected or 
	#accepted 
	proposed_move_state = False

	if acceptance_prob == 1:
		proposed_move_state = True 
		accept_move(changes_in_atom_arr, changed_atoms_specie)
	else:
		r = random.random() 
		if r < acceptance_prob:
			proposed_move_state = True 
			accept_move(changes_in_atom_arr, changed_atoms_specie)


	return proposed_move_state, E_new





# Temperature or kb*T can be given relative to J 
def Equilibrate_T(kT, num_row, num_col, n_iter, J):
	global spin_array
	#calculate energy 
	max_power = compute_max_power(num_row, num_col)

	addr = spin_array.ctypes.data
	spin_tuple = (addr, spin_array.shape, spin_array.dtype, max_power)

	E_total = Calculate_E(spin_tuple, J) 
	#array of energy
	E_arr = []
	num_accepted_move = 0

	#number of accepted moves where the energy changes
	num_accepted_move_nonzero_energy = 0

	for k in range(n_iter):
		proposed_move_state, E_new = propose_move(E_total, J, kT, max_power)
		if proposed_move_state == True:
			if (E_total - E_new) != 0.0:
				num_accepted_move_nonzero_energy = num_accepted_move_nonzero_energy + 1
			E_total = E_new
			num_accepted_move = num_accepted_move + 1
		E_arr.append(E_total)
		
	return E_arr, E_total, num_accepted_move/n_iter, num_accepted_move_nonzero_energy/n_iter
		
		

def main():
	global spin_array

	global boundary_atoms_list_specie1
	global interior_atoms_list_specie1
	global boundary_atoms_list_specie2
	global interior_atoms_list_specie2

	global boundary_atoms_dict_specie1
	global interior_atoms_dict_specie1
	global boundary_atoms_dict_specie2
	global interior_atoms_dict_specie2

	global prob_bb_base
	global prob_bi_base
	global prob_ii_base

	global prob_bb_expansion
	global prob_bi_expansion
	global prob_ii_expansion



	num_row = 40
	num_col = 40
	

	bJ_arr = [float(sys.argv[1])]

	if bJ_arr[0] == 0.4406878:
		
		prob_bb_base = 0.8642656663978563
		prob_bi_base = 0.13378197271944042
		prob_ii_base = 0.0019528246093657442

	elif bJ_arr[0] == 0.1:

		prob_bb_base = 0.5544459681235077
		prob_bi_base = 0.3114567270716232
		prob_ii_base = 0.13409730409587728

	elif bJ_arr[0] == 1.5:

		prob_bb_base = 0.9930308986391225
		prob_bi_base = 0.012893167460595545
		prob_ii_base = 0.0036061951403780212 

	else:
		print("ERROR: BJ DOES NOT HAVE EXPECTED VALUES!!")
		return 1

	
	n_iter = 2000000


	spin_array_initial = np.zeros((num_row, num_col))

	spin_array = np.load("spin_array_initial.npy")

	for i in range(num_row):
		for j in range(num_col):
			spin_array_initial[i][j] = spin_array[i][j]


	for bJ in bJ_arr:
	
		J = 20.0
		kT = float(J)/bJ

		for trial in range(1):

			if bJ_arr[0] == 0.4406878:
		
				prob_bb_expansion = 0.5
				prob_bi_expansion = 0.5
				prob_ii_expansion = 0.5

			elif bJ_arr[0] == 0.1:

				prob_bb_expansion = 0.2573616453849372
				prob_bi_expansion = 0.2985619550399164
				prob_ii_expansion = 0.2933017814025113

			elif bJ_arr[0] == 1.5:

				prob_bb_expansion = 0.13086994972714883
				prob_bi_expansion = 0.33836990735622247
				prob_ii_expansion = 0.010051681873343987 

			else:
				print("ERROR: BJ DOES NOT HAVE EXPECTED EXPANSION VALUES!!")
				return 1


			#initialize boundary/interior atoms list and dicts
			compute_all_atom_types()

			E_arr, E_total, accept_rate, accept_rate_nonzero_energy\
			 = Equilibrate_T(kT, num_row, num_col, n_iter, J)
		
			iter_arr = np.linspace(1,len(E_arr),len(E_arr))

			E_arr_v_iter = np.array([np.array(iter_arr) , np.array(E_arr)])
			E_arr_v_iter = E_arr_v_iter.T.tolist()

			
			fields = ["iter_num", "E"]
			csvfilename = "E_arr_bJ_" + str(bJ) + ".csv"

			with open(csvfilename, 'w') as csvfile: 
				#creating a csv writer object 
				csvwriter = csv.writer(csvfile)
				#writing the fields 
				csvwriter.writerow(fields)
				csvwriter.writerows(E_arr_v_iter)

			np.save("spin_array_bJ_" + str(bJ), spin_array)
			
			
			boundary_atoms_dict_specie1 = {}
			boundary_atoms_list_specie1 = []

			interior_atoms_dict_specie1 = {}
			interior_atoms_list_specie1 = []

			boundary_atoms_dict_specie2 = {}
			boundary_atoms_list_specie2 = []

			interior_atoms_dict_specie2 = {}
			interior_atoms_list_specie2 = []


			copy_arr(spin_array_initial , spin_array)
	
	
main()
