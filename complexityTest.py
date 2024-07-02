from complexity import Complexity
complexity = Complexity("dataset/61_iris.arff",distance_func="default",file_type="arff")


print(type(complexity.X)) # numpy.ndarray  lista de listas em que a interna corresponde a uma linha
print(type(complexity.y)) #numpy array com os valores de todos os targets
print(type(complexity.meta)) #list com zero por cada uma das colunas 
a = """# Feature Overlap
print(complexity.F1())
print(complexity.F1v())
print(complexity.F2())
# (...)

# Instance Overlap
print(complexity.R_value())
print(complexity.deg_overlap())
print(complexity.CM())
# (...)

# Structural Overlap
print(complexity.N1())
print(complexity.T1())
print(complexity.Clust())
# (...)

# Multiresolution Overlap
print(complexity.MRCA())
print(complexity.C1())
print(complexity.purity())"""