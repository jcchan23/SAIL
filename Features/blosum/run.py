blosum_dict = dict()
with open(f'blosum62.txt', 'r') as f:
    lines = f.readlines()[7:]
    for i in range(20):
        line = lines[i].strip().split()
        blosum_dict[line[0]] = [int(num) for num in line[1:21]]
        
sequence = "MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASAFIA"
feature = [blosum_dict[amino] for amino in sequence]
print(feature)
