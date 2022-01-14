aaphy7_dict = dict()
with open(f'./Features/aaphy7/aaphy7.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        aaphy7_dict[line[0]] = [float(num) for num in line[1:]]

sequence = "MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASAFIA"
feature = [aaphy7_dict[amino] for amino in sequence]
print(feature)
