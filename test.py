g = 7
a = (int)(1570039966)
b = (int)(505212543)
p = (int)(2330587351)

for i in range(100):
    print(f'Alice sends Bob: {pow(g,a+i,p)}')
    print(f'Bob sends Alice: {pow(g,b+i,p)}')
    print(f'The shared key: {pow(g,((a+i)*(b+i)),p)}\n\n')