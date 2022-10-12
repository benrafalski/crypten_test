# g = 6
# a = 2
# b = 4
# p = 11

# print(f'g={g}, a={a}, b={b}, p={p}')
# for i in range(5):
#     # print(f'Alice sends Bob: {pow(g,a+i,p)}')
#     # print(f'Bob sends Alice: {pow(g,b+i,p)}')
#     # print(f'The shared key: {pow(g,((a+i)*(b+i)),p)}\n\n')
#     print(f'{pow(g,a+i,p)}, {pow(g,b+i,p)} = {pow(g,((a+i)*(b+i)),p)}')

# g = 2 
import hashlib


p = 192962964156356506585560924172243284101 

# # observations of form (A, B) 
# o0 = (88645651316008659773893764520852274720, 176550691663072923290091823385758668004)
# o1 = (177291302632017319547787529041704549440, 160138419169789339994622722599274051907)
# o2 = (161619641107678132510014133911165814779, 127313874183222173403684521026304819713)
# o3 = (130276318058999758434467343650088345457, 61664784210087840221808117880366355325)
# o4 = (67589671961643010283373763127933406813, 123329568420175680443616235760732710650)

# # leaked shared key:
# K_3 = 150684264872702255826248413273287172053

A = 88645651316008659773893764520852274720
B = 176550691663072923290091823385758668004
g = 2
num = pow(A,2)*pow(B,2)*pow(g,4)
print(f'num = {num}\n')

e1 = num/p
r1 = num%p
print(f'e1 = {e1}')
print(f'r1 = {r1}\n')

e2 = p/r1
r2 = p%r1
print(f'e2 = {e2}')
print(f'r2 = {r2}\n')

e3 = r1/r2
r3 = r1%r2
print(f'e2 = {e3}')
print(f'r2 = {r3}\n')

e4 = r2/r3
r4 = r2%r3
print(f'e2 = {e4}')
print(f'r2 = {r4}\n')

e5 = r3/r4 
r5 = r3%r4
print(f'e2 = {e5}')
print(f'r2 = {r5}\n')

# Global Variables
x, y = 0, 1
 
# Function for extended Euclidean Algorithm
 
 
def gcdExtended(a, b):
    global x, y
 
    # Base Case
    if (a == 0):
        x = 0
        y = 1
        return b
 
    # To store results of recursive call
    gcd = gcdExtended(b % a, a)
    x1 = x
    y1 = y
 
    # Update x and y using results of recursive
    # call
    x = y1 - (b // a) * x1
    y = x1
 
    return gcd
 
 
def modInverse(A, M):
 
    g = gcdExtended(A, M)
    if (g != 1):
        print("Inverse doesn't exist")
 
    else:
 
        # m is added to handle negative x
        res = (x % M + M) % M
        print("Modular multiplicative inverse is ", res)
 
 
# Driver Code
if __name__ == "__main__":
    A = 3918985890479849183445512640271639717521344610333414198616637660280661469667200653620175590466555338967995477861458769773447264540221745392043452589670400
    M = 192962964156356506585560924172243284101
 
    # Function call
    modInverse(A, M)
    print((74660644131531435521091870151437626230*150684264872702255826248413273287172053)%192962964156356506585560924172243284101)

    ans = (74660644131531435521091870151437626230*150684264872702255826248413273287172053)%192962964156356506585560924172243284101
    print(hashlib.sha256(str(ans).encode()).digest().hex())
    print('bdeebe226af13c1c98214bfd331eb939d396fdffdcc3832575bd32d97d1d4f47')





