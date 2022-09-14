# 1 server in the middle -> think if using MPC then it will be multiple servers
# 2 server sends the clients (can be millions, like someones cell phone) the current global model
# 3 each client trains using the data that they have on their machines, then they send the updated model back to the server
# 4 this loops until a specified epoch or until an accuracy condition is met
# 5 Agregation is simply averaging the models -> lookup the Fedreated Averaging Algorithm


# mpc
# 1 each clent inputs their data
# 2 the output is the computation using each client's data yet the data is kept secret from each respective party
# input parties? the clents (users with cells phones who will train the model) -> unsure
# computing parties? the clients train on their phones, agregation is done by the individual computing parties
# result parties? the results will go back to the clents for more training until epoch is reached


# what is will look like
# 1 there should be multiple servers who will aggregate the global model
# 2 

#                                           --- this is the server in FL ---          --- this is each of the clients in FL ---
# Model Hiding: In the final scenario, one party has access to a trained model, while another party would like to apply that model to its own data. 
# However, the data and model need to be kept private. This can happen in cases where a model is proprietary, expensive to produce, and/or susceptible to white-box attacks, but has value to more than one party. 
# Previously, this would have required the second party to send its data to the first to apply the model, but privacy-preserving techniques can be used when the data can't be exposed.

# questions 
# who is the computational party?
# am I combining the two or doing some MPC then some FL
# data and model need to be kept private. Is data kept private using FL and model kept private using MPC? 
# need some ML background and what tensors are used for?

import crypten
import torch
import tensorflow as tf
import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

crypten.init()

arg = sys.argv[1]
print(arg)

if arg == "1":
    # Create torch tensor
    x = torch.tensor([1.0, 2.0, 3.0])

    # Encrypt x
    x_enc = crypten.cryptensor(x)

    # Decrypt x
    x_dec = x_enc.get_plain_text()   
    crypten.print(x_dec)
    # tensor([1., 2., 3.])


    # Create python list
    y = [4.0, 5.0, 6.0]

    # Encrypt x
    y_enc = crypten.cryptensor(y)

    # Decrypt x
    y_dec = y_enc.get_plain_text()
    crypten.print(y_dec)
    # tensor([4., 5., 6.])
elif arg == "2":
    #Arithmetic operations between CrypTensors and plaintext tensors
    x_enc = crypten.cryptensor([1.0, 2.0, 3.0]) 

    y = 2.0
    y_enc = crypten.cryptensor(2.0)


    # Addition -> adds 2 to each value
    z_enc1 = x_enc + y      # Public
    z_enc2 = x_enc + y_enc  # Private
    # Public  addition: tensor([3., 4., 5.])
    crypten.print("\nPublic  addition:", z_enc1.get_plain_text()) 
    # Private addition: tensor([3., 4., 5.])
    crypten.print("Private addition:", z_enc2.get_plain_text())


    # Subtraction
    z_enc1 = x_enc - y      # Public
    z_enc2 = x_enc - y_enc  # Private
    # Public  subtraction: tensor([-1.,  0.,  1.])
    crypten.print("\nPublic  subtraction:", z_enc1.get_plain_text())
    # Private subtraction: tensor([-1.,  0.,  1.])
    print("Private subtraction:", z_enc2.get_plain_text())

    # Multiplication
    z_enc1 = x_enc * y      # Public
    z_enc2 = x_enc * y_enc  # Private
    # Public  multiplication: tensor([2., 4., 6.])
    print("\nPublic  multiplication:", z_enc1.get_plain_text())
    # Private multiplication: tensor([2., 4., 6.])
    print("Private multiplication:", z_enc2.get_plain_text())

    # Division
    z_enc1 = x_enc / y      # Public
    z_enc2 = x_enc / y_enc  # Private
    # Public  division: tensor([0.5000, 1.0000, 1.5000])
    print("\nPublic  division:", z_enc1.get_plain_text())
    # Private division: tensor([0.5000, 1.0000, 1.5000])
    print("Private division:", z_enc2.get_plain_text())
elif arg == "3":
    # ---- COMPARISIONS ----
    #Construct two example CrypTensors
    x_enc = crypten.cryptensor([1.0, 2.0, 3.0, 4.0, 5.0])

    y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    y_enc = crypten.cryptensor(y)

    # Print values:
    # x:  tensor([1., 2., 3., 4., 5.])
    # y:  tensor([5., 4., 3., 2., 1.])
    print("x: ", x_enc.get_plain_text())
    print("y: ", y_enc.get_plain_text())

    # Less than
    z_enc1 = x_enc < y      # Public
    z_enc2 = x_enc < y_enc  # Private
    # Public  (x < y) : tensor([1, 1, 0, 0, 0])
    # Private (x < y) : tensor([1, 1, 0, 0, 0])
    print("\nPublic  (x < y) :", z_enc1.get_plain_text())
    print("Private (x < y) :", z_enc2.get_plain_text())

    # Less than or equal
    z_enc1 = x_enc <= y      # Public
    z_enc2 = x_enc <= y_enc  # Private
    # Public  (x <= y): tensor([1, 1, 1, 0, 0])
    # Private (x <= y): tensor([1, 1, 1, 0, 0])
    print("\nPublic  (x <= y):", z_enc1.get_plain_text())
    print("Private (x <= y):", z_enc2.get_plain_text())

    # Greater than
    z_enc1 = x_enc > y      # Public
    z_enc2 = x_enc > y_enc  # Private
    # Public  (x > y) : tensor([0, 0, 0, 1, 1])
    # Private (x > y) : tensor([0, 0, 0, 1, 1])
    print("\nPublic  (x > y) :", z_enc1.get_plain_text())
    print("Private (x > y) :", z_enc2.get_plain_text())

    # Greater than or equal
    z_enc1 = x_enc >= y      # Public
    z_enc2 = x_enc >= y_enc  # Private
    # Public  (x >= y): tensor([0, 0, 1, 1, 1])
    # Private (x >= y): tensor([0, 0, 1, 1, 1])
    print("\nPublic  (x >= y):", z_enc1.get_plain_text())
    print("Private (x >= y):", z_enc2.get_plain_text())

    # Equal
    z_enc1 = x_enc == y      # Public
    z_enc2 = x_enc == y_enc  # Private
    # Public  (x == y): tensor([0, 0, 1, 0, 0])
    # Private (x == y): tensor([0, 0, 1, 0, 0])
    print("\nPublic  (x == y):", z_enc1.get_plain_text())
    print("Private (x == y):", z_enc2.get_plain_text())

    # Not Equal
    z_enc1 = x_enc != y      # Public
    z_enc2 = x_enc != y_enc  # Private
    # Public  (x != y): tensor([1, 1, 0, 1, 1])
    # Private (x != y): tensor([1, 1, 0, 1, 1])
    print("\nPublic  (x != y):", z_enc1.get_plain_text())
    print("Private (x != y):", z_enc2.get_plain_text())

elif arg == "4": 
    torch.set_printoptions(sci_mode=False)

    #Construct example input CrypTensor
    x = torch.tensor([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    x_enc = crypten.cryptensor(x)

    # Reciprocal
    z = x.reciprocal()          # Public
    z_enc = x_enc.reciprocal()  # Private
    # Public  reciprocal: tensor([10.0000,  3.3333,  2.0000,  1.0000,  0.6667,  0.5000,  0.4000])
    # Private reciprocal: tensor([10.0009,  3.3335,  2.0000,  1.0000,  0.6667,  0.5000,  0.4000])
    print("\nPublic  reciprocal:", z)
    print("Private reciprocal:", z_enc.get_plain_text())

    # Logarithm
    z = x.log()          # Public
    z_enc = x_enc.log()  # Private
    # Public  logarithm: tensor([-2.3026, -1.2040, -0.6931,  0.0000,  0.4055,  0.6931,  0.9163])
    # Private logarithm: tensor([    -2.3181,     -1.2110,     -0.6997,      0.0004,      0.4038,        
    #             0.6918,      0.9150])
    print("\nPublic  logarithm:", z)
    print("Private logarithm:", z_enc.get_plain_text())

    # Exp
    z = x.exp()          # Public
    z_enc = x_enc.exp()  # Private
    # Public  exponential: tensor([ 1.1052,  1.3499,  1.6487,  2.7183,  4.4817,  7.3891, 12.1825])       
    # Private exponential: tensor([ 1.1021,  1.3440,  1.6468,  2.7121,  4.4574,  7.3280, 12.0188])
    print("\nPublic  exponential:", z)
    print("Private exponential:", z_enc.get_plain_text())

    # Sqrt
    z = x.sqrt()          # Public
    z_enc = x_enc.sqrt()  # Private
    # Public  square root: tensor([0.3162, 0.5477, 0.7071, 1.0000, 1.2247, 1.4142, 1.5811])
    # Private square root: tensor([0.3147, 0.5477, 0.7071, 0.9989, 1.2237, 1.4141, 1.5811])
    print("\nPublic  square root:", z)
    print("Private square root:", z_enc.get_plain_text())

    # Tanh
    z = x.tanh()          # Public
    z_enc = x_enc.tanh()  # Private
    # Public  tanh: tensor([0.0997, 0.2913, 0.4621, 0.7616, 0.9051, 0.9640, 0.9866])
    # Private tanh: tensor([0.0994, 0.2914, 0.4636, 0.7636, 0.9069, 0.9652, 0.9873])
    print("\nPublic  tanh:", z)
    print("Private tanh:", z_enc.get_plain_text())

elif arg == "5": 
    x_enc = crypten.cryptensor(2.0)
    y_enc = crypten.cryptensor(4.0)

    a, b = 2, 3

    # Normal Control-flow code will raise an error
    try:
        if x_enc < y_enc:
            z = a
        else:
            z = b
    except RuntimeError as error:
        print(f"RuntimeError caught: \"{error}\"\n")

        
    # Instead use a mathematical expression
    use_a = (x_enc < y_enc)
    z_enc = use_a * a + (1 - use_a) * b
    # z: tensor(2)
    print("z:", z_enc.get_plain_text())
        
        
    # Or use the `where` function
    z_enc = crypten.where(x_enc < y_enc, a, b)
    # z: tensor(2)
    print("z:", z_enc.get_plain_text())

elif arg == "6": 
    x_enc = crypten.cryptensor([1.0, 2.0, 3.0])
    y_enc = crypten.cryptensor([4.0, 5.0, 6.0])

    # Indexing
    z_enc = x_enc[:-1]
    # tensor([1., 2.])
    print("Indexing:\n", z_enc.get_plain_text())

    # Concatenation
    z_enc = crypten.cat([x_enc, y_enc])
    # tensor([1., 2., 3., 4., 5., 6.])
    print("\nConcatenation:\n", z_enc.get_plain_text())

    # Stacking
    z_enc = crypten.stack([x_enc, y_enc])
    # tensor([[1., 2., 3.],
    #     [4., 5., 6.]])
    print('\nStacking:\n', z_enc.get_plain_text())

    # Reshaping
    w_enc = z_enc.reshape(-1, 6)
    # tensor([[1., 2., 3., 4., 5., 6.]])
    print('\nReshaping:\n', w_enc.get_plain_text())