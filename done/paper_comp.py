import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten.optim as optimizer
import crypten.nn as nn
# create model , criterion , and optimizer:

# is about 2.5–3 orders of magnitu***************************
# The results also show that increasing the batch size is an effective way to reduce inference
model_enc = nn. Sequential (
nn.Linear(sample_dim , hidden_dim ),
nn.ReLU (),
nn.Linear(hidden_dim , num_classes ),
). encrypt ()
criterion = nn. CrossEntropyLoss ()
optimizer = optimizer .SGD(
model_enc . parameters (), lr=0.1 , momentum =0.9 ,
)
# perform prediction on sample:
target_enc = crypten. cryptensor (target , src =0)
sample_enc = crypten. cryptensor (sample , src =0)
output_enc = model_enc ( sample_enc )
# perform backward pass and update parameters:
model_enc . zero_grad ()
loss_enc = criterion (output_enc , target_enc )
loss_enc.backward ()
optimizer .step ()

