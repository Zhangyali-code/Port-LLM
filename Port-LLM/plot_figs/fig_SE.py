import torch
import matplotlib.pyplot as plt

'''
Take the SE data acquired with a 32×8 antenna on the base station and a user speed of 120km/h as an example.
'''
#####################################10UE###############################################
##32-8/V120
a_Transformer = torch.tensor([-21.9261, -21.4367, -22.2830, -22.0958, -22.5221, -22.4892, -20.4557,
        -21.5215])
a_our_model = torch.tensor([-23.4674, -23.3546, -23.4186, -24.2023, -23.5942, -23.5488, -22.8173,
        -23.3796])
a_GRU = torch.tensor([-19.3949, -18.9204, -19.9340, -20.0763, -18.9012, -18.9066, -18.8573,
        -19.0715])
a_LSTM = torch.tensor([-18.0899, -16.7807, -17.0232, -16.9225, -17.2894, -17.2604, -16.5620,
        -16.5872])
a_RNN = torch.tensor([-16.8286, -16.8852, -17.0296, -16.6148, -16.4046, -16.7904, -15.9830,
        -15.8312])

# Convert tensor to numpy array
a_our_model_np = a_our_model.numpy()
a_Transformer_np = a_Transformer.numpy()
a_LSTM_np = a_LSTM.numpy()
a_GRU_np = a_GRU.numpy()
a_RNN_np = a_RNN.numpy()

# create data points for the x-axis
x = range(len(a_our_model_np))

# plot figs
plt.plot(x, a_our_model_np, marker='o', linestyle='-', color='b', label='Port-LLM')
plt.plot(x, a_Transformer_np, marker='s', linestyle='--', color='g', label='Transformer')
plt.plot(x, a_LSTM_np, marker='^', linestyle=':', color='r', label='LSTM')
plt.plot(x, a_GRU_np, marker='v', linestyle='-.', color='m', label='GRU')
plt.plot(x, a_RNN_np, marker='*', linestyle='-', color='k', label='RNN')

#
plt.xlabel('Prediction Time Step')
plt.ylabel(r'NMSE$_{\mathrm{v}}$ (dB)')
# plt.title('NMSE over Prediction Time Steps for Different Models')

#
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))

#
plt.grid(True)

#
plt.show()