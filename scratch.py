from neural_net_train import Neural_net

nn = Neural_net(2, (8, 8, 8, 8, 8), 1) 
nn.load_model('he_2(16, 16)1')
print(nn.predict([[0.2,0.16]]))